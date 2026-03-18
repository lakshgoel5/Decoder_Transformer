import torch
import torch.nn as nn
from typing import Any, Dict, List
import datetime

# x.unsqueeze(dim) -> adds a dimension of size 1 at the given dimension
# x.unsqueeze(0) -> (1, L, d_model)
# x.unsqueeze(1) -> (L, 1, d_model)
# x.unsqueeze(2) -> (L, d_model, 1)

# x.squeeze(dim) -> removes the dimension of size 1 at the given dimension
# x.squeeze() -> removes all dimensions of size 1
# e.g., (1, L, d_model).squeeze(0) -> (L, d_model)

# In nn layer, weights passed as (out_features, in_features)
# Bias passed as (out_features)

# In nn linera, PyTorch immediately initializes the weight matrix using Kaiming Uniform initialization by default. So the weights are random but intelligently random — not just noise.
# But custom initialization is still better

# nn.Linear(in_features, out_features)
# nn.Embedding(num_embeddings, embedding_dim)

DIM = False

WEIGHT_TIEING = True

# Initialization
XAVIER = True
NORMAL = False
MEAN_INIT_WEIGHTS = 0.0
STD_INIT_WEIGHTS = 0.02
MEAN_INIT_EMBEDDING = 0.0
STD_INIT_EMBEDDING = 0.02

# PE
ROPE = False
ALIBI = False
LEARNED_PE = False

# Activation in FFN
SWIGLU = True

class TransformerBlock(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        d_model = config["d_model"]
        n_heads = config["n_heads"]
        d_head = config["d_head"]  
        d_ff = config["d_ff"]

        # Q,K,V, merging heads
        self.W_Q_all = nn.Linear(d_model, n_heads * d_head, bias=False) # (in_features, out_features)
        self.W_K_all = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_V_all = nn.Linear(d_model, n_heads * d_head, bias=False)
        # Join all heads to give d_model
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)
        
        # Feed forward
        self.W_up = nn.Linear(d_model, d_ff, bias=True)
        self.W_down = nn.Linear(d_ff, d_model, bias=True)
        self.W_gate = nn.Linear(d_model, d_ff, bias=True) # For SwiGLU

        self.dropout = nn.Dropout(config.get("dropout", 0.1))

        self.gamma_1 = nn.Parameter(torch.ones(d_model))
        self.beta_1  = nn.Parameter(torch.zeros(d_model))
        self.gamma_2 = nn.Parameter(torch.ones(d_model))
        self.beta_2  = nn.Parameter(torch.zeros(d_model))

    def multihead(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # x -> (B, L, d_model)
        # attention_mask -> (B, L)
        # return -> (B, L, d_model)

        # Q -> (B, L, d_model) @ (d_model, d_head) -> (B, L, d_head)
        # K -> (B, L, d_model) @ (d_model, d_head) -> (B, L, d_head)
        # V -> (B, L, d_model) @ (d_model, d_head) -> (B, L, d_head)
        # QK^T -> (B, L, d_head) @ (B, d_head, L) -> (B, L, L) or say (B, L_query, L_key)
        # Softmax(QK^T / sqrt(d_head)) -> (B, L, L)
        # Softmax(QK^T / sqrt(d_head)) @ V -> (B, L, d_head)
        # Attn @ V

        B, L, _ = x.shape

        # (B, L, d_model) @ (d_model, n_heads * d_head) -> (B, L, n_heads * d_head)
        q_all = self.W_Q_all(x) # Query
        k_all = self.W_K_all(x) # Key
        v_all = self.W_V_all(x) # Value

        # (B, L, n_heads * d_head) -> (B, L, n_heads, d_head)
        q_all = q_all.view(B, L, self.config["n_heads"], self.config["d_head"])
        k_all = k_all.view(B, L, self.config["n_heads"], self.config["d_head"])
        v_all = v_all.view(B, L, self.config["n_heads"], self.config["d_head"])

        # (B, L, n_heads, d_head) -> (B, n_heads, L, d_head)
        q_all = q_all.transpose(1,2)
        k_all = k_all.transpose(1,2)
        v_all = v_all.transpose(1,2)

        # Find alpha_i_j
        # (B, n_heads, L, d_head) @ (B, n_heads, d_head, L) -> (B, n_heads, L, L)
        S = (q_all @ k_all.transpose(-2,-1)) / (self.config["d_head"] ** 0.5) # Now floatint point
        # Weight Matrix of q_i * k_j where each row is for a word
        # How much token i should look at token j (as query is of i)

        if DIM:
            print("[DIM][HEAD] q_all:", q_all.shape)
            print("[DIM][HEAD] k_all:", k_all.shape)
            print("[DIM][HEAD] v_all:", v_all.shape)
            print("[DIM][HEAD] S:", S.shape)
            print("[DIM][HEAD] attention_mask:", attention_mask.shape)


        if self.config["mode"] == "tanh-clipped":
            tau = self.config["tau"]
            S = tau * torch.tanh(S)

        # Apply Padding and causal masking
        B, L, _ = x.shape

        # In language modeling, token i is not allowed to look ahead at token i+1
        # diagonal = 1 -> Stands for "Triangle Upper". It takes that grid and zeroes out everything on the main diagonal and below it.
        # [[  0.0, -inf, -inf, -inf],
        # [0.0,  0.0, -inf, -inf],
        # [0.0,  0.0,  0.0, -inf],
        # [0.0,  0.0,  0.0,  0.0]]
        # to
        # [[False,  True,  True,  True],
        # [False, False,  True,  True],
        # [False, False, False,  True],
        # [False, False, False, False]]
        causal_mask = torch.triu(torch.ones((L, L), device=x.device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)

        # (B, L) -> (B, 1, 1, L) to broadcast across n_heads and L_query
        pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)

        # (B, n_heads, L, L)
        S = S.masked_fill(causal_mask, float("-inf"))
        S = S.masked_fill(pad_mask, float("-inf"))

        # Softmax
        S = torch.softmax(S, dim=-1)
        # By applying softmax over dim=-1 or dim=3, PyTorch locks in a specific batch and a specific row (a single Query), looks at all the columns in that row (all the Keys), and applies the softmax function to them.

        out = S @ v_all
        # (B, n_heads, L, L) @ (B, n_heads, L, d_head) -> (B, n_heads, L, d_head)

        # (B, n_heads, L, d_head) -> (B, L, n_heads, d_head)
        out = out.transpose(1,2)

        # (B, L, n_heads, d_head) -> (B, L, d_model)
        out = out.reshape(B, L, self.config["d_model"])

        return out
        
    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (B, L, d_model)
        # return -> (B, L, d_model)

        # up -> (B, L, d_ff)
        # down -> (B, L, d_model)
        
        up = self.W_up(x)

        activation = None
        if SWIGLU:
            print("[SWIGLU] Using SwiGLU activation in feed forward network\n")
            gate = torch.nn.functional.silu(self.W_gate(x))
            activation = gate * up
        else:
            activation = torch.nn.functional.gelu(up)

        down = self.W_down(activation)

        return down
        

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # x -> (B, L, d_model)
        # attention_mask -> (B, L)
        # return -> (B, L, d_model)

        # Pre-Norm
        x_norm = layer_norm(x, self.beta_1, self.gamma_1)

        # (B, L, d_model) -> (B, L, n_heads * d_head)
        # d_model is n_heads * d_head!!! Creacked it!
        head_output = self.multihead(x_norm, attention_mask)

        # This function joins a list or tuple of tensors into a single tensor. Unlike torch.stack, it does not add a new dimension; it expands an existing one.
        z1 = self.W_O(head_output)

        # Residual connection
        x = x + z1

        # Pre-Norm
        x_norm = layer_norm(x, self.beta_2, self.gamma_2)
        z2 = self.feed_forward(x_norm)

        # Residual connection
        x = x + z2

        return x
        

class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        super().__init__()
        self.config = config
        self.max_len = 2048

        if (self.config["d_model"] % self.config["n_heads"] != 0):
            raise ValueError("d_model must be divisible by n_heads")

        # nn.ModuleList is ensuring that all n_layers of your TransformerBlock are properly recognized by PyTorch
        self.blocks = nn.ModuleList([
            TransformerBlock(config, l + 1) for l in range(config["n_layers"])
        ])

        d_model = self.config["d_model"]

        self.gamma_final = nn.Parameter(torch.ones(d_model))
        self.beta_final  = nn.Parameter(torch.zeros(d_model))

        # Maps vectors to logits over vocab
        self.W_devocab = nn.Linear(d_model, config["vocab_size"], bias=False) # (in_features, out_features)

        # nn.Embedding is optimized specifically for this — instead of doing a full matrix multiply one_hot @ W, it does a direct memory lookup which is much faster.
        # Each row is a token vector
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])

        self.init_pe()

        self.init_weights()

    def init_pe(self):

        # TODO: Implement RoPE, ALiBi, Learned PE as well
        if ROPE:
            print("[INIT] Using RoPE Positional Encoding\n")

        elif ALIBI:
            print("[INIT] Using ALiBi Positional Encoding\n")

        elif LEARNED_PE:
            print("[INIT] Using Learned Positional Encoding\n")

        positions = torch.arange(self.max_len, dtype=torch.float32) #(self.max_len)        
        i = torch.arange(self.config["d_model"], dtype=torch.float32) // 2 #(d_model)
        denominator = 10000 ** (2 * i / self.config["d_model"]) # (d_model)            

        pe = positions.unsqueeze(1) / denominator.unsqueeze(0) # (L, d_model)
        pe[:, 0: :2] = torch.sin(pe[:, 0: :2])
        pe[:, 1: :2] = torch.cos(pe[:, 1: :2])

        self.register_buffer('pe', pe)

    def init_weights(self):
        # self.modules() is a method in PyTorch, typically used within a torch.nn.Module subclass, to return an iterator over all modules (layers) in a network, including the network itself and its submodules.

        if XAVIER:
            print("[INIT] Using Xavier Initialization\n")
        else:
            print("[INIT] Using Normal Initialization\n")

        for m in self.modules():
            if isinstance(m, nn.Linear):

                if XAVIER:
                    nn.init.xavier_normal_(m.weight)
                else: # Fallback
                    nn.init.normal_(m.weight, mean=MEAN_INIT_WEIGHTS, std=STD_INIT_WEIGHTS)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=MEAN_INIT_EMBEDDING, std=STD_INIT_EMBEDDING)

        # Weight tying
        # embedding normal weights dominate W_devocab
        if WEIGHT_TIEING:
            self.W_devocab.weight = self.token_embedding.weight
                

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        # raise NotImplementedError("Implement set_weights as described in assignment document")
        # Both layers and heads 1 indexed
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
        self.model_weights = nn.ParameterDict()

        # .copy_(): In PyTorch, any function that ends with an underscore (_) means it is an in-place operation. It directly replaces the existing values in memory with the new ones, rather than creating a brand-new tensor.
        self.token_embedding.weight.data.copy_(weights["W_vocab"].T) # DEBUG: Left

        self.model_weights["W_devocab"] = nn.Parameter(weights["W_devocab"]) # Done

        num_layers = self.config["n_layers"]
        num_heads = self.config["n_heads"]

        self.model_weights["beta_final"] = nn.Parameter(weights["beta_final"]) # Done
        self.model_weights["gamma_final"] = nn.Parameter(weights["gamma_final"]) # Done

        for l, block in enumerate(self.blocks, start=1):

            # for h in range(1, num_heads + 1):
            #     # self.model_weights[f"W_{l}_Q_{h}"] = nn.Parameter(weights[f"W_{l}_Q_{h}"].T)
            #     w = weights[f"W_{l}_Q_{h}"]
            #     linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
            #     linear.weight = nn.Parameter(w)
            #     block.W_Q.append(linear)

            #     # self.model_weights[f"W_{l}_K_{h}"] = nn.Parameter(weights[f"W_{l}_K_{h}"].T)
            #     w = weights[f"W_{l}_K_{h}"]
            #     linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
            #     linear.weight = nn.Parameter(w)
            #     block.W_K.append(linear)

            #     # self.model_weights[f"W_{l}_V_{h}"] = nn.Parameter(weights[f"W_{l}_V_{h}"].T)
            #     w = weights[f"W_{l}_V_{h}"]
            #     linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
            #     linear.weight = nn.Parameter(w)
            #     block.W_V.append(linear)

            # Each W_Q is (d_head, d_model)
            W_Q_stacked = torch.cat([weights[f"W_{l}_Q_{h}"] for h in range(1, num_heads + 1)], dim=0) # Stacked vertically
            W_K_stacked = torch.cat([weights[f"W_{l}_K_{h}"] for h in range(1, num_heads + 1)], dim=0)
            W_V_stacked = torch.cat([weights[f"W_{l}_V_{h}"] for h in range(1, num_heads + 1)], dim=0)

            # W_Q_stacked.shape[1] is d_model
            # W_Q_stacked.shape[0] is num_heads * d_head which is d_model
            block.W_Q_all = nn.Linear(W_Q_stacked.shape[1], W_Q_stacked.shape[0], bias=False)
            block.W_K_all = nn.Linear(W_K_stacked.shape[1], W_K_stacked.shape[0], bias=False)
            block.W_V_all = nn.Linear(W_V_stacked.shape[1], W_V_stacked.shape[0], bias=False)

            block.W_Q_all.weight = nn.Parameter(W_Q_stacked.T)  # linear(x) = x @ weight.T = x @ W_Q_stacked
            block.W_K_all.weight = nn.Parameter(W_K_stacked.T)
            block.W_V_all.weight = nn.Parameter(W_V_stacked.T)

            # self.model_weights[f"W_{l}_O"] = nn.Parameter(weights[f"W_{l}_O"].T)
            w_o = weights[f"W_{l}_O"] # Done
            block.W_O = nn.Linear(w_o.shape[1], w_o.shape[0], bias=False)
            block.W_O.weight = nn.Parameter(w_o.T)  # linear(x) = x @ weight.T = x @ w_o

            w_up = weights[f"W_{l}_up"]
            w_down = weights[f"W_{l}_down"]
            block.W_up = nn.Linear(w_up.shape[0], w_up.shape[1], bias=True) # Done
            block.W_down = nn.Linear(w_down.shape[0], w_down.shape[1], bias=True) # Done
            block.W_up.weight = nn.Parameter(w_up.T)
            block.W_down.weight = nn.Parameter(w_down.T)
            block.W_up.bias = nn.Parameter(weights[f"b_{l}_up"]) # Done
            block.W_down.bias = nn.Parameter(weights[f"b_{l}_down"]) # Done

            block.beta_1 = nn.Parameter(weights[f"beta_{l}_1"]) # Done
            block.beta_2 = nn.Parameter(weights[f"beta_{l}_2"]) # Done
            block.gamma_1 = nn.Parameter(weights[f"gamma_{l}_1"]) # Done
            block.gamma_2 = nn.Parameter(weights[f"gamma_{l}_2"]) # Done

    def positional_enc(self, input_ids: torch.Tensor) -> torch.Tensor:
        # PE(pos, 2i) = sin(pos / 10000 ^ {2i/d_model})
        # PE(pos, 2i + 1) = cos(pos / 10000 ^ {2i/d_model})
        # input_ids -> (B, L)
        # return -> (L, d_model)

        # Optimized for GPU
        B, L = input_ids.shape
        d_model = self.config["d_model"]
        device = input_ids.device

        positions = torch.arange(L, device=device) #(L)
        # positions range from 0 to L-1 (some are padded)
        
        i = torch.arange(d_model, device=device) // 2 #(d_model)
        denominator = 10000 ** (2 * i / d_model) # (d_model)            
        
        pe = positions.unsqueeze(1) / denominator.unsqueeze(0) # (L, d_model)
        pe[:, 0: :2] = torch.sin(pe[:, 0: :2])
        pe[:, 1: :2] = torch.cos(pe[:, 1: :2])

        if DIM:
            print("[DIM][PE]positions", positions.shape)
            print("[DIM][PE]denominator", denominator.shape)
            print("[DIM][PE]pe", pe.shape)
        
        return pe

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the model. The output should be a tensor of shape (T, |Vocab|).

        Parameters:
            - input_ids: A tensor of shape (batch_size, sequence_len) containing token IDs.
            - attention_mask: A tensor of shape (batch_size, sequence_len) containing 1s for valid tokens and 0s for padding.

        Returns:
            - A tensor of shape (batch_size, sequence_len, vocab_size) containing the logits for each token in the vocabulary.
            Logits are the raw, unnormalized scores output by the model, which can be converted to probabilities using a softmax function.
        """
        # raise NotImplementedError("Implement forward as described in assignment document")
        # Input embedding
        X = self.token_embedding(input_ids) # (B, L, d_model)

        B, L, _ = X.shape
        # Get Positional Encoding
        X = X + self.pe[:L, :].to(X.device).unsqueeze(0) # (B, L, d_model)

        if DIM:
            print("[DIM][FORWARD]X", X.shape)
        
        # Transformer Blocks
        for block in self.blocks:
            X = block(X, attention_mask)

        if DIM:
            print("[DIM][FORWARD]X", X.shape)
            print("[DIM][FORWARD]beta_final", self.beta_final.shape)
            print("[DIM][FORWARD]gamma_final", self.gamma_final.shape)
        
        X_final = layer_norm(X, self.beta_final, self.gamma_final)

        if DIM:
            print("[DIM][FORWARD]X_final", X_final.shape)

        # X_final -> (B, L, d_model)
        # each token has become a vector of size d_model that summarizes "what comes next?"
        logits = self.W_devocab(X_final)  # (B, L, Vocab_size)
        # x.W + B

        return logits

# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html
def layer_norm(X: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    # X -> (B, L, d_model)
    # beta -> (d_model)
    # gamma -> (d_model)
    # return -> (B, L, d_model)
    
    # Each word vector is normalized
    # Functions picks last dimension that matches "normalized_shape"
    # We have B x L independent normalization operations
    return torch.nn.functional.layer_norm(X, normalized_shape = [X.shape[-1]], weight = gamma, bias = beta)


def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    model.set_weights(weights)

    return model


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  # Assume 0 is the padding token ID
    # raise NotImplementedError("Implement collate_fn as described in assignment document")

    input_ids_list = batch["input_ids"]
    attention_mask_list = batch["attention_mask"]

    # https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=PAD_ID
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list, batch_first=True, padding_value=PAD_ID
    )

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask
    }
    
