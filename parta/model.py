import torch
import torch.nn as nn
from typing import Any, Dict, List

# x.unsqueeze(dim) -> adds a dimension of size 1 at the given dimension
# x.unsqueeze(0) -> (1, L, d_model)
# x.unsqueeze(1) -> (L, 1, d_model)
# x.unsqueeze(2) -> (L, d_model, 1)

# x.squeeze(dim) -> removes the dimension of size 1 at the given dimension
# x.squeeze() -> removes all dimensions of size 1
# e.g., (1, L, d_model).squeeze(0) -> (L, d_model)

DEBUG = False

class TransformerBlock(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    def multihead(self, x: torch.Tensor, attention_mask: torch.Tensor, head_idx: int) -> torch.Tensor:
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

        q = self.W_Q[head_idx - 1](x) # Query
        k = self.W_K[head_idx - 1](x) # Key
        v = self.W_V[head_idx - 1](x) # Value

        # Find alpha_i_j
        S = (q @ k.transpose(1,2)) / (self.config["d_head"] ** 0.5) # Now floatint point
        # Weight Matrix of q_i * k_j where each row is for a word
        # How much token i should look at token j (as query is of i)

        if DEBUG:
            print("[DIM][HEAD] q:", q.shape)
            print("[DIM][HEAD] k:", k.shape)
            print("[DIM][HEAD] v:", v.shape)
            print("[DIM][HEAD] S:", S.shape)
            print("[DIM][HEAD] attention_mask:", attention_mask.shape)


        if self.config["mode"] == "tanh-clipped":
            tau = self.config["tau"]
            S = tau * torch.tanh(S)

        # Apply Padding and causal masking
        B, L, _ = x.shape
        causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        pad = (attention_mask == 0).float() * float("-inf")
        pad = torch.nan_to_num(pad).unsqueeze(1)  # (B, 1, L)
        S = S + causal + pad

        # Softmax
        S = torch.softmax(S, dim=-1)
        # By applying softmax over dim=-1 or dim=2, PyTorch locks in a specific batch and a specific row (a single Query), looks at all the columns in that row (all the Keys), and applies the softmax function to them.

        return S @ v
        
    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (B, L, d_model)
        # return -> (B, L, d_model)

        # up -> (B, L, d_ff)
        # down -> (B, L, d_model)
        
        up = self.W_up(x)

        gelu = torch.nn.functional.gelu(up)

        down = self.W_down(gelu)

        return down
        

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # x -> (B, L, d_model)
        # attention_mask -> (B, L)
        # return -> (B, L, d_model)

        # Pre-Norm
        x_norm = layer_norm(x, self.beta_1, self.gamma_1)

        head_outputs = []
        for head_idx in range(1, self.config["n_heads"] + 1):
            head_outputs.append(self.multihead(
                x_norm, 
                attention_mask, 
                head_idx,
            ))

        # This function joins a list or tuple of tensors into a single tensor. Unlike torch.stack, it does not add a new dimension; it expands an existing one.
        z1 = self.W_O(torch.cat(head_outputs, dim=-1))

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
        self.config = config
        super().__init__()
        self.model_weights = None

        if (self.config["d_model"] % self.config["n_heads"] != 0):
            raise ValueError("d_model must be divisible by n_heads")

        # nn.ModuleList is ensuring that all n_layers of your TransformerBlock are properly recognized by PyTorch
        self.blocks = nn.ModuleList([
            TransformerBlock(config, l + 1) for l in range(config["n_layers"])
        ])

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
        self.model_weights["W_vocab"] = nn.Parameter(weights["W_vocab"].T)
        self.model_weights["W_devocab"] = nn.Parameter(weights["W_devocab"])

        num_layers = self.config["n_layers"]
        num_heads = self.config["n_heads"]

        self.model_weights["beta_final"] = nn.Parameter(weights["beta_final"])
        self.model_weights["gamma_final"] = nn.Parameter(weights["gamma_final"])

        for l, block in enumerate(self.blocks, start=1):

            block.W_Q = nn.ModuleList()
            block.W_K = nn.ModuleList()
            block.W_V = nn.ModuleList()

            for h in range(1, num_heads + 1):
                # self.model_weights[f"W_{l}_Q_{h}"] = nn.Parameter(weights[f"W_{l}_Q_{h}"].T)
                w = weights[f"W_{l}_Q_{h}"]
                linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
                linear.weight = nn.Parameter(w)
                block.W_Q.append(linear)

                # self.model_weights[f"W_{l}_K_{h}"] = nn.Parameter(weights[f"W_{l}_K_{h}"].T)
                w = weights[f"W_{l}_K_{h}"]
                linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
                linear.weight = nn.Parameter(w)
                block.W_K.append(linear)

                # self.model_weights[f"W_{l}_V_{h}"] = nn.Parameter(weights[f"W_{l}_V_{h}"].T)
                w = weights[f"W_{l}_V_{h}"]
                linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
                linear.weight = nn.Parameter(w)
                block.W_V.append(linear)

            # self.model_weights[f"W_{l}_O"] = nn.Parameter(weights[f"W_{l}_O"].T)
            w_o = weights[f"W_{l}_O"]
            block.W_O = nn.Linear(w_o.shape[1], w_o.shape[0], bias=False)
            block.W_O.weight = nn.Parameter(w_o)

            w_up = weights[f"W_{l}_up"]
            w_down = weights[f"W_{l}_down"]
            block.W_up = nn.Linear(w_up.shape[0], w_up.shape[1], bias=True)
            block.W_down = nn.Linear(w_down.shape[0], w_down.shape[1], bias=True)
            block.W_up.weight = nn.Parameter(w_up.T)
            block.W_down.weight = nn.Parameter(w_down.T)
            block.W_up.bias = nn.Parameter(weights[f"b_{l}_up"])
            block.W_down.bias = nn.Parameter(weights[f"b_{l}_down"])

            block.beta_1 = nn.Parameter(weights[f"beta_{l}_1"])
            block.beta_2 = nn.Parameter(weights[f"beta_{l}_2"])
            block.gamma_1 = nn.Parameter(weights[f"gamma_{l}_1"])
            block.gamma_2 = nn.Parameter(weights[f"gamma_{l}_2"])

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

        if DEBUG:
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
        X = self.model_weights["W_vocab"][input_ids] # (B, L, d_model)

        # Get Positional Encoding
        X = X + self.positional_enc(input_ids).unsqueeze(0) # (B, L, d_model)

        if DEBUG:
            print("[DIM][FORWARD]X", X.shape)
        
        # Transformer Blocks
        for block in self.blocks:
            X = block(X, attention_mask)

        if DEBUG:
            print("[DIM][FORWARD]X", X.shape)
            print("[DIM][FORWARD]beta_final", self.model_weights["beta_final"].shape)
            print("[DIM][FORWARD]gamma_final", self.model_weights["gamma_final"].shape)
        
        X_final = layer_norm(X, self.model_weights["beta_final"], self.model_weights["gamma_final"])

        if DEBUG:
            print("[DIM][FORWARD]X_final", X_final.shape)

        # self.model_weights["W_devocab"] -> (d_model, Vocab_size)
        # X_final -> (B, L, d_model)
        logits = X_final @ self.model_weights["W_devocab"] # (B, L, Vocab_size)

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
    
