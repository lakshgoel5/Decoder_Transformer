import torch
import torch.nn as nn
from typing import Any, Dict, List


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
        self.model_weights["W_vocab"] = nn.Parameter(weights["W_vocab"])
        self.model_weights["W_devocab"] = nn.Parameter(weights["W_devocab"])

        num_layers = self.config["n_layers"]
        num_heads = self.config["n_heads"]

        self.model_weights["beta_final"] = nn.Parameter(weights["beta_final"])
        self.model_weights["gamma_final"] = nn.Parameter(weights["gamma_final"])

        for l in range(1, num_layers + 1):
            for h in range(1, num_heads + 1):
                self.model_weights[f"W_{l}_Q_{h}"] = nn.Parameter(weights[f"W_{l}_Q_{h}"])
                self.model_weights[f"W_{l}_K_{h}"] = nn.Parameter(weights[f"W_{l}_K_{h}"])
                self.model_weights[f"W_{l}_V_{h}"] = nn.Parameter(weights[f"W_{l}_V_{h}"])

            self.model_weights[f"W_{l}_O"] = nn.Parameter(weights[f"W_{l}_O"])

            self.model_weights[f"W_{l}_up"] = nn.Parameter(weights[f"W_{l}_up"])
            self.model_weights[f"W_{l}_down"] = nn.Parameter(weights[f"W_{l}_down"])
            self.model_weights[f"b_{l}_up"] = nn.Parameter(weights[f"b_{l}_up"])
            self.model_weights[f"b_{l}_down"] = nn.Parameter(weights[f"b_{l}_down"])

            self.model_weights[f"beta_{l}_1"] = nn.Parameter(weights[f"beta_{l}_1"])
            self.model_weights[f"beta_{l}_2"] = nn.Parameter(weights[f"beta_{l}_2"])
            self.model_weights[f"gamma_{l}_1"] = nn.Parameter(weights[f"gamma_{l}_1"])
            self.model_weights[f"gamma_{l}_2"] = nn.Parameter(weights[f"gamma_{l}_2"])      

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
        raise NotImplementedError("Implement forward as described in assignment document")



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
    
