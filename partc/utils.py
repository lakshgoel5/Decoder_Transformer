def dummy_function():
    pass

def collate_fn(batch):
    PAD_ID = 0  # Assume 0 is the padding token ID
    # raise NotImplementedError("Implement collate_fn as described in assignment document")

    input_ids_list = [b["input_ids"] for b in batch]
    attention_mask_list = [b["attention_mask"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    # https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=PAD_ID
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list, batch_first=True, padding_value=PAD_ID
    )

    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels
    }