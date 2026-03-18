from torch.utils.data import Dataset
import torch
import math


def dummy_function():
    pass

class TextDataset(Dataset): # child class
    def __init__(self, encoded_corpus, max_len=2048):
        self.encoded_corpus = encoded_corpus
        self.samples = []
        for tokens in encoded_corpus:
            if len(tokens) < 2:
                continue
            # Chunk into max_len blocks
            for i in range(0, len(tokens) - 1, max_len):
                chunk = tokens[i : i + max_len + 1]  # +1 for the label shift
                if len(chunk) >= 2:
                    self.samples.append(chunk)
        
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:],  dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def __len__(self):
        return len(self.samples)

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

def compute_loss(logits, labels):
    # cross_entropy expects 2D inputs for the data and 1D for the labels.
    # logits shape: (B, L, V)
    # labels shape: (B, L)
    # logits.view(-1, logits.size(-1)): (B*L, V)
    # labels.view(-1): (B*L,)
    # ignore_index=-100: ignore the padding tokens
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

def compute_bpc(total_loss_sum: float, total_chars: int) -> float:
    if total_chars == 0:
        return float("nan")
    return total_loss_sum / (total_chars * math.log(2.0))


def evaluate(model, dataloader, device, char_lengths):
    model.eval()
    model.to(device)

    total_loss_sum = 0.0  # sum_i (L_i * T_i)
    total_tokens = 0
    total_chars = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, labels)  # mean over non-pad tokens

            # count non-pad tokens in this batch (labels pad is -100)
            valid_tokens = (labels != -100).sum().item()

            total_loss_sum += loss.item() * valid_tokens
            total_tokens += valid_tokens

            # map batch indices back to original sequence indices to accumulate character counts
            for i in range(labels.size(0)):
                seq_idx = batch_idx * dataloader.batch_size + i
                if seq_idx < len(char_lengths):
                    total_chars += char_lengths[seq_idx]

    avg_token_loss = total_loss_sum / max(total_tokens, 1)
    bpc = compute_bpc(total_loss_sum, total_chars)

    return avg_token_loss, bpc