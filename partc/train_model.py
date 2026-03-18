# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel

# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function, collate_fn, TextDataset, compute_loss  # Replace with actual utility functions as needed

# You can structure your code as you see fit as long as the CLI works as specified.
# Finally, treat this as your FINAL MODEL TRAINING SCRIPT. Do not perform hyperparameter tuning here.
# You can create separate scripts for hyperparameter tuning if needed.
from multiprocessing import Pool, cpu_count
import torch
from tqdm import tqdm
from functools import partial
import time
import os
import json
from pathlib import Path
from torch.utils.data import DataLoader

DIM = True

def init_worker(tokenizer_path):
    global _tokenizer
    _tokenizer = BPETokenizer()
    _tokenizer.load(tokenizer_path)

def encode_sentence(sentence):
    return _tokenizer.encode(sentence)

# def encode_sentence(sentence, tokenizer_path):
#     """Encode a single sentence - used for multiprocessing"""
#     tokenizer = BPETokenizer()
#     tokenizer.load(tokenizer_path)
#     return tokenizer.encode(sentence)

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 0.0005

def main(args):
    # raise NotImplementedError("This is a placeholder for the training script. Please implement the training logic here.")

    # Determine number of processes to use
    num_processes = cpu_count()

    # GPUs
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Training on {device}")

    os.makedirs(args.output_model_path, exist_ok=True)

    # --- Load tokenizer ----
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)
    print(f"Tokenizer loaded from {args.tokenizer_path}.")
    unk_id = tokenizer.get_unk_id()

    # --- Load corpus ----
    corpus = []
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(line.strip())
    # DEBUG
    # with open(args.train_path, 'r', encoding='utf-8') as f:
    #     corpus = f.readlines()

    # Parallel encoding of all sentences
    print("Encoding corpus in parallel...")
    with Pool(processes=num_processes, initializer=init_worker, initargs=(args.tokenizer_path,)) as pool:
        encoded_corpus = list(tqdm(
            pool.imap(encode_sentence, corpus, chunksize=max(1, len(corpus) // (num_processes * 4))),
            total=len(corpus),
            desc="Encoding"
        ))

    # --- Initialize model ----

    path = Path("./partc/config.json")
    config = None
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model = LanguageModel(config)

    if DIM:
        total = 0
        for name, param in model.named_parameters():
            print(f"{name:60s} | {str(list(param.shape)):30s} | {param.numel():,}")
            total += param.numel()

        print(f"\n[DEBUG][PARAMETERS] Total trainable parameters: {total:,}")

    model.train() # Dropout is Active; BatchNorm Updates stats
    model.to(device) # Move model to GPU

    # --- Optimizer and loss function ----
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_dataset = TextDataset(encoded_corpus) # DEBUG Max Len
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    # a dict with 3 keys: input_ids, attention_mask, labels
    # input ids, labels comes from __getitem__()
    # attention mask comes from collate fn()
    

    # --- Training loop ----
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        st = time.time()

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, labels)

            loss.backward() # Backprop — compute gradients for every weight

            optimizer.step() # Update weights using gradients

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Avg Loss: {avg_loss:.4f} | Time: {time.time() - st:.2f}s")

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch {epoch}/{NUM_EPOCHS} | PPL: {ppl:.4f}")
            
    # --- Run validation and model selection ---
    # Last day

    # --- Save best performing model ----
    checkpoint_path = os.path.join(args.output_model_path, "best_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": epoch,
        # "valid_loss": best_valid_loss,
    }, checkpoint_path)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
