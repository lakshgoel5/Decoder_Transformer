# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel

# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function, collate_function, TextDataset, compute_loss, compute_bpc, evaluate  # Replace with actual utility functions as needed

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
import datetime
import math
from torch.optim.lr_scheduler import LambdaLR

import wandb

# Allowed
# DataLoader, Adam, Cross entropy, Unicodedata, Regex

# Not allowed
# torch.nn.functional.scaled_dot_product_attention is not allowed
# flash attention

# Design choice
# Vocab size

DIM = False
ADAM_W = True
FRACTION_WARMUP = 0.1

GRAD_CLIP_NORM = 1.0

COSINE_LR = True
LINEAR_LR = False

ACCUMULATION_STEPS = 4
BACKGROUND_CPUS = 1

Z_LOSS = False

QK_NORM = False

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
NUM_EPOCHS = 25
LR = 0.0005

def set_globals(config):
    global ADAM_W, FRACTION_WARMUP, GRAD_CLIP_NORM, COSINE_LR, LINEAR_LR, BATCH_SIZE, NUM_EPOCHS, LR, ACCUMULATION_STEPS, Z_LOSS, QK_NORM
    BATCH_SIZE = config.get("batch_size", 32)
    NUM_EPOCHS = config.get("num_epochs", 25)
    LR = config.get("learning_rate", 0.0005)
    
    ADAM_W = config.get("adam_w", True)
    FRACTION_WARMUP = config.get("fraction_warmup", 0.1)

    GRAD_CLIP_NORM = config.get("grad_clip_norm", 1.0)

    COSINE_LR = config.get("cosine_lr", True)
    LINEAR_LR = config.get("linear_lr", False)
    ACCUMULATION_STEPS = config.get("accumulation_steps", 1)

    Z_LOSS = config.get("z_loss", False)
    QK_NORM = config.get("qk_norm", False)

def main(args):
    # raise NotImplementedError("This is a placeholder for the training script. Please implement the training logic here.")

    # Determine number of processes to use
    num_processes = cpu_count()
    if num_processes >=2:
        BACKGROUND_CPUS = 2
    else:
        BACKGROUND_CPUS = 1

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
    train_char_lengths = []
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            corpus.append(text)
            train_char_lengths.append(len(text))
    # DEBUG
    # with open(args.train_path, 'r', encoding='utf-8') as f:
    #     corpus = f.readlines()

    # --- Load validation corpus (for BPC / selection) ----
    # Later

    # Parallel encoding of all sentences
    print("Encoding corpus in parallel...")
    with Pool(processes=num_processes, initializer=init_worker, initargs=(args.tokenizer_path,)) as pool:
        encoded_corpus = list(tqdm(
            pool.imap(encode_sentence, corpus, chunksize=max(1, len(corpus) // (num_processes * 4))),
            total=len(corpus),
            desc="Encoding"
        ))
    
    # --- Load validation corpus (for BPC / selection) ----
    print(f"Loading validation corpus from {args.valid_path}...")
    valid_corpus = []
    valid_char_lengths = []
    with open(args.valid_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            valid_corpus.append(text)
            valid_char_lengths.append(len(text))

    print("Encoding validation corpus in parallel...")
    with Pool(processes=num_processes, initializer=init_worker, initargs=(args.tokenizer_path,)) as pool:
        encoded_valid_corpus = list(tqdm(
            pool.imap(encode_sentence, valid_corpus, chunksize=max(1, len(valid_corpus) // (num_processes * 4))),
            total=len(valid_corpus),
            desc="Encoding Valid"
        ))

    # --- Initialize model ----

    path = Path(args.config_path)
    config = None
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    set_globals(config)

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
    if ADAM_W:
        print("[OPTIMIZER] Using AdamW with Weight Decay")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
        # First beta: This tracks the average of past gradients. A value of 0.9 means the optimizer relies heavily on the direction it was already going, helping it barrel through noisy batches.
        # Second beta: This tracks the average of past squared gradients to scale the learning rate for each specific weight.
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_dataset = TextDataset(encoded_corpus) # DEBUG Max Len
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_function, 
        num_workers=BACKGROUND_CPUS, # Uses background CPU cores to load data
        pin_memory=True, # Speeds up CPU-to-GPU memory transfer
        prefetch_factor=2 # Queues up batches in advance
    )
    
    valid_dataset = TextDataset(encoded_valid_corpus)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,  collate_fn=collate_function, 
        num_workers=BACKGROUND_CPUS, 
        pin_memory=True, 
        prefetch_factor=2
    )
    # a dict with 3 keys: input_ids, attention_mask, labels
    # input ids, labels comes from __getitem__()
    # attention mask comes from collate fn()

    # In validation, shuffle = False

    total_steps = NUM_EPOCHS * (len(train_dataloader) // ACCUMULATION_STEPS)
    warmup_steps = int(FRACTION_WARMUP * total_steps) # FRACTION_WARMUP of training steps for warmup

    def lr_lambda(current_step):
        lr = 0.0
        if LINEAR_LR:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps)) # Gradually increasing to 1
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))) # Linearly decay to 0 after warmup
        else:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps)) # Gradually increasing to 1
            
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            # progress goes from 0 to 1 over the course of training after warmup
            # cos(0) = 1, cos(pi) = -1, so this will decay from 1 to 0 following a cosine curve

            lr = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(lr, 0.05)

    # Tool to adjust Learning rate
    # new_lr = initial_lr * lr_lambda(epoch)
    scheduler = LambdaLR(optimizer, lr_lambda)


    # ---stats---
    training_start = time.time()
    total_tokens_processed = 0

    best_val_loss = float('inf')
    best_checkpoint = None

    # --wandb----
    wandb.init(
        project="Hindi-LLM-V100",
        name=f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",
        config=config
    )

    # --- Training loop ----
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_loss_sum = 0.0
        total_z_loss = 0.0
        total_tokens = 0

        st = time.time()

        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, labels)
            combined_loss = loss

            z_loss = torch.tensor(0.0, device=device)
            if Z_LOSS:
                valid_mask = (labels != -100)
                valid_logits = logits[valid_mask]
                # log(Z) = logsumexp(logits)
                log_z = torch.logsumexp(valid_logits, dim=-1)

                # z_loss = 10^-4 * log^2(Z)
                z_loss = (1e-4) * torch.mean(log_z ** 2)

                combined_loss += z_loss

            scaled_loss = combined_loss / ACCUMULATION_STEPS

            scaled_loss.backward() # Backprop — compute gradients for every weight


            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_dataloader):
                # Gradient Clipping # DEBUG Parameter
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                optimizer.step() # Update weights using gradients
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_z_loss += z_loss.item()

            with torch.no_grad():
                # labels == -100 are ignored positions
                mask = (labels != -100)
                batch_tokens = mask.sum().item()
                total_tokens += batch_tokens
                total_loss_sum += loss.item() * batch_tokens

            total_tokens_processed += input_ids.numel()

            wandb.log({
                "batch_loss": loss.item(),
                "z_loss": z_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch
            })

        avg_loss = total_loss / len(train_dataloader)
        ppl = torch.exp(torch.tensor(avg_loss)).item()

        train_bpc = compute_bpc(total_loss_sum, sum(train_char_lengths))

        # --- Validation ---
        val_loss, val_bpc = evaluate(model, valid_dataloader, device, valid_char_lengths)
        
        # --- Model Selection ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_bpc": val_bpc
            }
            print(f"New best model found at epoch {epoch} with val_loss: {val_loss:.4f}")

        elapsed_total = time.time() - training_start
        tokens_per_sec = total_tokens_processed / max(elapsed_total, 1e-8)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"PPL: {ppl:.2f} | Val BPC: {val_bpc:.4f} | "
            f"Train BPC: {train_bpc:.4f} | "
            f"Epoch time: {time.time()-st:.1f}s | "
            f"Total elapsed: {str(datetime.timedelta(seconds=int(elapsed_total)))} | "
            f"Tokens/sec: {tokens_per_sec:.0f}"
        )

        wandb.log({
            "avg_epoch_loss": avg_loss,
            "avg_z_loss": total_z_loss / len(train_dataloader),
            "perplexity": ppl,
            "train_bpc": train_bpc,
            "val_loss": val_loss,
            "val_bpc": val_bpc
        })

    wandb.finish()
            
    # --- Save best performing model ----
    checkpoint_path = os.path.join(args.output_model_path, "best_model.pt")
    if best_checkpoint is None:
        # fallback: save last model
        best_checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
        }
    torch.save(best_checkpoint, checkpoint_path)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    ### TODO: Remove this at submission
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config file')

    args = parser.parse_args()
    main(args)
