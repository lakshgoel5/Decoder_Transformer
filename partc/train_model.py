# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel

# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function  # Replace with actual utility functions as needed

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

DEBUG = False

def encode_sentence(sentence, tokenizer_path):
    """Encode a single sentence - used for multiprocessing"""
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer.encode(sentence)

def main(args):
    # raise NotImplementedError("This is a placeholder for the training script. Please implement the training logic here.")

    # Determine number of processes to use
    num_processes = cpu_count()

    # GPUs
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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
    with Pool(processes=num_processes) as pool:
        encode_func = partial(encode_sentence, tokenizer_path=args.tokenizer_path)
        encoded_corpus = list(tqdm(
            pool.imap(encode_func, corpus, chunksize=max(1, len(corpus) // (num_processes * 4))),
            total=len(corpus),
            desc="Encoding"
        ))

    # --- Initialize model ----

    path = Path("./partc/config.json")
    config = None
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    st = time.time()

    model = LanguageModel(config)

    if DEBUG:
        total = 0
        for name, param in model.named_parameters():
            print(f"{name:60s} | {str(list(param.shape)):30s} | {param.numel():,}")
            total += param.numel()

        print(f"\n[DEBUG][PARAMETERS] Total trainable parameters: {total:,}")

    model.train() # Dropout is Active; BatchNorm Updates stats
    model.to(device) # Move model to GPU

    # --- Optimizer and loss function ----
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Old code
    # outputs = []
    # bsz = 16 # Batch size
    # for st in range(0, len(input_ids), bsz):
    #     en = min(st + bsz, len(input_ids)) # End of batch
    #     batch = {
    #         "input_ids": [input_ids[i] for i in range(st, en)],
    #         "attention_mask": [torch.ones_like(input_ids[i]) for i in range(st, en)]
    #     }
    #     padded_batch = collate_fn(batch)
    #     padded_batch = {k: v.to(device) for k, v in padded_batch.items()}
    #     with torch.no_grad():
    #         logits = model(input_ids=padded_batch["input_ids"], attention_mask=padded_batch["attention_mask"])
    #     logits = logits.cpu()
    #     for i in range(en - st):
    #         outputs.append({
    #             "logits": logits[i][:len(batch["input_ids"][i])]
    #         })

    # --- Training loop ----

    # --- Run validation and model selection ---

    # --- Save best performing model ----



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
