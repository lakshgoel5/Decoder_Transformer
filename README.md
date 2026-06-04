# Hindi Language Model — BPE Tokenizer + Decoder-Only Transformer

A from-scratch implementation of a Hindi language model built in three parts: a **Transformer decoder** (Part A), a **Hindi-aware BPE tokenizer** (Part B), and an end-to-end **training pipeline** with hyperparameter tuning (Part C).

---

## Overview

| Part | What I built |
|------|-------------|
| **Part A** | Decoder-only Transformer with multi-head attention, SwiGLU FFN, RoPE/ALiBi/sinusoidal PE, and a custom `collate_fn` |
| **Part B** | BPE tokenizer from scratch with Hindi-specific matra bonding, vibhakti protection, and weighted pair scoring |
| **Part C** | Full training loop with AdamW, cosine LR scheduler with warmup, gradient accumulation, and BPC-based model selection |

---

## Part A — Transformer Language Model (`parta/model.py`)

### Architecture

A standard **decoder-only Transformer** with pre-norm residual connections, implemented entirely in PyTorch without any `nn.Transformer` shortcuts.

**Multi-Head Self-Attention:**
- All Q, K, V projections fused into single linear layers (`W_Q_all`, `W_K_all`, `W_V_all`) for efficiency.
- Causal masking + padding mask applied jointly before softmax.
- `nan_to_num` on softmax output to handle all-masked rows gracefully.

**Feed-Forward Network:**
- **SwiGLU activation** (default): `SiLU(W_gate(x)) * W_up(x)` — the same gating mechanism used in LLaMA/Mistral.
- Fallback to **GeLU** when SwiGLU is disabled.

**Positional Encoding (configurable):**
- Sinusoidal PE (default)
- **RoPE** — Rotary position embeddings applied directly to Q and K before attention
- ALiBi (stub)
- Learned PE

**Other design choices:**
- **Weight tying** between token embedding (`W_vocab`) and the output projection (`W_devocab`)
- **Xavier initialization** for linear layers; normal init for embeddings
- **QK-Norm** (optional) — LayerNorm applied to Q and K before attention scores
- `tanh-clipped` attention mode: scales attention scores by `τ * tanh(S)` to prevent attention entropy collapse

### `collate_fn`

Pads variable-length sequences to the batch maximum using `pad_sequence` and returns `input_ids`, `attention_mask`.

---

## Part B — Hindi BPE Tokenizer (`partb/bpe_tokenizer.py`)

### Why a custom tokenizer for Hindi?

Standard BPE treats every character equally, which causes two critical problems for Devanagari:

1. **Matra splitting**: `का` → `क` + `ा` — the vowel diacritic `ा` has no independent pronunciation.
2. **Nukta separation**: `ड़` → `ड` + `़` — nukta is a diacritic that modifies the consonant.

### What I implemented

**Linguistically-aware initial segmentation (`word_to_unit`):**
- Consonant + all following matras/halant-conjuncts are treated as a single atomic unit before BPE merging begins.
- Halant (`्`) glues the preceding consonant to the next one, forming conjuncts (`क्ष`, `त्र`).

**Weighted pair scoring (`get_pair_weight`):**
- Nukta bonds: weight **30×** (must not be split from consonant)
- Matra bonds: weight **5×** (strong bond with preceding consonant)
- Halant conjuncts: weight **8×**
- Vibhakti (postpositions: `ने`, `को`, `से`, `का`, `के`, `की`, `में`, ...) protected from merging: weight **0.5×**

**Protected token pre-seeding:**
- Hindi vibhaktis, common verb suffixes, plural oblique markers, and punctuation are added to the vocabulary before BPE training, guaranteeing they are never split.

**Efficient training:**
- Frequency-weighted pair counts with a `pair_to_words` index for O(1) affected-word lookups on merge.
- Min-frequency pruning (`min_freq=3`) to avoid merging noise.
- Time-limited training (175-minute wall clock guard for HPC environments).

**Encode / Decode:**
- `encode`: Applies learned merges in the recorded order using `apply_merge_order` (greedy left-to-right, earliest merge wins).
- `decode`: Concatenates tokens and strips the `Ġ` space prefix.
- `save` / `load`: Serializes full tokenizer state to JSON.

---

## Part C — Training Pipeline (`partc/train_model.py`)

### Training Setup

- **Optimizer**: AdamW with `weight_decay=0.1`
- **LR Schedule**: Cosine annealing with linear warmup (15% of total steps); minimum LR floor of `1e-5`
- **Gradient accumulation**: 4 steps (effective batch size = 64)
- **Gradient clipping**: `max_norm=1.0`
- **Parallelism**: Corpus encoding parallelized with `multiprocessing.Pool` using a persistent worker pool
- **Early stopping**: Best checkpoint saved by validation BPC; 5.8-hour wall-clock guard for HPC

### Model Selection

Evaluated every epoch on a held-out validation set. Metric: **Bits Per Character (BPC)** = `NLL_loss / (total_chars * ln(2))`. Best BPC checkpoint is saved as `final_model/best_model.pt`.

### Final Model Config (`optum.json`)

| Hyperparameter | Value |
|---------------|-------|
| `d_model` | 384 |
| `n_heads` | 8 |
| `d_head` | 48 |
| `d_ff` | 1536 |
| `n_layers` | 4 |
| `vocab_size` | 10,000 |
| `dropout` | 0.1 |
| `activation` | SwiGLU |
| `positional_enc` | Sinusoidal |
| `weight_tying` | ✅ |
| `xavier_init` | ✅ |
| `qk_norm` | ✅ |
| `learning_rate` | 0.0026 |
| `warmup` | 15% |
| `epochs` | 100 |

---

## Repository Structure

```
A2/
├── parta/
│   ├── model.py              # Transformer LM: attention, FFN, PE, collate_fn
│   └── check.py              # Sanity checks for Part A
├── partb/
│   ├── bpe_tokenizer.py      # BPE tokenizer with Hindi-aware weighted merges
│   ├── train_tokenizer.py    # Script to train and save the tokenizer
│   ├── evaluate_tokenizer.py # Compression ratio & OOV rate evaluation
│   └── final_tokenizer/      # Pre-trained tokenizer checkpoint (JSON)
├── partc/
│   ├── train_model.py        # Full training loop (AdamW, cosine LR, BPC eval)
│   ├── utils.py              # TextDataset, collate_fn, BPC computation
│   ├── optum.json            # Final (best) model config
│   ├── baseline.json         # Baseline config for comparison
│   └── final_model/          # Saved best model checkpoint
├── data/
│   ├── tokenizer_corpus.txt  # Corpus for BPE training
│   ├── train.txt             # LM training data
│   └── valid.txt             # LM validation data
├── run_parta.sh
├── run_partb.sh
└── run_partc.sh
```

---

## Running the Pipeline

### Part A — Model check
```bash
bash run_parta.sh
```

### Part B — Train BPE Tokenizer
```bash
bash run_partc.sh --train-tokenizer
# Trains on ./data/hindi_mid_corpus.txt
# Saves to ./partb/final_tokenizer/
```

Or directly:
```bash
python -m partb.train_tokenizer \
  --input_corpus_path ./data/hindi_mid_corpus.txt \
  --train_path ./data/train.txt \
  --output_tokenizer_path ./partb/final_tokenizer/ \
  --vocab_size 10000
```

### Part C — Train Language Model
```bash
bash run_partc.sh --train-model
# Trains on ./data/hindi_mid_corpus.txt
# Validates on ./data/valid_mid_corpus.txt
# Saves best checkpoint to ./partc/final_model/best_model.pt
```

Or directly:
```bash
python -m partc.train_model \
  --train_path ./data/hindi_mid_corpus.txt \
  --valid_path ./data/valid_mid_corpus.txt \
  --tokenizer_path ./partb/final_tokenizer/ \
  --output_model_path ./partc/final_model/
```

---

## Dependencies

```
torch
tqdm
```

