# MidiBERT-Piano Preference Learning Pipeline

A complete, end-to-end preference learning pipeline built on top of [MidiBERT-Piano](https://github.com/wazenmai/MIDI-BERT). The framework trains a **Reward Model** for symbolic music using synthetic preference data, enabling downstream RLHF alignment of music generation models.

## Overview

The pipeline consists of three components:

| Script | Purpose |
|--------|---------|
| `generate_preference_dataset.py` | Tokenizes MAESTRO MIDI files into Compound Word (CP) representation and generates preference pairs via targeted corruption strategies |
| `finetune_reward_model.py` | Fine-tunes MidiBERT as a scalar reward model using LoRA (PEFT) and Bradley–Terry preference loss |
| `test_reward_model.py` | Runs inference on the trained reward model, scores sequences, computes pairwise accuracy, and saves a detailed report |

## Architecture

```
Input: (B, 512, 4) CP tokens
    ↓  4 × Embedding(vocab, 256) → concat → (B, 512, 1024)
    ↓  Linear projection → (B, 512, 768)
    ↓  12-layer BERT Transformer (LoRA on layers 10–11, Q & V)
    ↓  Attention-masked mean pooling → (B, 768)
    ↓  Reward Head MLP: 768 → 256 → 64 → 1
Output: (B,) scalar rewards
```

See [`model_dimensions.txt`](model_dimensions.txt) for full architectural details.

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repo
git clone https://github.com/rimraf-adi/music-finetune.git
cd music-finetune

# Install dependencies
uv sync
```

### Prerequisites (not tracked in git)

Download these and place them in the project root:

- **MAESTRO v3.0.0** — extract to `maestro-v3.0.0/` ([download](https://magenta.tensorflow.org/datasets/maestro))
- **MidiBERT checkpoint** — save as `pretrain_model.ckpt` ([from MidiBERT repo](https://github.com/wazenmai/MIDI-BERT))

## Usage

### 1. Generate Preference Dataset

```bash
uv run python generate_preference_dataset.py \
    --midi_dir maestro-v3.0.0 \
    --output_dir midi_dataset \
    --max_seq_len 512
```

This tokenizes MIDI files and applies five corruption strategies (bar shuffling, pitch/position/duration randomization, note dropout) to produce `originals.npy` and `corrupted.npy`.

### 2. Fine-tune Reward Model

```bash
uv run python finetune_reward_model.py \
    --data_dir midi_dataset \
    --ckpt_file pretrain_model.ckpt \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-4 \
    --lora_r 8 \
    --lora_layers 2 \
    --output_dir reward_model_output
```

Trains MidiBERT + LoRA adapters + Reward Head with Bradley–Terry loss. Saves the best checkpoint to `reward_model_output/best_reward_model.pt`.

### 3. Test / Inference

```bash
# Sanity check with untrained model
uv run python test_reward_model.py --data_dir midi_dataset --cpu

# Test a trained checkpoint
uv run python test_reward_model.py \
    --data_dir midi_dataset \
    --reward_ckpt reward_model_output/best_reward_model.pt
```

Results are saved to `test_output/test_results.txt`.

## Report

The LaTeX report (`report.tex`) provides full theoretical context, architectural motivation, and implementation details. Compile with:

```bash
pdflatex report.tex
```

## Project Structure

```
music-finetune/
├── generate_preference_dataset.py   # Stage 1: MIDI → CP tokens → corruption pairs
├── finetune_reward_model.py         # Stage 2: LoRA fine-tuning with Bradley–Terry loss
├── test_reward_model.py             # Stage 3: Inference & evaluation
├── model_dimensions.txt             # Full architectural dimensions reference
├── report.tex                       # LaTeX report with theory & implementation
├── report.pdf                       # Compiled report
├── pyproject.toml                   # Project metadata & dependencies
└── .gitignore
```

## Key Design Decisions

- **Compound Word (CP) tokenization** — Each token is a 4-tuple `[Bar, Position, Pitch, Duration]`, reducing sequence length vs. flat MIDI-like representations
- **LoRA (rank 8)** on the last 2 transformer layers — keeps ~99.5% of parameters frozen, preventing catastrophic forgetting
- **n² pair generation** — every original paired with every corruption for maximum contrastive diversity
- **Attention-masked mean pooling** — correctly excludes padding tokens from the aggregate representation

## References

- Chou et al., "MidiBERT-Piano: Large-scale pre-training for symbolic music understanding," 2021
- Hu et al., "LoRA: Low-rank adaptation of large language models," ICLR 2022
- Hawthorne et al., "Enabling factorized piano music modeling and generation with the MAESTRO dataset," ICLR 2019
- Bradley & Terry, "Rank analysis of incomplete block designs," Biometrika 1952
