"""
Test / Inference script for the MidiBERT Reward Model.

This script:
  1. Validates that the trained reward model checkpoint exists
  2. Loads the fine-tuned model (LoRA adapters + Reward Head MLP)
  3. Loads the preference dataset (originals + corrupted)
  4. Scores every sequence and computes pairwise accuracy
  5. Writes a detailed results report to `test_results.txt`

Prerequisites:
  - A trained reward model checkpoint (from finetune_reward_model.py)
    e.g.  reward_model_output/best_reward_model.pt
  - The preference dataset directory containing originals.npy & corrupted.npy

Usage:
    python test_reward_model.py \
        --reward_ckpt reward_model_output/best_reward_model.pt \
        --data_dir midi_dataset
"""

import os
import sys
import math
import random
import pickle
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Import model classes from the training script ──────────────────────────────
from finetune_reward_model import (
    MidiBert,
    MidiBertRewardModel,
    RewardHead,
    apply_lora,
    PreferencePairDataset,
)
from torch.utils.data import DataLoader

try:
    from transformers import BertConfig
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    sys.exit(1)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def count_parameters(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_number(n):
    """Pretty-print large numbers: 85,054,464 → 85.05M"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def validate_checkpoint(reward_ckpt_path):
    """
    Validate that the trained reward model checkpoint exists and is loadable.
    Exits with a clear error message if not found.
    """
    if not reward_ckpt_path:
        print("=" * 60)
        print("ERROR: No reward model checkpoint specified.")
        print("=" * 60)
        print()
        print("You must train the reward model first, then point to the")
        print("saved checkpoint. Example:")
        print()
        print("  Step 1 — Train:")
        print("    python finetune_reward_model.py \\")
        print("        --data_dir midi_dataset \\")
        print("        --ckpt_file pretrain_model.ckpt \\")
        print("        --output_dir reward_model_output")
        print()
        print("  Step 2 — Test:")
        print("    python test_reward_model.py \\")
        print("        --reward_ckpt reward_model_output/best_reward_model.pt \\")
        print("        --data_dir midi_dataset")
        print()
        sys.exit(1)

    if not os.path.exists(reward_ckpt_path):
        print("=" * 60)
        print(f"ERROR: Checkpoint not found: {reward_ckpt_path}")
        print("=" * 60)
        print()
        print("The trained reward model checkpoint does not exist at the")
        print("specified path. Make sure you have:")
        print()
        print("  1. Run the fine-tuning script first:")
        print("       python finetune_reward_model.py --output_dir reward_model_output")
        print()
        print("  2. Copied/downloaded the checkpoint to this machine.")
        print()
        print("Expected file: best_reward_model.pt or final_reward_model.pt")
        print(f"Looked at:     {os.path.abspath(reward_ckpt_path)}")
        print()
        sys.exit(1)

    # Quick sanity check: try loading the file header
    try:
        ckpt = torch.load(reward_ckpt_path, map_location='cpu', weights_only=False)
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in ckpt:
                print(f"ERROR: Checkpoint is missing required key '{key}'.")
                print(f"       This may not be a valid reward model checkpoint.")
                print(f"       File: {reward_ckpt_path}")
                sys.exit(1)
        return ckpt
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        sys.exit(1)


def load_trained_reward_model(ckpt, reward_ckpt_path, device='cpu'):
    """
    Load a trained reward model from a validated checkpoint dict.

    The checkpoint contains the LoRA config, reward head weights, and
    the e2w/w2e dictionaries needed to reconstruct the model.
    """
    saved_args = ckpt.get('args', {})
    e2w = ckpt.get('e2w')
    w2e = ckpt.get('w2e')

    # Reconstruct e2w/w2e from default if not in checkpoint
    if e2w is None or w2e is None:
        print("  [INFO] Checkpoint missing e2w/w2e, using default CP dictionary")
        from generate_preference_dataset import CPTokenizer
        tok = CPTokenizer()
        e2w, w2e = tok.e2w, tok.w2e

    # Build MidiBERT base model (weights will be overwritten by checkpoint)
    config = BertConfig(
        max_position_embeddings=512,
        position_embedding_type='relative_key_query',
        hidden_size=768,
    )
    midibert = MidiBert(config, e2w, w2e)

    # Build reward model
    reward_model = MidiBertRewardModel(midibert, hidden_size=768, use_layer=-1)

    # Apply LoRA with the same config used during training
    lora_r = saved_args.get('lora_r', 8)
    lora_alpha = saved_args.get('lora_alpha', 16)
    lora_dropout = saved_args.get('lora_dropout', 0.05)
    lora_layers = saved_args.get('lora_layers', 2)

    print(f"  Applying LoRA (r={lora_r}, alpha={lora_alpha}, layers={lora_layers})")
    reward_model = apply_lora(
        reward_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        n_layers_to_tune=lora_layers,
    )

    # Load the trained state dict (base BERT + LoRA adapters + reward head)
    missing, unexpected = reward_model.load_state_dict(
        ckpt['model_state_dict'], strict=False
    )
    if missing:
        print(f"  [WARN] Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")

    epoch = ckpt.get('epoch', '?')
    val_acc = ckpt.get('val_acc', '?')
    val_loss = ckpt.get('val_loss', '?')
    print(f"  Loaded from epoch {epoch} (val_acc={val_acc}, val_loss={val_loss})")

    return reward_model, e2w, w2e, saved_args


# ─── Scoring ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_sequences(model, sequences, device, batch_size=32, desc="Scoring"):
    """
    Score an array of CP token sequences.

    Args:
        model: MidiBertRewardModel (in eval mode)
        sequences: np.ndarray of shape (N, seq_len, 4)
        device: torch device
        batch_size: inference batch size

    Returns:
        np.ndarray of shape (N,) — scalar rewards
    """
    model.eval()
    all_scores = []

    n = len(sequences)
    for start in tqdm(range(0, n, batch_size), desc=desc):
        batch = sequences[start:start + batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)
        scores = model(batch_tensor)  # (B,)
        all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores, axis=0)


@torch.no_grad()
def compute_pairwise_accuracy(model, dataloader, device, max_pairs=5000):
    """Compute accuracy on preference pairs: is r(original) > r(corrupted)?"""
    model.eval()
    correct = 0
    total = 0

    for preferred, rejected in tqdm(dataloader, desc="Pairwise accuracy"):
        preferred = preferred.to(device)
        rejected = rejected.to(device)

        r_pref = model(preferred)
        r_rej = model(rejected)

        correct += (r_pref > r_rej).sum().item()
        total += preferred.size(0)

        if total >= max_pairs:
            break

    return correct / max(total, 1), total


# ─── Main Test / Inference ─────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 0. Validate checkpoint exists BEFORE doing anything else ──
    print("\n[0/4] Validating checkpoint...")
    ckpt = validate_checkpoint(args.reward_ckpt)
    print(f"  ✓ Checkpoint found: {args.reward_ckpt}")
    print(f"  ✓ Contains keys: {list(ckpt.keys())}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    if not args.cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    elif not args.cpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"  Device: {device}")

    # ── 1. Load trained model ──
    print(f"\n[1/4] Loading trained reward model from: {args.reward_ckpt}")
    reward_model, e2w, w2e, saved_args = load_trained_reward_model(
        ckpt, args.reward_ckpt, device='cpu'
    )
    reward_model = reward_model.to(device)
    reward_model.eval()

    total_params, trainable_params = count_parameters(reward_model)
    print(f"  Total parameters:     {format_number(total_params)} ({total_params:,})")
    print(f"  Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")

    # ── 2. Load dataset ──
    print(f"\n[2/4] Loading dataset from: {args.data_dir}")
    originals_path = os.path.join(args.data_dir, 'originals.npy')
    corrupted_path = os.path.join(args.data_dir, 'corrupted.npy')

    if not os.path.exists(originals_path) or not os.path.exists(corrupted_path):
        print(f"ERROR: Dataset files not found in {args.data_dir}")
        print("Run generate_preference_dataset.py first!")
        sys.exit(1)

    originals = np.load(originals_path, allow_pickle=True)
    corrupted = np.load(corrupted_path, allow_pickle=True)
    print(f"  Originals:  {originals.shape}")
    print(f"  Corrupted:  {corrupted.shape}")

    # ── 3. Score all sequences ──
    print(f"\n[3/4] Scoring sequences (batch_size={args.batch_size})...")
    orig_scores = score_sequences(
        reward_model, originals, device,
        batch_size=args.batch_size, desc="Originals"
    )
    corr_scores = score_sequences(
        reward_model, corrupted, device,
        batch_size=args.batch_size, desc="Corrupted"
    )

    # ── 4. Compute metrics & pairwise accuracy ──
    print("\n[4/4] Computing metrics...")

    # Per-sequence comparison (matched pairs: original_i vs corrupted_i)
    matched_correct = (orig_scores > corr_scores).sum()
    matched_acc = matched_correct / len(orig_scores)

    # Full pairwise accuracy on a subset
    dataset = PreferencePairDataset(originals_path, corrupted_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    pairwise_acc, n_tested = compute_pairwise_accuracy(
        reward_model, loader, device, max_pairs=args.max_test_pairs
    )

    # ── Collect statistics ──
    orig_mean, orig_std = orig_scores.mean(), orig_scores.std()
    corr_mean, corr_std = corr_scores.mean(), corr_scores.std()
    margin = orig_mean - corr_mean

    # ── Build the report ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 72

    report_lines = [
        separator,
        "  MidiBERT Reward Model — Test / Inference Report",
        separator,
        "",
        f"Timestamp:       {timestamp}",
        f"Device:          {device}",
        f"Checkpoint:      {args.reward_ckpt}",
        f"Training epoch:  {saved_args.get('epochs', '?')}",
        f"Training lr:     {saved_args.get('lr', '?')}",
        f"LoRA rank:       {saved_args.get('lora_r', '?')}",
        f"LoRA layers:     {saved_args.get('lora_layers', '?')}",
        "",
        f"{'─' * 40}",
        "MODEL ARCHITECTURE",
        f"{'─' * 40}",
        f"  Total parameters:      {format_number(total_params):>10s}  ({total_params:,})",
        f"  Trainable parameters:  {format_number(trainable_params):>10s}  ({trainable_params:,})",
        f"  Trainable ratio:       {trainable_params / max(total_params, 1) * 100:.2f}%",
        "",
        f"  Input shape:           (B, 512, 4)",
        f"  Embedding dims:        4 × 256 = 1024 → projected to 768",
        f"  Transformer layers:    12",
        f"  Attention heads:       12",
        f"  Hidden size:           768",
        f"  Reward head:           768 → 256 → 64 → 1",
        "",
        f"{'─' * 40}",
        "DATASET",
        f"{'─' * 40}",
        f"  Originals shape:       {originals.shape}",
        f"  Corrupted shape:       {corrupted.shape}",
        f"  Possible pairs (n²):   {len(originals) * len(corrupted):,}",
        "",
        f"{'─' * 40}",
        "REWARD SCORES (individual sequences)",
        f"{'─' * 40}",
        f"  Originals  — mean: {orig_mean:+.6f}  std: {orig_std:.6f}",
        f"  Corrupted  — mean: {corr_mean:+.6f}  std: {corr_std:.6f}",
        f"  Margin (orig − corr):  {margin:+.6f}",
        "",
    ]

    # Top-5 and bottom-5 originals
    sorted_orig = np.argsort(orig_scores)
    report_lines.append(f"  Top-5 highest-scoring originals:")
    for idx in sorted_orig[-5:][::-1]:
        report_lines.append(f"    idx={idx:>4d}  reward={orig_scores[idx]:+.6f}")
    report_lines.append(f"  Bottom-5 lowest-scoring originals:")
    for idx in sorted_orig[:5]:
        report_lines.append(f"    idx={idx:>4d}  reward={orig_scores[idx]:+.6f}")

    report_lines += [
        "",
        f"{'─' * 40}",
        "PAIRWISE ACCURACY",
        f"{'─' * 40}",
        f"  Matched pairs (orig_i vs corr_i):",
        f"    Correct: {matched_correct}/{len(orig_scores)}  Accuracy: {matched_acc:.4f}",
        "",
        f"  Random cross-pairs (full n² subset):",
        f"    Tested:  {n_tested:,}  Accuracy: {pairwise_acc:.4f}",
        "",
    ]

    # Interpretation
    report_lines.append(f"{'─' * 40}")
    report_lines.append("INTERPRETATION")
    report_lines.append(f"{'─' * 40}")

    if pairwise_acc > 0.9:
        verdict = "EXCELLENT — Model clearly distinguishes real from corrupted music."
    elif pairwise_acc > 0.7:
        verdict = "GOOD — Model has learned meaningful preferences."
    elif pairwise_acc > 0.55:
        verdict = "FAIR — Model shows some preference signal, more training may help."
    else:
        verdict = "POOR — Model is near chance; check training or data."

    report_lines.append(f"  {verdict}")
    report_lines += [
        "",
        separator,
        "  End of Report",
        separator,
        "",
    ]

    report_text = "\n".join(report_lines)

    # ── Write to file ──
    output_path = os.path.join(args.output_dir, 'test_results.txt')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)

    # Also print to console
    print("\n" + report_text)
    print(f"Results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test / Inference for MidiBERT Reward Model'
    )

    # Paths
    parser.add_argument('--reward_ckpt', type=str, required=True,
                        help='Path to trained reward model checkpoint '
                             '(e.g. reward_model_output/best_reward_model.pt). '
                             'This is REQUIRED — train the model first.')
    parser.add_argument('--data_dir', type=str, default='midi_dataset',
                        help='Directory with originals.npy and corrupted.npy')
    parser.add_argument('--output_dir', type=str, default='test_output',
                        help='Directory to save test_results.txt')

    # Inference
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Inference batch size')
    parser.add_argument('--max_test_pairs', type=int, default=5000,
                        help='Max preference pairs to test for pairwise accuracy')

    # Misc
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


if __name__ == '__main__':
    main()
