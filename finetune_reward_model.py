"""
Finetune MIDI-BERT as a Reward Model using PEFT (LoRA).

This script:
1. Loads the pre-trained MIDI-BERT checkpoint (pretrain_model.ckpt)
2. Adds a feedforward reward head on top (768 → 256 → 1)
3. Applies LoRA to the last few transformer layers via PEFT
4. Trains with Bradley-Terry preference loss on the generated dataset
5. The reward head + LoRA adapters are fully trainable

Usage:
    python finetune_reward_model.py \
        --data_dir preference_data \
        --ckpt_file pretrain_model.ckpt \
        --epochs 10 \
        --batch_size 16 \
        --lr 2e-4 \
        --lora_r 8 \
        --lora_layers 2 \
        --output_dir reward_model_output
"""

import os
import sys
import math
import json
import pickle
import random
import re
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from transformers import BertModel, BertConfig
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    sys.exit(1)

try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    print("ERROR: peft not installed. Run: pip install peft")
    sys.exit(1)

# ─── MIDI-BERT Model (vendored from the repo) ─────────────────────────────────

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class MidiBert(nn.Module):
    """MIDI-BERT model using Compound Word (CP) representation."""

    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()

        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # Token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # Padding tokens
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
        self.mask_word_np = np.array(
            [self.e2w[etype][f'{etype} <MASK>'] for etype in self.classes],
            dtype=np.int64
        )
        self.pad_word_np = np.array(
            [self.e2w[etype][f'{etype} <PAD>'] for etype in self.classes],
            dtype=np.int64
        )

        # Embeddings for each token type
        self.word_emb = nn.ModuleList([
            Embeddings(self.n_tokens[i], self.emb_sizes[i])
            for i in range(len(self.classes))
        ])

        # Linear to merge embeddings
        self.in_linear = nn.Linear(sum(self.emb_sizes), bertConfig.d_model)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # input_ids shape: (batch, seq_len, 4)
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat(embs, dim=-1)  # (batch, seq_len, 1024)
        emb_linear = self.in_linear(embs)  # (batch, seq_len, 768)

        y = self.bert(
            inputs_embeds=emb_linear,
            attention_mask=attn_mask,
            output_hidden_states=output_hidden_states
        )
        return y

    def get_rand_tok(self):
        return np.array([
            random.choice(range(c)) for c in self.n_tokens
        ])


# ─── Reward Head ───────────────────────────────────────────────────────────────

class RewardHead(nn.Module):
    """MLP that maps pooled BERT output to a scalar reward."""

    def __init__(self, hidden_size, intermediate_size=256, second_layer_size=64, dropout_rate=0.1):
        super().__init__()
        self.pool_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_size, second_layer_size),
            nn.GELU(),
            nn.Linear(second_layer_size, 1),
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            rewards: (batch,) — scalar per sequence
        """
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        pooled = self.pool_norm(pooled)  # (batch, hidden_size)
        return self.head(pooled).squeeze(-1)  # (batch,)


# ─── Reward Model (MIDI-BERT + Reward Head) ───────────────────────────────────

class MidiBertRewardModel(nn.Module):
    """
    MIDI-BERT encoder with a reward head for preference learning.
    """

    def __init__(self, midibert: MidiBert, hidden_size: int = 768,
                 use_layer: int = -1, intermediate_size: int = 256,
                 second_layer_size: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.midibert = midibert
        self.reward_head = RewardHead(hidden_size, intermediate_size, second_layer_size, dropout_rate)
        self.use_layer = use_layer  # which hidden state layer to use (-1 = last)

    def _make_attention_mask(self, input_ids):
        """Create attention mask: 1 where not padding, 0 where padding."""
        # Padding is where Bar == bar_pad_word
        bar_pad = self.midibert.bar_pad_word
        mask = (input_ids[:, :, 0] != bar_pad).long()  # (batch, seq_len)
        return mask

    def forward(self, input_ids, **kwargs):
        """
        Args:
            input_ids: (batch, seq_len, 4) — CP token indices
            **kwargs: absorbed for PEFT compatibility (e.g., attention_mask)
        Returns:
            rewards: (batch,) — scalar reward per sequence
        """
        attn_mask = self._make_attention_mask(input_ids)

        outputs = self.midibert(
            input_ids, attn_mask=attn_mask, output_hidden_states=True
        )

        # Use specified hidden state layer
        hidden_states = outputs.hidden_states[self.use_layer]  # (batch, seq, 768)

        return self.reward_head(hidden_states, attn_mask)

    def get_reward(self, input_ids):
        """Convenience: get reward score without gradients."""
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PreferencePairDataset(Dataset):
    """Dataset of paired original and corrupted token sequences."""
    def __init__(self, originals_path, corrupted_path):
        self.originals = np.load(originals_path, allow_pickle=True)
        self.corrupted = np.load(corrupted_path, allow_pickle=True)
        assert len(self.originals) == len(self.corrupted), \
            f"Mismatch: {len(self.originals)} originals vs {len(self.corrupted)} corrupted"
        
        self.n = len(self.originals)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        preferred = torch.tensor(self.originals[idx], dtype=torch.long)
        rejected = torch.tensor(self.corrupted[idx], dtype=torch.long)
        return preferred, rejected


# ─── Training ─────────────────────────────────────────────────────────────────

def load_midibert_from_checkpoint(ckpt_path, device='cpu'):
    """
    Load a pre-trained MIDI-BERT from checkpoint.

    The checkpoint typically contains:
      - 'state_dict': model weights
      - 'e2w': event-to-word dictionary
      - 'w2e': word-to-event dictionary
    """
    print(f"  Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract dictionaries
    if 'e2w' in ckpt and 'w2e' in ckpt:
        e2w = ckpt['e2w']
        w2e = ckpt['w2e']
    else:
        # Try loading from a separate dict file
        print("  [WARN] Checkpoint missing e2w/w2e, loading default dict")
        from generate_preference_dataset import CPTokenizer
        tok = CPTokenizer()
        e2w, w2e = tok.e2w, tok.w2e

    # Build model config
    config = BertConfig(
        max_position_embeddings=512,
        position_embedding_type='relative_key_query',
        hidden_size=768,
    )

    midibert = MidiBert(config, e2w, w2e)

    # Load state dict
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    if isinstance(state_dict, dict) and not any(k.startswith('bert.') or k.startswith('word_emb') for k in state_dict):
        # Maybe the entire checkpoint IS the state dict
        pass

    # Handle key prefixes (some checkpoints use 'module.' prefix)
    cleaned = {}
    for k, v in state_dict.items():
        k_clean = k.replace('module.', '').replace('midibert.', '')
        cleaned[k_clean] = v

    # Try loading; be lenient with missing keys (we may have added/removed heads)
    missing, unexpected = midibert.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [INFO] Missing keys (expected for new heads): {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  [INFO] Unexpected keys (from pre-training heads): {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")

    return midibert, e2w, w2e


def apply_lora(model, lora_r=8, lora_alpha=16, lora_dropout=0.05,
               n_layers_to_tune=2):
    """
    Apply LoRA adapters to transformer attention query/value projections.

    For the standard 12-layer MidiBERT architecture, this follows the spec in
    model_dimensions.txt and targets layers 10 and 11.
    """
    # Get total number of layers
    total_layers = model.midibert.bert.config.num_hidden_layers

    if total_layers == 12 and n_layers_to_tune == 2:
        # Architecture spec: last 2 layers are exactly indices 10 and 11.
        target_layer_indices = [10, 11]
    else:
        target_layer_indices = list(
            range(max(0, total_layers - n_layers_to_tune), total_layers)
        )

    if not target_layer_indices:
        raise ValueError("No target layers selected for LoRA.")

    # Target: attention query and value projections in last N layers
    target_modules = []
    for layer_idx in target_layer_indices:
        target_modules.extend([
            f"midibert.bert.encoder.layer.{layer_idx}.attention.self.query",
            f"midibert.bert.encoder.layer.{layer_idx}.attention.self.value",
        ])

    hidden_size = model.midibert.hidden_size
    lora_params_per_projection = (hidden_size * lora_r) + (lora_r * hidden_size)
    expected_lora_params = len(target_modules) * lora_params_per_projection

    print(f"  LoRA target layers: {target_layer_indices}")
    print(f"  LoRA target projections: {len(target_modules)} (Q/V per layer)")
    print(f"  Expected LoRA params: {expected_lora_params:,}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=['reward_head'],  # Ensure reward head is fully trainable
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch using Bradley-Terry preference loss."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_margin = 0.0
    total_margin_sq = 0.0
    total_pref_prob = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for batch_idx, (preferred, rejected) in enumerate(pbar):
        preferred = preferred.to(device)  # (B, seq_len, 4)
        rejected = rejected.to(device)

        # Forward pass
        r_preferred = model(preferred)  # (B,)
        r_rejected = model(rejected)    # (B,)
        margin = r_preferred - r_rejected

        # Margin-based Bradley-Terry loss
        target_margin = 1.0
        loss = -F.logsigmoid(margin - target_margin).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Metrics
        batch_size = preferred.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (r_preferred > r_rejected).sum().item()
        total_samples += batch_size
        total_margin += margin.sum().item()
        total_margin_sq += (margin ** 2).sum().item()
        total_pref_prob += torch.sigmoid(margin).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{total_correct / total_samples:.3f}',
        })

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    avg_margin = total_margin / total_samples
    margin_var = max(total_margin_sq / total_samples - (avg_margin ** 2), 0.0)
    avg_pref_prob = total_pref_prob / total_samples
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'margin_mean': avg_margin,
        'margin_std': margin_var ** 0.5,
        'preferred_prob_mean': avg_pref_prob,
        'correct': total_correct,
        'samples': total_samples,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_margin = 0.0
    total_margin_sq = 0.0
    total_pref_prob = 0.0

    for preferred, rejected in dataloader:
        preferred = preferred.to(device)
        rejected = rejected.to(device)

        r_preferred = model(preferred)
        r_rejected = model(rejected)
        margin = r_preferred - r_rejected

        target_margin = 1.0
        loss = -F.logsigmoid(margin - target_margin).mean()

        batch_size = preferred.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (r_preferred > r_rejected).sum().item()
        total_samples += batch_size
        total_margin += margin.sum().item()
        total_margin_sq += (margin ** 2).sum().item()
        total_pref_prob += torch.sigmoid(margin).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    avg_margin = total_margin / max(total_samples, 1)
    margin_var = max(total_margin_sq / max(total_samples, 1) - (avg_margin ** 2), 0.0)
    avg_pref_prob = total_pref_prob / max(total_samples, 1)
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'margin_mean': avg_margin,
        'margin_std': margin_var ** 0.5,
        'preferred_prob_mean': avg_pref_prob,
        'correct': total_correct,
        'samples': total_samples,
    }


def find_latest_checkpoint(output_dir):
    """Return the latest checkpoint_epoch_*.pt path in output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None

    latest_epoch = -1
    latest_path = None
    pattern = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")

    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if not match:
            continue
        epoch_num = int(match.group(1))
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_path = os.path.join(output_dir, name)

    return latest_path


def load_resume_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model/optimizer state from a reward-model checkpoint for resume."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' not in ckpt:
        raise KeyError("Checkpoint missing 'model_state_dict'; cannot resume training.")

    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  [WARN] Resume checkpoint missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  [WARN] Resume checkpoint unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")

    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else:
        print("  [WARN] Resume checkpoint has no optimizer_state_dict; optimizer will be fresh.")

    start_epoch = int(ckpt.get('epoch', 0))
    best_val_acc = float(ckpt.get('val_acc', 0.0))
    best_epoch = start_epoch
    return start_epoch, best_val_acc, best_epoch


def main():
    args = parse_args()

    # Load from config JSON if provided
    if args.config_json and os.path.exists(args.config_json):
        print(f"Loading hyperparameters from {args.config_json}...")
        with open(args.config_json, 'r') as f:
            config = json.load(f)
            for k, v in config.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                    print(f"  Overrode {k} = {v}")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    if not args.cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    elif not args.cpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    # ── 1. Load MIDI-BERT ──
    print("\n[1/5] Loading pre-trained MIDI-BERT...")
    midibert, e2w, w2e = load_midibert_from_checkpoint(args.ckpt_file, device='cpu')

    # ── 2. Build Reward Model ──
    print("\n[2/5] Building Reward Model...")
    reward_model = MidiBertRewardModel(
        midibert,
        hidden_size=args.hidden_size,
        use_layer=-1,  # last hidden layer
        intermediate_size=args.head_intermediate_size,
        second_layer_size=args.head_second_layer_size,
        dropout_rate=args.head_dropout_rate
    )

    # ── 3. Apply LoRA ──
    print(f"\n[3/5] Applying LoRA (r={args.lora_r}, layers={args.lora_layers})...")
    reward_model = apply_lora(
        reward_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        n_layers_to_tune=args.lora_layers,
    )
    reward_model = reward_model.to(device)

    # ── 4. Load Dataset ──
    print(f"\n[4/5] Loading preference dataset from: {args.data_dir}")
    originals_path = os.path.join(args.data_dir, 'originals.npy')
    corrupted_path = os.path.join(args.data_dir, 'corrupted.npy')

    if not os.path.exists(originals_path) or not os.path.exists(corrupted_path):
        print(f"ERROR: Dataset files not found in {args.data_dir}")
        print("Run generate_preference_dataset.py first!")
        sys.exit(1)

    full_dataset = PreferencePairDataset(originals_path, corrupted_path)
    print(f"  Total pairs: {len(full_dataset)}")

    # Train/val split
    n_total = len(full_dataset)
    
    # Cap dataset size if requested (for practical training on CPU/MPS)
    if args.max_train_pairs > 0 and n_total > args.max_train_pairs:
        n_use = args.max_train_pairs
        n_discard = n_total - n_use
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset, [n_use, n_discard],
            generator=torch.Generator().manual_seed(args.seed)
        )
        n_total = n_use
        print(f"  Capped to {n_total} pairs (--max_train_pairs)")
    
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed + 1)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    print(f"  Train: {n_train} pairs, Val: {n_val} pairs")

    # ── 5. Train ──
    print(f"\n[5/5] Training for {args.epochs} epochs...")
    os.makedirs(args.output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, reward_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = 0

    resume_path = None
    if args.resume_from:
        if args.resume_from.lower() == 'latest':
            resume_path = find_latest_checkpoint(args.output_dir)
            if resume_path is None:
                print(
                    f"ERROR: --resume_from latest specified, but no checkpoint_epoch_*.pt found in {args.output_dir}"
                )
                sys.exit(1)
        else:
            resume_path = args.resume_from

    if resume_path:
        if not os.path.exists(resume_path):
            print(f"ERROR: Resume checkpoint not found: {resume_path}")
            sys.exit(1)

        print(f"  Resuming from checkpoint: {resume_path}")
        start_epoch, best_val_acc, best_epoch = load_resume_checkpoint(
            resume_path, reward_model, optimizer, device
        )
        print(f"  Resume start epoch: {start_epoch + 1}")
        print(f"  Best val acc so far: {best_val_acc:.3f}")

        if start_epoch >= args.epochs:
            print(
                f"  Checkpoint is already at epoch {start_epoch}, which is >= --epochs {args.epochs}. Nothing to train."
            )
            return

    total_steps = len(train_loader) * args.epochs
    completed_steps = len(train_loader) * start_epoch

    # Restore scheduler position when resuming to keep LR schedule continuous.
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, last_epoch=completed_steps - 1
    )

    log_lines = []
    metrics_history = []
    log_path = os.path.join(args.output_dir, 'training_log.txt')
    metrics_path = os.path.join(args.output_dir, 'training_metrics.json')
    if start_epoch > 0 and os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_lines.extend([line.rstrip('\n') for line in f])
    if start_epoch > 0 and os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                previous_metrics = json.load(f)
            metrics_history.extend(previous_metrics.get('history', []))
        except Exception:
            pass

    no_improve_epochs = 0
    final_epoch = start_epoch
    last_train_metrics = None
    last_val_metrics = None

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            reward_model, train_loader, optimizer, scheduler, device, epoch
        )
        val_metrics = evaluate(reward_model, val_loader, device)
        last_train_metrics = train_metrics
        last_val_metrics = val_metrics
        final_epoch = epoch + 1

        log = (f"Epoch {epoch + 1}/{args.epochs} | "
               f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.3f} | "
               f"Train Margin: {train_metrics['margin_mean']:+.4f} | Train P(win): {train_metrics['preferred_prob_mean']:.3f} | "
               f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.3f} | "
               f"Val Margin: {val_metrics['margin_mean']:+.4f} | Val P(win): {val_metrics['preferred_prob_mean']:.3f}")
        print(log)
        log_lines.append(log)

        epoch_metrics = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
        }
        metrics_history.append(epoch_metrics)

        # Save best model
        is_best = val_metrics['accuracy'] > (best_val_acc + args.early_stop_min_delta)
        if is_best:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            no_improve_epochs = 0
            save_path = os.path.join(args.output_dir, 'best_reward_model.pt')
            # Save both LoRA adapters and reward head
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': reward_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'metrics': epoch_metrics,
                'best_epoch': best_epoch,
                'best_val_acc': best_val_acc,
                'e2w': e2w,
                'w2e': w2e,
                'args': vars(args),
            }, save_path)
            print(f"  * Saved best model (val_acc={val_metrics['accuracy']:.3f})")
        else:
            no_improve_epochs += 1

        # Optional periodic checkpointing
        if args.save_every_epochs > 0 and ((epoch + 1) % args.save_every_epochs == 0):
            checkpoint_path = os.path.join(
                args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': reward_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'metrics': epoch_metrics,
                'e2w': e2w,
                'w2e': w2e,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Early stopping
        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(
                f"  Early stopping after {no_improve_epochs} epochs without improvement."
            )
            break

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_reward_model.pt')
    torch.save({
        'epoch': final_epoch,
        'model_state_dict': reward_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': None if last_val_metrics is None else last_val_metrics['accuracy'],
        'val_loss': None if last_val_metrics is None else last_val_metrics['loss'],
        'metrics': None if last_val_metrics is None or last_train_metrics is None else {
            'epoch': final_epoch,
            'train': last_train_metrics,
            'val': last_val_metrics,
        },
        'metrics_history': metrics_history,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'e2w': e2w,
        'w2e': w2e,
        'args': vars(args),
    }, final_path)

    metrics_path = os.path.join(args.output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'final_epoch': final_epoch,
            'history': metrics_history,
        }, f, indent=2)

    # Save training log
    with open(log_path, 'w') as f:
        for line in log_lines:
            f.write(line + '\n')

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"{'=' * 60}")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best val acc:   {best_val_acc:.3f}")
    print(f"  Best model:     {os.path.join(args.output_dir, 'best_reward_model.pt')}")
    print(f"  Final model:    {final_path}")
    print(f"  Training log:   {log_path}")
    print(f"  Metrics JSON:   {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Finetune MIDI-BERT as reward model with PEFT (LoRA)'
    )

    # Paths
    parser.add_argument('--data_dir', type=str, default='preference_data',
                        help='Directory with preference dataset (from generate script)')
    parser.add_argument('--ckpt_file', type=str, default='pretrain_model.ckpt',
                        help='Path to MIDI-BERT pretrain checkpoint')
    parser.add_argument('--output_dir', type=str, default='reward_model_output',
                        help='Output directory for trained model')
    parser.add_argument('--resume_from', type=str, default= '',
                        help="Resume checkpoint path, or 'latest' to auto-pick latest checkpoint in output_dir")

    # Model
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='BERT hidden size')
    parser.add_argument('--head_intermediate_size', type=int, default=256,
                        help='Reward head intermediate layer size')
    parser.add_argument('--head_second_layer_size', type=int, default=64,
                        help='Reward head second layer size')
    parser.add_argument('--head_dropout_rate', type=float, default=0.1,
                        help='Reward head dropout rate')
    parser.add_argument('--config_json', type=str, default='',
                        help='Path to a JSON file containing hyperparameters. Overrides other args.')

    # LoRA
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    parser.add_argument('--lora_layers', type=int, default=4,
                        help='Number of last transformer layers to apply LoRA to')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--max_train_pairs', type=int, default=0,
                        help='Max training pairs (0=all, default=50000 for practical training)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers')

    # Checkpointing / early stopping
    parser.add_argument('--save_every_epochs', type=int, default=1,
                        help='Save a checkpoint every N epochs (0=disabled)')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Stop if val acc does not improve for N epochs (0=disabled)')
    parser.add_argument('--early_stop_min_delta', type=float, default=5e-3,
                        help='Minimum val acc improvement to reset patience')

    # Misc
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    parser.add_argument('--seed', type=int, default=52,
                        help='Random seed')

    return parser.parse_args()


if __name__ == '__main__':
    main()
