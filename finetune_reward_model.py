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
import pickle
import random
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

    def __init__(self, hidden_size, intermediate_size=256):
        super().__init__()
        self.pool_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
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
                 use_layer: int = -1):
        super().__init__()
        self.midibert = midibert
        self.reward_head = RewardHead(hidden_size)
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
    """Dataset of (preferred, rejected) token sequence pairs generated on-the-fly."""

    def __init__(self, originals_path, corrupted_path):
        # Load the base arrays. Shape: (n_files, max_seq_len, 4)
        self.originals = np.load(originals_path, allow_pickle=True)
        self.corrupted = np.load(corrupted_path, allow_pickle=True)
        assert len(self.originals) == len(self.corrupted), \
            f"Mismatch: {len(self.originals)} originals vs {len(self.corrupted)} corrupted"
        
        self.n = len(self.originals)

    def __len__(self):
        # Generate the full n² combinations virtually
        return self.n * self.n

    def __getitem__(self, idx):
        # Decode the flat index into (i, j) pair indices
        # i = original file index (preferred)
        # j = corrupted file index (rejected)
        i = idx // self.n
        j = idx % self.n
        
        preferred = torch.tensor(self.originals[i], dtype=torch.long)
        rejected = torch.tensor(self.corrupted[j], dtype=torch.long)
        
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
    Apply LoRA adapters to the last N transformer layers' attention modules.
    """
    # Get total number of layers
    total_layers = model.midibert.bert.config.num_hidden_layers

    # Target: attention query and value projections in last N layers
    target_modules = []
    for layer_idx in range(total_layers - n_layers_to_tune, total_layers):
        target_modules.extend([
            f"midibert.bert.encoder.layer.{layer_idx}.attention.self.query",
            f"midibert.bert.encoder.layer.{layer_idx}.attention.self.value",
        ])

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

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for batch_idx, (preferred, rejected) in enumerate(pbar):
        preferred = preferred.to(device)  # (B, seq_len, 4)
        rejected = rejected.to(device)

        # Forward pass
        r_preferred = model(preferred)  # (B,)
        r_rejected = model(rejected)    # (B,)

        # Bradley-Terry loss: -log σ(r_w - r_l)
        loss = -F.logsigmoid(r_preferred - r_rejected).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Metrics
        total_loss += loss.item() * preferred.size(0)
        total_correct += (r_preferred > r_rejected).sum().item()
        total_samples += preferred.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{total_correct / total_samples:.3f}',
        })

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for preferred, rejected in dataloader:
        preferred = preferred.to(device)
        rejected = rejected.to(device)

        r_preferred = model(preferred)
        r_rejected = model(rejected)

        loss = -F.logsigmoid(r_preferred - r_rejected).mean()

        total_loss += loss.item() * preferred.size(0)
        total_correct += (r_preferred > r_rejected).sum().item()
        total_samples += preferred.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def main():
    args = parse_args()

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
        use_layer=-1  # last hidden layer
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

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    best_val_acc = 0.0
    best_epoch = 0
    log_lines = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            reward_model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss, val_acc = evaluate(reward_model, val_loader, device)

        log = (f"Epoch {epoch + 1}/{args.epochs} | "
               f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
               f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
        print(log)
        log_lines.append(log)

        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_path = os.path.join(args.output_dir, 'best_reward_model.pt')
            # Save both LoRA adapters and reward head
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': reward_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'e2w': e2w,
                'w2e': w2e,
                'args': vars(args),
            }, save_path)
            print(f"  ★ Saved best model (val_acc={val_acc:.3f})")

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_reward_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': reward_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'e2w': e2w,
        'w2e': w2e,
        'args': vars(args),
    }, final_path)

    # Save training log
    log_path = os.path.join(args.output_dir, 'training_log.txt')
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

    # Model
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='BERT hidden size')

    # LoRA
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    parser.add_argument('--lora_layers', type=int, default=2,
                        help='Number of last transformer layers to apply LoRA to')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--max_train_pairs', type=int, default=0,
                        help='Max training pairs (0=all, default=50000 for practical training)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers')

    # Misc
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    parser.add_argument('--seed', type=int, default=52,
                        help='Random seed')

    return parser.parse_args()


if __name__ == '__main__':
    main()
