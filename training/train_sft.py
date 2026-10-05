import argparse
import os
import glob
import pickle
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from configs.config import Config
from models.cp_transformer import CPTransformer
from models.reference_encoder import ReferenceEncoder
from features.mft import MFTExtractor
from utils.checkpoint import CheckpointManager


class SFTDataset(Dataset):
    """
    Dataset that pairs clean MIDI token sequences with their ground-truth MFT trajectories.
    """
    def __init__(self, sequences, mft_extractor, num_bars_target=32):
        self.samples = []
        for seq in sequences:
            if len(seq) < 16:
                continue
            # Extract ground-truth features per bar
            feats = mft_extractor.extract(seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3])
            if len(feats) == 0:
                continue
            
            # Pad or truncate feats to fixed num_bars_target
            feats_padded = np.zeros((num_bars_target, 4), dtype=np.float32)
            valid_bars = min(len(feats), num_bars_target)
            feats_padded[:valid_bars] = feats[:valid_bars]
            
            self.samples.append((seq, feats_padded))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        seq, feats = self.samples[idx]
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        return seq_tensor, feats_tensor


def sft_collate_fn(batch):
    sequences, feats_list = zip(*batch)
    max_len = max(len(s) for s in sequences)
    
    # Pad sequences to max_len
    b_sz = len(sequences)
    padded_seqs = torch.zeros((b_sz, max_len, 4), dtype=torch.long)
    for i, s in enumerate(sequences):
        padded_seqs[i, :len(s)] = s
        
    feats_tensor = torch.stack(feats_list, dim=0) # (B, num_bars, 4)
    
    # Teacher forcing: inputs are seq[:-1], targets are seq[1:]
    inputs = padded_seqs[:, :-1]
    targets = padded_seqs[:, 1:]
    
    input_dict = {
        'bar': inputs[..., 0],
        'position': inputs[..., 1],
        'pitch': inputs[..., 2],
        'duration': inputs[..., 3]
    }
    target_dict = {
        'bar': targets[..., 0],
        'position': targets[..., 1],
        'pitch': targets[..., 2],
        'duration': targets[..., 3]
    }
    return input_dict, target_dict, feats_tensor


def train_sft(epochs=5, batch_size=32, lr=1e-4):
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Starting Supervised Conditioning Fine-Tuning (SFT) on {device}...")
    
    mft_extractor = MFTExtractor()
    
    # Load processed sequences
    data_file = pathlib.Path(config.data.processed_dir) / "pretrain.pkl"
    print(f"Loading sequences from {data_file}...")
    with open(data_file, "rb") as f:
        sequences = pickle.load(f)
        
    print(f"Building SFT dataset from {len(sequences)} sequences...")
    dataset = SFTDataset(sequences, mft_extractor, num_bars_target=32)
    print(f"Built {len(dataset)} valid conditioned pairs.")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=sft_collate_fn)
    
    # Initialize policy and reference encoder
    policy = CPTransformer(config).to(device)
    ref_encoder = ReferenceEncoder(config).to(device)
    
    # Load pretrained weights into policy
    pretrain_dirs = sorted(glob.glob("checkpoints/pretrain_*"))
    if pretrain_dirs:
        latest_pretrain_dir = pretrain_dirs[-1]
        step_dirs = sorted(glob.glob(f"{latest_pretrain_dir}/step_*"))
        if step_dirs:
            latest_step_dir = step_dirs[-1]
            print(f"Loading base CPTransformer checkpoint from {latest_step_dir}...")
            policy.load_state_dict(torch.load(os.path.join(latest_step_dir, "model.pt"), map_location=device, weights_only=True))
            
    optimizer = AdamW(list(policy.parameters()) + list(ref_encoder.parameters()), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    ckpt_mgr = CheckpointManager("checkpoints/sft_conditioned")
    
    global_step = 0
    policy.train()
    ref_encoder.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (input_dict, target_dict, y_ref) in enumerate(loader):
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            target_dict = {k: v.to(device) for k, v in target_dict.items()}
            y_ref = y_ref.to(device) # (B, num_bars, 4)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # Encode reference trajectory into sequence prefix
                ref_embeddings = ref_encoder(y_ref) # (B, num_bars, d_model)
                
                # Forward pass conditioned on reference
                logits_dict = policy(
                    input_dict['bar'], input_dict['position'],
                    input_dict['pitch'], input_dict['duration'],
                    ref_embeddings=ref_embeddings
                )
                
                loss_dict = policy.compute_loss(logits_dict, target_dict)
                loss = loss_dict['loss_total']
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(ref_encoder.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx+1}/{len(loader)}] | Step {global_step} | Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s ---")
        
        # Save checkpoint each epoch
        ckpt_mgr.save(step=epoch+1, model=policy, optimizer=optimizer)
        # Also save reference encoder weights
        torch.save(ref_encoder.state_dict(), f"checkpoints/sft_conditioned/step_{epoch+1:07d}/ref_encoder.pt")
        print(f"Saved SFT checkpoint for epoch {epoch+1}")
        
    print("SFT Conditioning Pre-training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train_sft(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
