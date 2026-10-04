import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import pickle
import pathlib

from configs.config import Config
from models.cp_transformer import CPTransformer
from models.reward_model import RewardModel
from models.reference_encoder import ReferenceEncoder
from training.rewards import compute_combined_reward
from utils.logging import ExperimentLogger
from utils.metrics_store import GRPOMetricsStore
from utils.checkpoint import CheckpointManager
from data.dataset import build_dataloaders
from features.mft import MFTExtractor

def train_grpo(config: Config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    metrics_store = GRPOMetricsStore()
    checkpoint_manager = CheckpointManager("checkpoints/grpo")
    
    # Load Models
    active_policy = CPTransformer(config).to(device)
    
    # LOAD PRETRAIN CHECKPOINT
    import os
    import glob
    pretrain_dirs = sorted(glob.glob("checkpoints/pretrain_*"))
    if pretrain_dirs:
        latest_pretrain_dir = pretrain_dirs[-1]
        step_dirs = sorted(glob.glob(f"{latest_pretrain_dir}/step_*"))
        if step_dirs:
            latest_step_dir = step_dirs[-1]
            print(f"Loading pretrain checkpoint from {latest_step_dir}")
            active_policy.load_state_dict(torch.load(os.path.join(latest_step_dir, "model.pt"), weights_only=True))
            
    ref_policy = CPTransformer(config).to(device)
    ref_policy.load_state_dict(active_policy.state_dict())
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad = False
        
    reward_model = RewardModel(config).to(device)
    
    # LOAD REWARD MODEL CHECKPOINT
    reward_step_dirs = sorted(glob.glob("checkpoints/reward_model/step_*"))
    if reward_step_dirs:
        latest_reward_step_dir = reward_step_dirs[-1]
        print(f"Loading reward model checkpoint from {latest_reward_step_dir}")
        reward_model.load_state_dict(torch.load(os.path.join(latest_reward_step_dir, "model.pt"), weights_only=True))
        
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
        
    ref_encoder = ReferenceEncoder(config).to(device)
    mft_extractor = MFTExtractor()
    
    optimizer = AdamW(list(active_policy.parameters()) + list(ref_encoder.parameters()), lr=1e-5)
    
    with open(pathlib.Path(config.data.processed_dir) / "grpo.pkl", "rb") as f:
        grpo_seqs = pickle.load(f)
        
    loaders = build_dataloaders(config, [], grpo_seqs)
    train_loader = loaders['grpo']
    
    if train_loader is None or len(train_loader) == 0:
        print("GRPO train loader is empty.")
        return
        
    train_iter = iter(train_loader)
    
    total_steps = config.grpo.total_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    G = 4 # Group size for GRPO
    beta = 0.01 # KL penalty weight
    lambd = 0.01 # Entropy penalty weight
    H_target = 2.0
    eps = 1e-8
    clip_ratio = 0.2
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    print("Starting GRPO Phase 2 Training...")
    
    active_policy.train()
    ref_encoder.train()
    
    for step in range(total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        prompt_dict, completion_dict = batch
        prompt_bar = prompt_dict['bar'].to(device)
        prompt_pos = prompt_dict['position'].to(device)
        prompt_pitch = prompt_dict['pitch'].to(device)
        prompt_dur = prompt_dict['duration'].to(device)
        
        B = prompt_bar.size(0)
        
        # Create random y_ref trajectories for training (batch, num_features, time_steps)
        y_ref = torch.zeros((B, 4, 32), device=device)
        y_ref[:, 0, :] = torch.rand((B, 32), device=device) * 20.0          # Note density: 0-20
        y_ref[:, 1, :] = torch.rand((B, 32), device=device) * 87.0 + 21.0   # Pitch centroid: 21-108
        y_ref[:, 2, :] = torch.rand((B, 32), device=device)                 # Tonal tension: 0-1
        y_ref[:, 3, :] = torch.rand((B, 32), device=device)                 # Rhythmic complexity: 0-1
        
        # We need to expand prompts and y_ref for group size G
        prompt_bar = prompt_bar.repeat_interleave(G, dim=0)
        prompt_pos = prompt_pos.repeat_interleave(G, dim=0)
        prompt_pitch = prompt_pitch.repeat_interleave(G, dim=0)
        prompt_dur = prompt_dur.repeat_interleave(G, dim=0)
        y_ref_expanded = y_ref.repeat_interleave(G, dim=0)
        
        # 2. Encode Reference
        # ReferenceEncoder expects (B, num_bars, num_features)
        y_ref_t = y_ref_expanded.transpose(1, 2)
        encoded_ref = ref_encoder(y_ref_t)
        
        # 2b. Generate completions (active policy)
        with torch.no_grad():
            completions = active_policy.generate(
                prompt_bar=prompt_bar,
                prompt_pos=prompt_pos,
                prompt_pitch=prompt_pitch,
                prompt_dur=prompt_dur,
                max_new_tokens=config.grpo.completion_len,
                ref_embeddings=encoded_ref
            )
            
        # 3. Evaluate combined reward
        completions_tensor = torch.stack([
            completions['bar'], completions['position'], 
            completions['pitch'], completions['duration']
        ], dim=-1)
        
        rewards = compute_combined_reward(completions_tensor, y_ref_expanded, reward_model, mft_extractor, config)
        
        # 4. Group-relative advantage A_hat
        rewards_view = rewards.view(B, G)
        mean_r = rewards_view.mean(dim=1, keepdim=True)
        std_r = rewards_view.std(dim=1, keepdim=True)
        A_hat = ((rewards_view - mean_r) / (std_r + eps)).view(B * G)
        
        # 5. Compute GRPO Loss
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            active_log_probs = active_policy.log_probs(completions['bar'], completions['position'], completions['pitch'], completions['duration'], ref_embeddings=encoded_ref).sum(dim=1)
            active_entropy = active_policy.entropy(completions['bar'], completions['position'], completions['pitch'], completions['duration'], ref_embeddings=encoded_ref)['entropy_normalized']
            
            with torch.no_grad():
                # The reference policy is the base pre-trained model, which was NOT trained with ref_embeddings.
                # Passing ref_embeddings to it would corrupt its predictions and cause KL divergence to explode.
                ref_log_probs = ref_policy.log_probs(completions['bar'], completions['position'], completions['pitch'], completions['duration']).sum(dim=1)
                
            # Compute ratio in float32 to prevent float16/float32 overflow
            log_ratio = (active_log_probs - ref_log_probs).float()
            
            # Clamp log_ratio to [-20, 20] so ratio is bounded between ~2e-9 and ~4.8e8
            log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            
            # Group-relative advantage A_hat should also be float32 for safety
            A_hat_f32 = A_hat.float()
            
            surrogate1 = ratio * A_hat_f32
            surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * A_hat_f32
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            kl_div = -log_ratio.mean()
            kl_loss = beta * kl_div
            
            mean_entropy = active_entropy.mean()
            entropy_loss = lambd * (F.relu(H_target - mean_entropy) ** 2)
            
            total_loss = policy_loss + kl_loss + entropy_loss
            
        # 6. Backward and step
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(active_policy.parameters()) + list(ref_encoder.parameters()), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Log metrics
        if step % 10 == 0:
            metrics_store.record(step, grpo_loss=total_loss.item())
            print(f"Step {step}: Loss {total_loss.item():.4f} | Reward {rewards.mean().item():.4f}")
            
        if step % 50 == 0 and step > 0:
            checkpoint_manager.save(step=step, model=active_policy, optimizer=optimizer)
            print(f"Saved checkpoint at step {step}")
            
    print("GRPO training completed successfully.")

if __name__ == "__main__":
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    config = Config()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    train_grpo(config)
