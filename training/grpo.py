import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

from configs.config import Config
from models.cp_transformer import CPTransformer # Ensure this exists
from models.reward_model import RewardModel   # Ensure this exists
from models.reference_encoder import ReferenceEncoder # Ensure this exists
from training.rewards import compute_combined_reward
from utils.logging import ExperimentLogger
from utils.metrics_store import GRPOMetricsStore
from utils.checkpoint import CheckpointManager
from data.dataset import build_dataloaders
from features.mft_extractor import MFTExtractor # Ensure this exists


def train_grpo(config: Config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    logger = ExperimentLogger("grpo_training", log_dir="logs")
    metrics_store = GRPOMetricsStore()
    checkpoint_manager = CheckpointManager("checkpoints/grpo")
    
    # Load Models
    active_policy = CPTransformer(config).to(device)
    ref_policy = CPTransformer(config).to(device)
    ref_policy.load_state_dict(active_policy.state_dict())
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad = False
        
    reward_model = RewardModel(config).to(device)
    # Assume reward model is pre-trained
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
        
    ref_encoder = ReferenceEncoder(config).to(device)
    mft_extractor = MFTExtractor(config)
    
    optimizer = AdamW(list(active_policy.parameters()) + list(ref_encoder.parameters()), lr=1e-5)
    
    train_loader, _ = build_dataloaders(config)
    train_iter = iter(train_loader)
    
    total_steps = 10000
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    G = 4 # Group size for GRPO
    prompt_len = 16
    beta = 0.01 # KL penalty weight
    lambd = 0.01 # Entropy penalty weight
    H_target = 2.0
    eps = 1e-8
    clip_ratio = 0.2
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    logger.info("Starting GRPO Phase 2 Training...")
    
    active_policy.train()
    ref_encoder.train()
    
    for step in range(total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        # batch["cp_sequences"]: (B, seq_len, 4)
        # batch["y_ref"]: (B, num_features, time_steps)
        prompts = batch["cp_sequences"][:, :prompt_len, :].to(device)
        y_ref = batch["y_ref"].to(device)
        B = prompts.size(0)
        
        # 1. Encode reference
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            ref_embeddings = ref_encoder(y_ref) # (B, ref_seq_len, d_model)
            
        # We need to expand prompts and y_ref for group size G
        # prompts: (B, prompt_len, 4) -> (B*G, prompt_len, 4)
        prompts_expanded = prompts.repeat_interleave(G, dim=0)
        y_ref_expanded = y_ref.repeat_interleave(G, dim=0)
        
        # 2. Generate completions (active policy)
        with torch.no_grad():
            # generate() should accept prompts and ref_embeddings and return full sequences and log_probs
            # This is a simplification; assume it returns completed sequences (B*G, max_seq_len, 4)
            completions = active_policy.generate(prompts_expanded, ref_embeddings=ref_embeddings.repeat_interleave(G, dim=0), max_length=config.data.max_seq_len)
            
        # 3. Evaluate combined reward
        # completions: (B*G, max_seq_len, 4)
        rewards = compute_combined_reward(completions, y_ref_expanded, reward_model, mft_extractor, config) # (B*G,)
        
        # 4. Group-relative advantage A_hat
        rewards = rewards.view(B, G)
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True)
        A_hat = ((rewards - mean_r) / (std_r + eps)).view(B * G)
        
        # 5. Compute GRPO Loss
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Forward pass on completions
            # Need log probs of completions under active and ref policy
            # get_log_probs should return sum of log probs for the generated part
            active_log_probs, active_entropy = active_policy.get_log_probs_and_entropy(completions, ref_embeddings=ref_embeddings.repeat_interleave(G, dim=0))
            with torch.no_grad():
                ref_log_probs, _ = ref_policy.get_log_probs_and_entropy(completions, ref_embeddings=ref_embeddings.repeat_interleave(G, dim=0))
                
            ratio = torch.exp(active_log_probs - ref_log_probs)
            
            # Clipped surrogate objective
            surrogate1 = ratio * A_hat
            surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * A_hat
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # KL penalty
            kl_div = (ref_log_probs - active_log_probs).mean()
            kl_loss = beta * kl_div
            
            # Entropy penalty
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
            metrics_store.update({
                "grpo_loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "kl_loss": kl_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "reward_mean": rewards.mean().item(),
                "reward_std": std_r.mean().item(),
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=step)
            
            logger.info(f"Step {step}: Loss {total_loss.item():.4f} | Reward {rewards.mean().item():.4f}")
            
        if step % 1000 == 0 and step > 0:
            checkpoint_manager.save(active_policy, optimizer, step)
            logger.info(f"Saved checkpoint at step {step}")


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
