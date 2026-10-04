import argparse
import os
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from configs.config import Config
from models.reward_model import RewardModel  # Ensure this exists
from data.dataset import build_dataloaders    # Ensure this exists
from utils.logging import ExperimentLogger    # Ensure this exists
from utils.metrics_store import MetricsStore  # Ensure this exists
from utils.checkpoint import CheckpointManager # Ensure this exists


def corrupt_sequences(sequences: torch.Tensor, config: Config) -> torch.Tensor:
    """
    Applies synthetic corruption to create 'loser' sequences.
    
    Args:
        sequences: (batch_size, seq_len, 4) Clean CP sequences.
        config: Configuration.
        
    Returns:
        corrupted: (batch_size, seq_len, 4) Corrupted CP sequences.
    """
    corrupted = sequences.clone()
    batch_size, seq_len, _ = corrupted.shape
    
    for i in range(batch_size):
        corruption_type = random.choice(["shuffle_bars", "pitch_noise"])
        
        if corruption_type == "shuffle_bars":
            # Very simplified shuffle: just chunk into 4 and shuffle chunks
            chunks = torch.chunk(corrupted[i], chunks=4, dim=0)
            shuffled_chunks = list(chunks)
            random.shuffle(shuffled_chunks)
            corrupted[i] = torch.cat(shuffled_chunks, dim=0)
            
        elif corruption_type == "pitch_noise":
            # Add noise to pitch (index 2 of the 4 tuple)
            pitch_noise = torch.randint(-2, 3, (seq_len,), device=sequences.device)
            new_pitch = corrupted[i, :, 2] + pitch_noise
            # Clamp to pitch range
            new_pitch = torch.clamp(new_pitch, config.data.min_pitch, config.data.max_pitch)
            corrupted[i, :, 2] = new_pitch
            
    return corrupted


def train_reward(config: Config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Initialize logger and metrics
    logger = ExperimentLogger("reward_model_training", log_dir="logs")
    metrics_store = MetricsStore()
    checkpoint_manager = CheckpointManager("checkpoints/reward_model")
    
    # Initialize model
    model = RewardModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Get dataloaders
    train_loader, val_loader = build_dataloaders(config)
    
    margin = 0.5
    num_epochs = 10
    best_val_loss = float('inf')
    
    logger.info("Starting reward model training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Assuming batch is a dictionary with "cp_sequences"
            winners = batch["cp_sequences"].to(device)
            
            # Create losers via corruption
            losers = corrupt_sequences(winners, config).to(device)
            
            # Forward pass
            reward_winners = model(winners).squeeze(-1)
            reward_losers = model(losers).squeeze(-1)
            
            # Bradley-Terry loss
            loss = -torch.log(torch.sigmoid(reward_winners - reward_losers - margin) + 1e-8).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                metrics_store.update({"train_loss": loss.item()}, step=epoch * len(train_loader) + batch_idx)
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: Train Loss = {loss.item():.4f}")
                
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                winners = batch["cp_sequences"].to(device)
                losers = corrupt_sequences(winners, config).to(device)
                
                reward_winners = model(winners).squeeze(-1)
                reward_losers = model(losers).squeeze(-1)
                
                loss = -torch.log(torch.sigmoid(reward_winners - reward_losers - margin) + 1e-8).mean()
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        metrics_store.update({"val_loss": avg_val_loss}, step=(epoch + 1) * len(train_loader))
        logger.info(f"Epoch {epoch} Summary: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_manager.save(model, optimizer, epoch, is_best=True)
            logger.info("Saved new best checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MidiBERT Reward Model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    # In a real scenario, you might load from args.config here
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    train_reward(config)
