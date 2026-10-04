import torch
import torch.nn as nn
from typing import Any

from configs.config import Config


def compute_tracking_reward(
    cp_sequences: torch.Tensor,
    y_ref: torch.Tensor,
    config: Config,
    mft_extractor: Any
) -> torch.Tensor:
    """
    Computes the tracking error penalty (negative RMSE per feature weighted by config).
    
    Args:
        cp_sequences: (batch_size, seq_len, 4) The generated CP sequences.
        y_ref: (batch_size, num_features, time_steps) The reference features.
        config: Master configuration.
        mft_extractor: Object to extract MFT features from CP sequences.
        
    Returns:
        reward: (batch_size,) The tracking reward.
    """
    # Extract features from generated CP sequences
    batch_dict = {
        'bar': cp_sequences[..., 0],
        'position': cp_sequences[..., 1],
        'pitch': cp_sequences[..., 2],
        'duration': cp_sequences[..., 3]
    }
    y_gen_list = mft_extractor.extract_batch(batch_dict)
    
    # y_gen_list is a list of (num_bars, 4) arrays. We need to pad and stack them to (batch_size, num_features, time_steps)
    # The paper uses shape (batch_size, 4, num_bars)
    import numpy as np
    max_bars = max([f.shape[0] for f in y_gen_list] + [1])
    y_gen_padded = np.zeros((len(y_gen_list), max_bars, 4), dtype=np.float32)
    for i, f in enumerate(y_gen_list):
        if f.shape[0] > 0:
            y_gen_padded[i, :f.shape[0], :] = f
            
    # Transpose to (batch_size, num_features, time_steps)
    y_gen_padded = np.transpose(y_gen_padded, (0, 2, 1))
    y_gen = torch.tensor(y_gen_padded, device=cp_sequences.device)
    
    # Calculate MSE per feature
    # Ensure shapes match; we might need to truncate to the shortest time_steps
    min_time_steps = min(y_gen.size(2), y_ref.size(2))
    y_gen = y_gen[:, :, :min_time_steps]
    y_ref = y_ref[:, :, :min_time_steps]
    
    # MSE: (batch_size, num_features)
    mse = torch.mean((y_gen - y_ref) ** 2, dim=2)
    
    # RMSE
    rmse = torch.sqrt(mse + 1e-8)
    
    # Weights from config
    # Weights shape: (num_features,)
    weights = torch.tensor(config.mft.weights, device=cp_sequences.device, dtype=torch.float32)
    
    # Weighted negative RMSE sum
    # shape: (batch_size,)
    tracking_reward = -torch.sum(rmse * weights.unsqueeze(0), dim=1)
    
    return tracking_reward


def compute_quality_reward(
    cp_sequences: torch.Tensor,
    reward_model: nn.Module
) -> torch.Tensor:
    """
    Computes the MidiBERT score for the sequences.
    
    Args:
        cp_sequences: (batch_size, seq_len, 4) The generated CP sequences.
        reward_model: The trained MidiBERT reward model.
        
    Returns:
        reward: (batch_size,) The quality reward.
    """
    with torch.no_grad():
        # Assuming the reward model returns a scalar for each sequence
        # Shape: (batch_size,)
        rewards = reward_model(cp_sequences[..., 0], cp_sequences[..., 1], cp_sequences[..., 2], cp_sequences[..., 3]).squeeze(-1)
        return rewards


def compute_combined_reward(
    cp_sequences: torch.Tensor,
    y_ref: torch.Tensor,
    reward_model: nn.Module,
    mft_extractor: Any,
    config: Config
) -> torch.Tensor:
    """
    Computes the combined scalar reward.
    
    Args:
        cp_sequences: (batch_size, seq_len, 4) The generated CP sequences.
        y_ref: (batch_size, num_features, time_steps) The reference features.
        reward_model: The trained MidiBERT reward model.
        mft_extractor: Object to extract MFT features from CP sequences.
        config: Master configuration.
        
    Returns:
        combined_reward: (batch_size,)
    """
    tracking_reward = compute_tracking_reward(cp_sequences, y_ref, config, mft_extractor)
    quality_reward = compute_quality_reward(cp_sequences, reward_model)
    
    # Combine rewards (you could add weights in the config for this)
    # For now, simply adding them.
    combined_reward = tracking_reward + quality_reward
    
    return combined_reward
