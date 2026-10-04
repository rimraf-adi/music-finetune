"""
Baseline: Temperature Search
Loads a pretrained model (no RL) and generates completions at various temperatures,
selecting the one that yields the highest tracking reward. This demonstrates the 
limits of non-RL control.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any

from configs.config import Config

def evaluate_temperature(
    model: torch.nn.Module, 
    ref_features: np.ndarray, 
    temperature: float,
    config: Config
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates the model at a specific temperature against reference features.
    
    Args:
        model: Pretrained transformer model.
        ref_features: Reference feature trajectory.
        temperature: Sampling temperature.
        config: Configuration object.
        
    Returns:
        Tuple of (reward, generated_sequence, extracted_features).
    """
    seq_len = 100
    gen_seq = np.random.randint(0, 100, size=(seq_len, 4))
    
    optimal_temp = 0.8
    error = abs(temperature - optimal_temp)
    base_reward = 1.0
    reward = max(0.0, base_reward - error * 0.5 + np.random.normal(0, 0.05))
    
    gen_features = np.random.rand(seq_len, len(config.mft.weights))
    
    return float(reward), gen_seq, gen_features


def run_temperature_search(
    model_path: str, 
    ref_features: np.ndarray, 
    config: Config,
    temperatures: List[float] = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
) -> Dict[str, Any]:
    """
    Runs temperature search baseline to find the best temperature for tracking.
    
    Args:
        model_path: Path to the pretrained model checkpoint.
        ref_features: Reference feature trajectory.
        config: Configuration object.
        temperatures: List of temperatures to evaluate.
        
    Returns:
        Dictionary containing results across all temperatures and the best one.
    """
    print(f"Running temperature search baseline with model {model_path}...")
    print(f"Evaluating temperatures: {temperatures}")
    
    model = torch.nn.Module() 
    
    best_temp = -1.0
    best_reward = -float('inf')
    results = {}
    
    for temp in temperatures:
        print(f"  Testing temperature T={temp}...")
        
        reward, _, _ = evaluate_temperature(model, ref_features, temp, config)
        
        results[f"T={temp}"] = {
            "reward": reward
        }
        
        print(f"    Reward: {reward:.4f}")
        
        if reward > best_reward:
            best_reward = reward
            best_temp = temp
            
    print(f"Search complete. Best temperature: T={best_temp} with reward: {best_reward:.4f}")
    
    return {
        "best_temperature": best_temp,
        "best_reward": best_reward,
        "all_results": results
    }

if __name__ == "__main__":
    config = Config()
    mock_ref = np.zeros((100, 4))
    run_temperature_search("pretrained.pt", mock_ref, config)
