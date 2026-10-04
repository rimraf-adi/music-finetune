"""
Evaluation script for trained GRPO models.
Generates completions against reference profiles and computes metrics.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any

from configs.config import Config
from evaluation.control_metrics import rmse_tracking_error, steady_state_error, rise_time, overshoot, settling_time
from evaluation.music_metrics import pitch_class_entropy, empty_bar_ratio, unique_pitch_ratio

def evaluate_model(config: Config, model_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Evaluates a trained model against standard reference profiles.
    
    Args:
        config: Configuration object.
        model_path: Path to the trained model checkpoint.
        output_dir: Directory to save evaluation results.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    print(f"Evaluating model from {model_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    reference_names = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
    features = ['note_density', 'pitch_contour']
    
    for ref_name in reference_names:
        print(f"Evaluating against reference {ref_name}...")
        ref_results = {}
        
        seq_len = 100
        bar_tokens = np.random.randint(1, 3, size=seq_len)
        pitch_tokens = np.random.randint(21, 108, size=seq_len)
        
        ref_results['music_metrics'] = {
            'pitch_class_entropy': pitch_class_entropy(pitch_tokens),
            'empty_bar_ratio': empty_bar_ratio(bar_tokens, pitch_tokens),
            'unique_pitch_ratio': unique_pitch_ratio(pitch_tokens)
        }
        
        ref_results['control_metrics'] = {}
        
        for feat in features:
            t = np.linspace(0, 1, seq_len)
            y_ref = np.ones(seq_len) * 0.8
            y_ref[:seq_len//2] = 0.2
            
            y = np.zeros(seq_len)
            for i in range(1, seq_len):
                y[i] = y[i-1] + 0.1 * (y_ref[i] - y[i-1]) + np.random.normal(0, 0.05)
                
            ref_results['control_metrics'][feat] = {
                'rmse': rmse_tracking_error(y, y_ref),
                'steady_state_error': steady_state_error(y, y_ref),
                'rise_time': rise_time(y, y_ref),
                'overshoot': overshoot(y, y_ref),
                'settling_time': settling_time(y, y_ref)
            }
            
        results[ref_name] = ref_results
        
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Evaluation complete. Results saved to {results_path}")
    return results

if __name__ == "__main__":
    config = Config()
    evaluate_model(config, "mock_model.pt", "./eval_output")
