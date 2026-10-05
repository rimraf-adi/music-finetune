import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from configs.config import Config
from models.cp_transformer import CPTransformer
from models.reference_encoder import ReferenceEncoder
from features.mft import MFTExtractor


def evaluate_step_response(
    policy: CPTransformer,
    ref_encoder: ReferenceEncoder,
    mft_extractor: MFTExtractor,
    device: torch.device,
    channel: str = "pitch_centroid",
    num_bars: int = 16,
    step_bar: int = 8,
    val_low: float = 48.0,
    val_high: float = 76.0,
    num_trials: int = 5
) -> Dict[str, float]:
    """
    Evaluates policy response to an abrupt step change in target reference trajectory:
    Measures:
      - Steady-state error (before and after step)
      - Rise time (bars needed to reach 90% of target delta)
      - Overshoot percentage
    """
    channel_idx = {"note_density": 0, "pitch_centroid": 1, "tonal_tension": 2, "rhythmic_complexity": 3}[channel]
    
    # 1. Build step trajectory: val_low for bars 0..step_bar-1, val_high for bars step_bar..num_bars-1
    y_ref = torch.zeros((1, num_bars, 4), device=device)
    # Default values for other channels
    y_ref[..., 0] = 8.0   # note density
    y_ref[..., 1] = 60.0  # pitch centroid
    y_ref[..., 2] = 0.3   # tonal tension
    y_ref[..., 3] = 0.3   # rhythmic complexity
    
    # Apply step on target channel
    y_ref[0, :step_bar, channel_idx] = val_low
    y_ref[0, step_bar:, channel_idx] = val_high
    
    encoded_ref = ref_encoder(y_ref)
    
    trial_trajectories = []
    
    for _ in range(num_trials):
        prompt_bar = torch.tensor([[1]], device=device)
        prompt_pos = torch.tensor([[1]], device=device)
        prompt_pitch = torch.tensor([[int(val_low)]], device=device)
        prompt_dur = torch.tensor([[4]], device=device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                completions = policy.generate(
                    prompt_bar=prompt_bar, prompt_pos=prompt_pos,
                    prompt_pitch=prompt_pitch, prompt_dur=prompt_dur,
                    max_new_tokens=220, temperature=0.85,
                    ref_embeddings=encoded_ref
                )
                
        tokens = np.stack([
            completions['bar'][0].cpu().numpy(),
            completions['position'][0].cpu().numpy(),
            completions['pitch'][0].cpu().numpy(),
            completions['duration'][0].cpu().numpy()
        ], axis=-1)
        
        feats = mft_extractor.extract(tokens[:, 0], tokens[:, 1], tokens[:, 2], tokens[:, 3])
        if len(feats) >= num_bars:
            trial_trajectories.append(feats[:num_bars, channel_idx])
        elif len(feats) > 0:
            padded = np.pad(feats[:, channel_idx], (0, num_bars - len(feats)), mode='edge')
            trial_trajectories.append(padded)
            
    if not trial_trajectories:
        return {"error": "Failed to generate valid trajectories."}
        
    avg_trajectory = np.mean(trial_trajectories, axis=0) # (num_bars,)
    target_values = y_ref[0, :, channel_idx].cpu().numpy()
    
    # Steady state error before step (last 3 bars before step)
    pre_step_error = np.mean(np.abs(avg_trajectory[max(0, step_bar - 3):step_bar] - val_low))
    # Steady state error after step (last 3 bars of trajectory)
    post_step_error = np.mean(np.abs(avg_trajectory[-3:] - val_high))
    
    # Rise time (fraction of step achieved)
    step_delta = val_high - val_low
    post_step_values = avg_trajectory[step_bar:]
    target_90 = val_low + 0.9 * step_delta
    
    bars_to_90 = None
    for b_idx, val in enumerate(post_step_values):
        if (step_delta > 0 and val >= target_90) or (step_delta < 0 and val <= target_90):
            bars_to_90 = b_idx + 1
            break
            
    return {
        "channel": channel,
        "val_low": float(val_low),
        "val_high": float(val_high),
        "pre_step_error": float(pre_step_error),
        "post_step_error": float(post_step_error),
        "rise_time_bars": bars_to_90 if bars_to_90 is not None else float("inf"),
        "trajectory": avg_trajectory.tolist(),
        "target": target_values.tolist()
    }


def run_control_benchmark(policy_path="checkpoints/grpo/step_0002950/model.pt", ref_encoder_path=None):
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Loading policy from {policy_path} on {device}...")
    
    policy = CPTransformer(config).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy.eval()
    
    ref_encoder = ReferenceEncoder(config).to(device)
    if ref_encoder_path and Path(ref_encoder_path).exists():
        print(f"Loading trained ReferenceEncoder from {ref_encoder_path}...")
        ref_encoder.load_state_dict(torch.load(ref_encoder_path, map_location=device, weights_only=True))
    ref_encoder.eval()
    
    mft_extractor = MFTExtractor()
    
    print("\n--- CONTROL BENCHMARK: STEP RESPONSE ANALYSIS ---")
    
    # Test 1: Pitch Centroid Step Response (MIDI 48 -> 76, C3 -> E5)
    print("\nTesting Pitch Centroid Step Response (48 -> 76 semitones)...")
    res_pitch = evaluate_step_response(
        policy, ref_encoder, mft_extractor, device,
        channel="pitch_centroid", val_low=48.0, val_high=76.0, num_trials=4
    )
    print(f"  Pre-step Steady-State Error: {res_pitch['pre_step_error']:.2f} semitones")
    print(f"  Post-step Steady-State Error: {res_pitch['post_step_error']:.2f} semitones")
    print(f"  Rise Time: {res_pitch['rise_time_bars']} bars")
    
    # Test 2: Note Density Step Response (Sparse 4 notes/bar -> Dense 16 notes/bar)
    print("\nTesting Note Density Step Response (4 -> 16 notes/bar)...")
    res_density = evaluate_step_response(
        policy, ref_encoder, mft_extractor, device,
        channel="note_density", val_low=4.0, val_high=16.0, num_trials=4
    )
    print(f"  Pre-step Steady-State Error: {res_density['pre_step_error']:.2f} notes/bar")
    print(f"  Post-step Steady-State Error: {res_density['post_step_error']:.2f} notes/bar")
    print(f"  Rise Time: {res_density['rise_time_bars']} bars")
    
    return res_pitch, res_density

if __name__ == "__main__":
    run_control_benchmark()
