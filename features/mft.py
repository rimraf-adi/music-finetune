import numpy as np
import math
from typing import Dict, List, Tuple, Union
import torch

from configs.config import MFTConfig

class MFTExtractor:
    """
    Extracts Musical Feature Trajectories (MFT) from CP token sequences.
    This is the primary sensor in the control loop, computing 4 measurable
    features per bar: Note Density, Pitch Centroid, Tonal Tension, Rhythmic Complexity.
    """
    
    feature_names = ['note_density', 'pitch_centroid', 'tonal_tension', 'rhythmic_complexity']

    def __init__(self):
        self.ic_weights = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3, 5: 0.2, 6: 1.0}

    def _tokens_to_bars(self, bar_tokens: np.ndarray, pos_tokens: np.ndarray, 
                        pitch_tokens: np.ndarray, dur_tokens: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Group valid tokens into bars."""
        bars_data = []
        current_bar = []
        
        for b, pos, p, d in zip(bar_tokens, pos_tokens, pitch_tokens, dur_tokens):
            if b == 3 or pos == 17 or p == 89 or d == 65:  # EOS tokens
                break
            
            if b == 1:  # NEW_BAR
                if current_bar or not bars_data:
                    # Start a new bar. If we already had notes, save them.
                    # If this is the first NEW_BAR, just prepare the current_bar.
                    if current_bar:
                        bars_data.append(current_bar)
                        current_bar = []
                    elif not bars_data and not current_bar:
                        # Handles the very first NEW_BAR token without adding an empty bar beforehand
                        pass

            # Valid note condition: bar != PAD and bar != EOS, pitch != PAD and pitch != EOS
            if b != 0 and b != 3 and p != 0 and p != 89:
                current_bar.append((pos, p))
                
        if current_bar or not bars_data:
            bars_data.append(current_bar)
            
        return bars_data

    def extract(self, bar_tokens: Union[np.ndarray, torch.Tensor], 
                pos_tokens: Union[np.ndarray, torch.Tensor], 
                pitch_tokens: Union[np.ndarray, torch.Tensor], 
                dur_tokens: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract MFT from a single sequence of tokens.
        
        Args:
            bar_tokens: (N,) array of bar tokens
            pos_tokens: (N,) array of position tokens
            pitch_tokens: (N,) array of pitch tokens
            dur_tokens: (N,) array of duration tokens
            
        Returns:
            np.ndarray: (num_bars, 4) array of MFT values
        """
        if isinstance(bar_tokens, torch.Tensor): bar_tokens = bar_tokens.cpu().numpy()
        if isinstance(pos_tokens, torch.Tensor): pos_tokens = pos_tokens.cpu().numpy()
        if isinstance(pitch_tokens, torch.Tensor): pitch_tokens = pitch_tokens.cpu().numpy()
        if isinstance(dur_tokens, torch.Tensor): dur_tokens = dur_tokens.cpu().numpy()

        bars_data = self._tokens_to_bars(bar_tokens, pos_tokens, pitch_tokens, dur_tokens)
        num_bars = len(bars_data)
        
        if num_bars == 0:
            return np.zeros((0, 4), dtype=np.float32)
            
        features = np.zeros((num_bars, 4), dtype=np.float32)
        
        for i, notes in enumerate(bars_data):
            # a) Note Density ND(b)
            features[i, 0] = len(notes)
            
            # b) Pitch Centroid PC(b)
            if len(notes) > 0:
                pitches = [p + 20 for _, p in notes]  # Convert token to MIDI
                features[i, 1] = np.mean(pitches)
            else:
                features[i, 1] = 60.0  # Default middle C
                
            # c) Tonal Tension TT(b)
            if len(notes) >= 2:
                pitches = [p + 20 for _, p in notes]
                tension_sum = 0.0
                for j in range(len(pitches)):
                    for k in range(j + 1, len(pitches)):
                        diff = abs(pitches[j] - pitches[k]) % 12
                        ic = min(diff, 12 - diff)
                        if ic > 0:
                            tension_sum += self.ic_weights.get(ic, 0.0)
                max_pairs = len(pitches) * (len(pitches) - 1) / 2
                features[i, 2] = tension_sum / max_pairs if max_pairs > 0 else 0.0
            else:
                features[i, 2] = 0.0
                
            # d) Rhythmic Complexity RC(b)
            if len(notes) >= 2:
                positions = sorted([pos for pos, _ in notes])
                iois = [positions[j] - positions[j-1] for j in range(1, len(positions))]
                if iois:
                    unique, counts = np.unique(iois, return_counts=True)
                    probs = counts / counts.sum()
                    entropy = -np.sum(probs * np.log2(probs + 1e-9))
                    if len(unique) > 1:
                        features[i, 3] = entropy / np.log2(len(unique))
                    else:
                        features[i, 3] = 0.0
                else:
                    features[i, 3] = 0.0
            else:
                features[i, 3] = 0.0
                
        return features

    def extract_batch(self, batch_dict: Dict[str, Union[np.ndarray, torch.Tensor]]) -> List[np.ndarray]:
        """
        Extract MFT for a batch of sequences.
        
        Args:
            batch_dict: Dictionary containing batched tensors 'bar', 'position', 'pitch', 'duration'
            
        Returns:
            List of (num_bars, 4) arrays, one for each sequence in the batch.
        """
        bars = batch_dict['bar']
        positions = batch_dict['position']
        pitches = batch_dict['pitch']
        durations = batch_dict['duration']
        
        batch_size = bars.shape[0]
        results = []
        
        for i in range(batch_size):
            feats = self.extract(bars[i], positions[i], pitches[i], durations[i])
            results.append(feats)
            
        return results

    def extract_single_feature(self, tokens: Dict[str, Union[np.ndarray, torch.Tensor]], feature_name: str) -> np.ndarray:
        """Extract only a single named feature from a single sequence."""
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature must be one of {self.feature_names}")
            
        feats = self.extract(tokens['bar'], tokens['position'], tokens['pitch'], tokens['duration'])
        idx = self.feature_names.index(feature_name)
        return feats[:, idx]


def normalize(mft_values: np.ndarray, config: MFTConfig) -> np.ndarray:
    """
    Normalize MFT features to [0, 1] range using the MFTConfig ranges.
    
    Args:
        mft_values: (..., 4) array of raw MFT features
        config: MFTConfig object containing ranges
        
    Returns:
        (..., 4) array of normalized MFT features
    """
    norm = np.zeros_like(mft_values)
    ranges = [
        config.nd_range,
        config.pc_range,
        config.tt_range,
        config.rc_range
    ]
    
    for i in range(4):
        min_val, max_val = ranges[i]
        span = max_val - min_val
        if span > 0:
            norm[..., i] = np.clip((mft_values[..., i] - min_val) / span, 0.0, 1.0)
        else:
            norm[..., i] = 0.0
            
    return norm


def denormalize(normalized: np.ndarray, config: MFTConfig) -> np.ndarray:
    """
    Denormalize MFT features from [0, 1] back to raw scales.
    
    Args:
        normalized: (..., 4) array of normalized MFT features
        config: MFTConfig object containing ranges
        
    Returns:
        (..., 4) array of raw MFT features
    """
    denorm = np.zeros_like(normalized)
    ranges = [
        config.nd_range,
        config.pc_range,
        config.tt_range,
        config.rc_range
    ]
    
    for i in range(4):
        min_val, max_val = ranges[i]
        span = max_val - min_val
        denorm[..., i] = normalized[..., i] * span + min_val
        
    return denorm
