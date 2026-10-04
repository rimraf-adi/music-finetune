"""
Music quality metrics for evaluating generated compound word sequences.
"""

import numpy as np
from typing import List, Union


def pitch_class_entropy(pitch_tokens: Union[List[int], np.ndarray]) -> float:
    """
    Computes the Shannon entropy of the pitch class distribution.
    A higher entropy indicates a more uniform distribution of pitch classes.
    
    Args:
        pitch_tokens: Sequence of pitch tokens (0-108, where 21-108 are valid midi pitches).
        
    Returns:
        float: Pitch class entropy.
    """
    if len(pitch_tokens) == 0:
        return 0.0
        
    # Filter valid MIDI pitches (21-108, assuming standard CP format)
    valid_pitches = [p for p in pitch_tokens if 21 <= p <= 108]
    if not valid_pitches:
        return 0.0
        
    # Compute pitch classes (0-11)
    pitch_classes = [p % 12 for p in valid_pitches]
    
    # Calculate probabilities
    counts = np.bincount(pitch_classes, minlength=12)
    probs = counts / len(pitch_classes)
    
    # Calculate entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return float(entropy)


def empty_bar_ratio(bar_tokens: Union[List[int], np.ndarray], pitch_tokens: Union[List[int], np.ndarray]) -> float:
    """
    Computes the ratio of empty bars (bars with no notes) to the total number of bars.
    
    Args:
        bar_tokens: Sequence of bar tokens.
        pitch_tokens: Sequence of pitch tokens.
        
    Returns:
        float: Empty bar ratio.
    """
    if len(bar_tokens) == 0 or len(pitch_tokens) == 0:
        return 0.0
        
    num_bars = 0
    empty_bars = 0
    
    current_bar_has_notes = False
    
    # Iterate through tokens
    prev_bar = -1
    for b, p in zip(bar_tokens, pitch_tokens):
        if b not in (0, 3): # Pad=0, EOS=3 from config
            if b != prev_bar and prev_bar != -1:
                # End of previous bar
                num_bars += 1
                if not current_bar_has_notes:
                    empty_bars += 1
                current_bar_has_notes = False
            
            prev_bar = b
            
            if 21 <= p <= 108:
                current_bar_has_notes = True
                
    # Handle last bar
    if prev_bar != -1:
        num_bars += 1
        if not current_bar_has_notes:
            empty_bars += 1
            
    if num_bars == 0:
        return 0.0
        
    return float(empty_bars / num_bars)


def unique_pitch_ratio(pitch_tokens: Union[List[int], np.ndarray]) -> float:
    """
    Computes the ratio of unique pitches to the total number of pitches.
    
    Args:
        pitch_tokens: Sequence of pitch tokens.
        
    Returns:
        float: Unique pitch ratio.
    """
    valid_pitches = [p for p in pitch_tokens if 21 <= p <= 108]
    if not valid_pitches:
        return 0.0
        
    unique_pitches = set(valid_pitches)
    return float(len(unique_pitches) / len(valid_pitches))
