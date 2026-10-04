import numpy as np
from typing import List, Tuple, Dict

def step_reference(feature_idx: int, start_val: float, end_val: float, 
                  step_bar: int, total_bars: int) -> np.ndarray:
    """
    Feature jumps from value_start to value_end at bar step_bar.
    Other features get 'don't care' value = NaN.
    """
    ref = np.full((total_bars, 4), np.nan, dtype=np.float32)
    ref[:step_bar, feature_idx] = start_val
    ref[step_bar:, feature_idx] = end_val
    return ref

def ramp_reference(feature_idx: int, start_val: float, end_val: float, 
                   total_bars: int) -> np.ndarray:
    """
    Feature linearly increases from start_val to end_val over total_bars.
    Other features get 'don't care' value = NaN.
    """
    ref = np.full((total_bars, 4), np.nan, dtype=np.float32)
    ref[:, feature_idx] = np.linspace(start_val, end_val, total_bars)
    return ref

def sinusoidal_reference(feature_idx: int, center: float, amplitude: float, 
                        period_bars: float, total_bars: int) -> np.ndarray:
    """
    Feature oscillates around center with amplitude and period.
    Other features get 'don't care' value = NaN.
    """
    ref = np.full((total_bars, 4), np.nan, dtype=np.float32)
    t = np.arange(total_bars)
    ref[:, feature_idx] = center + amplitude * np.sin(2 * np.pi * t / period_bars)
    return ref

def hold_reference(feature_idx: int, value: float, total_bars: int) -> np.ndarray:
    """
    Feature stays constant at value.
    Other features get 'don't care' value = NaN.
    """
    ref = np.full((total_bars, 4), np.nan, dtype=np.float32)
    ref[:, feature_idx] = value
    return ref

def multi_reference(specs: List[Tuple[int, str, Dict]], total_bars: int) -> np.ndarray:
    """
    Multiple features tracked simultaneously.
    
    Args:
        specs: list of tuples (feature_idx, ref_type, params_dict)
               where ref_type is 'step', 'ramp', 'sinusoidal', or 'hold'
               and params_dict contains the kwargs for the respective function.
        total_bars: total number of bars
    """
    ref = np.full((total_bars, 4), np.nan, dtype=np.float32)
    
    for feature_idx, ref_type, params in specs:
        if ref_type == 'step':
            feat_ref = step_reference(feature_idx, total_bars=total_bars, **params)
        elif ref_type == 'ramp':
            feat_ref = ramp_reference(feature_idx, total_bars=total_bars, **params)
        elif ref_type == 'sinusoidal':
            feat_ref = sinusoidal_reference(feature_idx, total_bars=total_bars, **params)
        elif ref_type == 'hold':
            feat_ref = hold_reference(feature_idx, total_bars=total_bars, **params)
        else:
            raise ValueError(f"Unknown reference type: {ref_type}")
            
        ref[:, feature_idx] = feat_ref[:, feature_idx]
        
    return ref

def get_reference_library(total_bars: int = 16) -> Dict[str, np.ndarray]:
    """
    Returns the pre-defined reference library for GRPO tracking.
    Features: 0=ND, 1=PC, 2=TT, 3=RC.
    """
    library = {}
    
    # R1: Step in note density, 4->12 at bar 8
    library['R1'] = step_reference(0, 4.0, 12.0, 8, total_bars)
    
    # R2: Ramp in pitch centroid, MIDI 48->72
    library['R2'] = ramp_reference(1, 48.0, 72.0, total_bars)
    
    # R3: Step+hold in tonal tension, step to 0.7 at bar 4, hold for 12 bars
    library['R3'] = step_reference(2, 0.0, 0.7, 4, total_bars)  # Starts at 0.0, steps to 0.7 at bar 4
    
    # R4: Sinusoidal rhythmic complexity, center=0.5, amplitude=0.3, period=8
    library['R4'] = sinusoidal_reference(3, 0.5, 0.3, 8.0, total_bars)
    
    # R5: Multi-feature: step ND 4->12 at bar 8 + ramp PC 48->72 simultaneously
    specs = [
        (0, 'step', {'start_val': 4.0, 'end_val': 12.0, 'step_bar': 8}),
        (1, 'ramp', {'start_val': 48.0, 'end_val': 72.0})
    ]
    library['R5'] = multi_reference(specs, total_bars)
    
    # R6: Step-down in note density, 12->4 at bar 8
    library['R6'] = step_reference(0, 12.0, 4.0, 8, total_bars)
    
    return library
