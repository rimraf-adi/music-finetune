"""
Control theory metrics for evaluating reference tracking performance.
"""

import numpy as np


def rmse_tracking_error(y: np.ndarray, y_ref: np.ndarray) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between the generated sequence
    and the reference sequence.
    
    Args:
        y (np.ndarray): Generated feature values.
        y_ref (np.ndarray): Reference feature values.
        
    Returns:
        float: RMSE tracking error.
    """
    if len(y) == 0 or len(y_ref) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y - y_ref) ** 2)))


def steady_state_error(y: np.ndarray, y_ref: np.ndarray, last_fraction: float = 0.25) -> float:
    """
    Computes the steady-state error by averaging the error over the final
    portion of the sequence.
    
    Args:
        y (np.ndarray): Generated feature values.
        y_ref (np.ndarray): Reference feature values.
        last_fraction (float): Fraction of the sequence to consider as steady state.
        
    Returns:
        float: Steady-state error.
    """
    if len(y) == 0 or len(y_ref) == 0:
        return 0.0
    n_samples = max(1, int(len(y) * last_fraction))
    return float(np.mean(np.abs(y[-n_samples:] - y_ref[-n_samples:])))


def rise_time(y: np.ndarray, y_ref: np.ndarray, threshold: float = 0.9) -> float:
    """
    Computes the rise time as the index where the generated sequence first
    reaches a specified fraction of the final reference value.
    Assumes a step response where reference is constant.
    
    Args:
        y (np.ndarray): Generated feature values.
        y_ref (np.ndarray): Reference feature values.
        threshold (float): Fraction of the final reference value to reach.
        
    Returns:
        float: Rise time (in steps). Returns len(y) if threshold is never reached.
    """
    if len(y) == 0 or len(y_ref) == 0:
        return 0.0
    target_value = y_ref[-1] * threshold
    
    # Simple check for step up vs step down
    if y_ref[-1] > y[0]:
        indices = np.where(y >= target_value)[0]
    else:
        # For step down, we want to reach below the threshold (assuming threshold is between initial and final)
        # To be safe, just look at absolute distance
        initial_val = y[0]
        final_val = y_ref[-1]
        target_val = initial_val + (final_val - initial_val) * threshold
        if final_val > initial_val:
            indices = np.where(y >= target_val)[0]
        else:
            indices = np.where(y <= target_val)[0]
            
    if len(indices) > 0:
        return float(indices[0])
    return float(len(y))


def overshoot(y: np.ndarray, y_ref: np.ndarray) -> float:
    """
    Computes the maximum overshoot beyond the final reference value,
    expressed as a percentage of the step size.
    
    Args:
        y (np.ndarray): Generated feature values.
        y_ref (np.ndarray): Reference feature values.
        
    Returns:
        float: Overshoot percentage.
    """
    if len(y) == 0 or len(y_ref) == 0:
        return 0.0
        
    final_ref = y_ref[-1]
    initial_val = y[0]
    step_size = abs(final_ref - initial_val)
    
    if step_size < 1e-6:
        return 0.0
        
    if final_ref > initial_val:
        max_val = np.max(y)
        if max_val > final_ref:
            return float((max_val - final_ref) / step_size)
    else:
        min_val = np.min(y)
        if min_val < final_ref:
            return float((final_ref - min_val) / step_size)
            
    return 0.0


def settling_time(y: np.ndarray, y_ref: np.ndarray, band: float = 0.05) -> float:
    """
    Computes the settling time, which is the time required for the error to
    fall within and remain within a specified error band around the reference.
    
    Args:
        y (np.ndarray): Generated feature values.
        y_ref (np.ndarray): Reference feature values.
        band (float): Allowable error band as a fraction of the reference value (or absolute if ref is 0).
        
    Returns:
        float: Settling time (in steps). Returns len(y) if it never settles.
    """
    if len(y) == 0 or len(y_ref) == 0:
        return 0.0
        
    final_ref = y_ref[-1]
    if abs(final_ref) > 1e-6:
        allowed_error = abs(final_ref) * band
    else:
        allowed_error = band
        
    errors = np.abs(y - y_ref)
    settled_indices = np.where(errors <= allowed_error)[0]
    
    if len(settled_indices) == 0:
        return float(len(y))
        
    # We want the index after which it *stays* settled
    for i in range(len(y)):
        if np.all(errors[i:] <= allowed_error):
            return float(i)
            
    return float(len(y))
