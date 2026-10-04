"""
Visualization utilities for evaluating generated music and training progress.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tracking(y: np.ndarray, y_ref: np.ndarray, feature_name: str, save_path: str) -> None:
    """
    Plots the generated feature trajectory against the reference trajectory.
    
    Args:
        y (np.ndarray): Generated feature values.
        y_ref (np.ndarray): Reference feature values.
        feature_name (str): Name of the feature for the plot title and labels.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    
    t = np.arange(len(y))
    plt.plot(t, y_ref, 'k--', linewidth=2, label='Reference')
    plt.plot(t, y, 'b-', linewidth=1.5, alpha=0.8, label='Generated')
    
    plt.title(f'Tracking Performance: {feature_name}', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Feature Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(metrics_csv: str, save_dir: str) -> None:
    """
    Plots training curves from a metrics CSV file (e.g., from MetricsStore).
    
    Args:
        metrics_csv (str): Path to the metrics CSV file.
        save_dir (str): Directory to save the generated plots.
    """
    if not os.path.exists(metrics_csv):
        print(f"Warning: Metrics file {metrics_csv} not found.")
        return
        
    try:
        df = pd.read_csv(metrics_csv)
    except Exception as e:
        print(f"Error reading {metrics_csv}: {e}")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    exclude_cols = ['step', 'epoch', 'timestamp', 'time']
    plot_cols = [col for col in df.columns if col not in exclude_cols]
    
    x_col = 'step' if 'step' in df.columns else df.index
    x_label = 'Step' if 'step' in df.columns else 'Iteration'
    
    for col in plot_cols:
        plt.figure(figsize=(10, 6))
        
        if len(df) > 10:
            window = max(3, len(df) // 20)
            smoothed = df[col].rolling(window=window, min_periods=1).mean()
            x_data = df[x_col] if isinstance(x_col, str) else x_col
            plt.plot(x_data, df[col], alpha=0.3, color='blue', label='Raw')
            plt.plot(x_data, smoothed, color='blue', linewidth=2, label=f'Smoothed (w={window})')
        else:
            x_data = df[x_col] if isinstance(x_col, str) else x_col
            plt.plot(x_data, df[col], marker='o', color='blue', label=col)
            
        plt.title(f'Training Progress: {col}', fontsize=14)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(col.replace('_', ' ').title(), fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        safe_name = col.replace('/', '_')
        save_path = os.path.join(save_dir, f'{safe_name}_curve.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print(f"Saved {len(plot_cols)} training curve plots to {save_dir}")
