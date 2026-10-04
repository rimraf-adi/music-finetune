"""
MetricsStore: accumulate, aggregate, and persist metrics for paper figures.

Collects per-step scalars during training, then provides:
    - Running averages (smoothed curves)
    - Per-epoch / per-checkpoint summaries
    - Export to CSV / pandas DataFrame for plotting

This is the bridge between raw JSONL logs and matplotlib figures.
"""

import json
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MetricsStore:
    """
    In-memory accumulator for time-series metrics.

    Usage:
        store = MetricsStore()
        for step in range(1000):
            store.record(step, loss=4.2, reward=-3.1, kl=0.05)
        store.to_csv("metrics.csv")
        df = store.to_dataframe()  # requires pandas
    """

    def __init__(self):
        self._data: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def record(self, step: int, **metrics: float):
        """Record one or more metrics at a given step."""
        for key, value in metrics.items():
            if value is not None:
                self._data[key].append((step, float(value)))

    @property
    def keys(self) -> List[str]:
        return list(self._data.keys())

    def get(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (steps, values) arrays for a metric."""
        if key not in self._data:
            raise KeyError(f"Metric '{key}' not found. Available: {self.keys}")
        pairs = self._data[key]
        steps = np.array([p[0] for p in pairs])
        values = np.array([p[1] for p in pairs])
        return steps, values

    def smoothed(self, key: str, window: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Return exponentially smoothed values for plotting."""
        steps, values = self.get(key)
        if len(values) == 0:
            return steps, values
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        alpha = 2.0 / (window + 1)
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        return steps, smoothed

    def summary(self, key: str) -> Dict[str, float]:
        """Compute summary statistics for a metric."""
        _, values = self.get(key)
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "first": float(values[0]),
            "last": float(values[-1]),
            "median": float(np.median(values)),
        }

    def last_n_mean(self, key: str, n: int = 100) -> float:
        """Mean of the last n values — useful for steady-state metrics."""
        _, values = self.get(key)
        return float(np.mean(values[-n:]))

    def to_csv(self, filepath: str):
        """
        Export all metrics to a single CSV.
        Columns: step, metric_name, value
        (Long format — easy to pivot in pandas/R/matplotlib.)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "metric", "value"])
            for key in sorted(self._data.keys()):
                for step, value in self._data[key]:
                    writer.writerow([step, key, value])

    def to_dataframe(self):
        """Export to pandas DataFrame (long format)."""
        import pandas as pd
        rows = []
        for key in sorted(self._data.keys()):
            for step, value in self._data[key]:
                rows.append({"step": step, "metric": key, "value": value})
        return pd.DataFrame(rows)

    def to_wide_dataframe(self):
        """Export to pandas DataFrame (wide format — one column per metric)."""
        df = self.to_dataframe()
        return df.pivot_table(index="step", columns="metric", values="value")

    def save(self, filepath: str):
        """Persist store as JSON for later reload."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {k: [(s, v) for s, v in pairs] for k, pairs in self._data.items()},
                f,
            )

    @classmethod
    def load(cls, filepath: str) -> "MetricsStore":
        """Reload a saved MetricsStore."""
        store = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, pairs in data.items():
            store._data[key] = [(s, v) for s, v in pairs]
        return store


class GRPOMetricsStore(MetricsStore):
    """
    Specialized store for GRPO training that pre-defines
    all the metrics we need for the paper.

    Records per-step:
        Reward:    reward_mean, reward_std, reward_max, reward_min
        Tracking:  tracking_rmse, tracking_nd, tracking_pc, tracking_tt, tracking_rc
        Policy:    kl_divergence, entropy_normalized, clip_fraction
        Loss:      grpo_loss, policy_loss, kl_loss, entropy_loss
        Optimizer: learning_rate, grad_norm
        Timing:    step_time_sec, samples_per_sec
    """

    # Define all expected metric names for documentation
    METRIC_SCHEMA = {
        # Reward
        "reward_mean": "Mean reward across batch × group",
        "reward_std": "Std of rewards",
        "reward_max": "Max reward in batch",
        "reward_min": "Min reward in batch",
        "reward_tracking": "Tracking component of reward",
        "reward_quality": "MidiBERT quality component of reward",
        # Per-feature tracking error
        "tracking_rmse_total": "Total RMSE across all features",
        "tracking_rmse_nd": "RMSE for note density",
        "tracking_rmse_pc": "RMSE for pitch centroid",
        "tracking_rmse_tt": "RMSE for tonal tension",
        "tracking_rmse_rc": "RMSE for rhythmic complexity",
        # Policy health
        "kl_divergence": "KL(π_θ || π_ref)",
        "entropy_normalized": "Mean normalized entropy across attributes",
        "entropy_bar": "Entropy of bar attribute",
        "entropy_position": "Entropy of position attribute",
        "entropy_pitch": "Entropy of pitch attribute",
        "entropy_duration": "Entropy of duration attribute",
        "clip_fraction": "Fraction of ratio updates that were clipped",
        # Loss components
        "loss_total": "Total GRPO loss",
        "loss_policy": "Clipped surrogate loss",
        "loss_kl": "β × KL penalty",
        "loss_entropy": "λ × entropy penalty",
        # Optimizer
        "learning_rate": "Current learning rate",
        "grad_norm": "Gradient L2 norm (before clipping)",
        "grad_norm_clipped": "Gradient L2 norm (after clipping)",
        # Timing
        "step_time_sec": "Wall-clock time for this step",
        "samples_per_sec": "Throughput",
        # Advantage stats
        "advantage_mean": "Mean group-relative advantage",
        "advantage_std": "Std of advantages",
    }

    def record_grpo_step(self, step: int, metrics: Dict[str, float]):
        """Record a full GRPO step with validation."""
        self.record(step, **metrics)

    def get_paper_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Generate the summary table that goes into the paper.
        Returns stats for each recorded metric.
        """
        result = {}
        for key in self.keys:
            result[key] = self.summary(key)
        return result


class PretrainMetricsStore(MetricsStore):
    """Specialized store for pretraining phase."""

    METRIC_SCHEMA = {
        "loss_total": "Total cross-entropy loss (sum of 4 heads)",
        "loss_bar": "CE loss for bar attribute",
        "loss_position": "CE loss for position attribute",
        "loss_pitch": "CE loss for pitch attribute",
        "loss_duration": "CE loss for duration attribute",
        "eval_loss_total": "Evaluation total loss",
        "eval_loss_bar": "Evaluation bar loss",
        "eval_loss_position": "Evaluation position loss",
        "eval_loss_pitch": "Evaluation pitch loss",
        "eval_loss_duration": "Evaluation duration loss",
        "learning_rate": "Current learning rate",
        "grad_norm": "Gradient norm",
        "epoch": "Current epoch",
        "step_time_sec": "Time per step",
        "tokens_per_sec": "Training throughput",
    }
