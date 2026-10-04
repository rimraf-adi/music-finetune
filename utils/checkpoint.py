"""
Checkpoint manager: save/load model state with full metadata.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """
    Saves and loads checkpoints with:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict
        - Training step / epoch
        - Metrics at save time
        - Full config snapshot

    Keeps only the last `max_to_keep` checkpoints to save disk.
    Always keeps the best checkpoint (by a tracked metric).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        best_metric: str = "eval_loss_total",
        best_mode: str = "min",  # "min" or "max"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.best_metric = best_metric
        self.best_mode = best_mode
        self._best_value = float("inf") if best_mode == "min" else float("-inf")
        self._saved_checkpoints: list = []

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint. Returns the save path."""
        ckpt_path = self.checkpoint_dir / f"step_{step:07d}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Model weights
        torch.save(model.state_dict(), ckpt_path / "model.pt")

        # Optimizer
        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

        # Scheduler
        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")

        # Metadata
        meta = {
            "step": step,
            "metrics": metrics or {},
            "extra": extra or {},
        }
        if config is not None and hasattr(config, "__dataclass_fields__"):
            from utils.logging import _dataclass_to_dict
            meta["config"] = _dataclass_to_dict(config)

        with open(ckpt_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Track checkpoints
        self._saved_checkpoints.append(str(ckpt_path))

        # Check if this is the best
        is_best = False
        if metrics and self.best_metric in metrics:
            val = metrics[self.best_metric]
            if self.best_mode == "min" and val < self._best_value:
                self._best_value = val
                is_best = True
            elif self.best_mode == "max" and val > self._best_value:
                self._best_value = val
                is_best = True

        if is_best:
            best_link = self.checkpoint_dir / "best"
            if best_link.exists():
                if best_link.is_symlink():
                    best_link.unlink()
                else:
                    shutil.rmtree(best_link)
            # On Windows, symlinks may not work — use a marker file instead
            with open(self.checkpoint_dir / "best_checkpoint.txt", "w") as f:
                f.write(str(ckpt_path))

        # Prune old checkpoints (keep best + last N)
        self._prune()

        return str(ckpt_path)

    def _prune(self):
        """Remove old checkpoints beyond max_to_keep."""
        if len(self._saved_checkpoints) <= self.max_to_keep:
            return

        # Read best path
        best_path = None
        best_file = self.checkpoint_dir / "best_checkpoint.txt"
        if best_file.exists():
            best_path = best_file.read_text().strip()

        # Remove oldest, but never the best
        to_remove = self._saved_checkpoints[: -self.max_to_keep]
        for path in to_remove:
            if path != best_path and Path(path).exists():
                shutil.rmtree(path)
        self._saved_checkpoints = self._saved_checkpoints[-self.max_to_keep:]

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Load a checkpoint. Returns the metadata dict."""
        ckpt_path = Path(checkpoint_path)

        # Model
        model.load_state_dict(
            torch.load(ckpt_path / "model.pt", map_location=device, weights_only=True)
        )

        # Optimizer
        if optimizer is not None and (ckpt_path / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(ckpt_path / "optimizer.pt", map_location=device, weights_only=True)
            )

        # Scheduler
        if scheduler is not None and (ckpt_path / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(ckpt_path / "scheduler.pt", map_location=device, weights_only=True)
            )

        # Metadata
        with open(ckpt_path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        return meta

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_file = self.checkpoint_dir / "best_checkpoint.txt"
        if not best_file.exists():
            raise FileNotFoundError("No best checkpoint found.")
        best_path = best_file.read_text().strip()
        return self.load(best_path, model, optimizer, scheduler, device)

    def get_latest(self) -> Optional[str]:
        """Get path of the most recent checkpoint."""
        if not self._saved_checkpoints:
            # Scan directory
            dirs = sorted(self.checkpoint_dir.glob("step_*"))
            if not dirs:
                return None
            return str(dirs[-1])
        return self._saved_checkpoints[-1]
