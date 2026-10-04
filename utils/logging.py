"""
Structured JSON-Lines logger for training and evaluation.

Design principles:
    1. Every logged record is a single JSON line → trivial to parse later
    2. Each phase gets its own log file (pretrain, reward, grpo, eval)
    3. All metadata (config, git hash, timestamps) recorded at init
    4. Logs are append-only, crash-safe (flush after every write)
    5. Designed to be the SOLE data source for all paper figures

File layout created under `logs/<run_id>/`:
    meta.json                   – config snapshot + environment info
    pretrain_train.jsonl        – per-step pretraining metrics
    pretrain_eval.jsonl         – per-evaluation pretraining metrics
    reward_train.jsonl          – reward model training
    grpo_train.jsonl            – per-step GRPO metrics
    grpo_eval.jsonl             – periodic GRPO evaluation
    generation_samples.jsonl    – individual generated sequences + features
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class JsonLogger:
    """Append-only JSONL logger for a single log stream."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "a", encoding="utf-8")
        self._step_count = 0

    def log(self, record: Dict[str, Any], step: Optional[int] = None):
        """Write one JSON line. Adds timestamp and step automatically."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step if step is not None else self._step_count,
            **record,
        }
        self._file.write(json.dumps(entry, default=_json_serializer) + "\n")
        self._file.flush()
        self._step_count += 1

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ExperimentLogger:
    """
    Master logger that manages all log streams for a single run.

    Usage:
        logger = ExperimentLogger(run_id="grpo_v1", log_dir="logs", config=cfg)
        logger.pretrain_train.log({"loss": 4.2, "lr": 3e-4}, step=100)
        logger.grpo_train.log({"reward_mean": 2.5, "kl": 0.15}, step=50)
        logger.close()
    """

    def __init__(
        self,
        run_id: str,
        log_dir: str = "logs",
        config: Optional[Any] = None,
    ):
        self.run_id = run_id
        self.run_dir = Path(log_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
        }
        if config is not None:
            meta["config"] = _dataclass_to_dict(config)

        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=_json_serializer)

        # Create log streams
        self.pretrain_train = JsonLogger(self.run_dir / "pretrain_train.jsonl")
        self.pretrain_eval = JsonLogger(self.run_dir / "pretrain_eval.jsonl")
        self.reward_train = JsonLogger(self.run_dir / "reward_train.jsonl")
        self.grpo_train = JsonLogger(self.run_dir / "grpo_train.jsonl")
        self.grpo_eval = JsonLogger(self.run_dir / "grpo_eval.jsonl")
        self.generation_samples = JsonLogger(self.run_dir / "generation_samples.jsonl")

        self._all_loggers = [
            self.pretrain_train, self.pretrain_eval,
            self.reward_train,
            self.grpo_train, self.grpo_eval,
            self.generation_samples,
        ]

    def close(self):
        for logger in self._all_loggers:
            logger.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Helpers ─────────────────────────────────────────────────────────

def _json_serializer(obj):
    """Handle non-JSON-serializable types."""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return _dataclass_to_dict(obj)
    return str(obj)


def _dataclass_to_dict(obj) -> dict:
    """Recursively convert dataclass to dict."""
    if not hasattr(obj, "__dataclass_fields__"):
        return obj
    result = {}
    for field_name in obj.__dataclass_fields__:
        value = getattr(obj, field_name)
        if hasattr(value, "__dataclass_fields__"):
            result[field_name] = _dataclass_to_dict(value)
        else:
            result[field_name] = value
    return result
