"""
Centralized configuration for the entire project.
All hyperparameters in one place — matching the conference paper defaults,
with journal-scale upgrades where noted.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os


# ── Vocabulary ──────────────────────────────────────────────────────

@dataclass
class VocabConfig:
    """CP token vocabulary sizes.  |X| = 4 × 18 × 90 × 66"""
    bar_size: int = 4        # PAD=0, NEW_BAR=1, CONT=2, EOS=3
    position_size: int = 18  # PAD=0, pos_0..pos_15=1..16, EOS=17
    pitch_size: int = 90     # PAD=0, MIDI 21..108 → 1..88, EOS=89
    duration_size: int = 66  # PAD=0, dur_1..dur_64 → 1..64, EOS=65

    # Special token IDs (shared across attributes)
    pad_id: int = 0
    eos_id_bar: int = 3
    eos_id_pos: int = 17
    eos_id_pitch: int = 89
    eos_id_dur: int = 65

    @property
    def sizes(self) -> List[int]:
        return [self.bar_size, self.position_size, self.pitch_size, self.duration_size]

    @property
    def attr_names(self) -> List[str]:
        return ["bar", "position", "pitch", "duration"]


# ── Data ────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    # Dataset
    dataset_name: str = "atepp"
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    # Tokenization
    resolution: int = 16           # 16th note quantization
    max_bars_per_sequence: int = 32
    max_seq_len: int = 512         # tokens per sequence

    # Splits (deterministic)
    split_seed: int = 52
    pretrain_fraction: float = 0.75  # 75% pretrain, 25% GRPO

    # MIDI processing
    min_pitch: int = 21   # Piano lowest A0
    max_pitch: int = 108  # Piano highest C8
    max_duration_steps: int = 64  # in 16th notes = 4 whole notes


# ── CP Transformer (Plant) ──────────────────────────────────────────

@dataclass
class TransformerConfig:
    """~26M parameters matching conference paper Table I."""
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    attr_embed_dim: int = 128      # per-attribute embedding dim
    max_seq_len: int = 512
    dropout: float = 0.1

    @property
    def total_embed_input(self) -> int:
        """4 attributes × 128 = 512, projected to d_model."""
        return 4 * self.attr_embed_dim


# ── MidiBERT Reward Sensor ──────────────────────────────────────────

@dataclass
class RewardModelConfig:
    # MidiBERT backbone
    midibert_name: str = "wazenmai/MIDI-BERT"
    midibert_hidden: int = 768
    midibert_layers: int = 12

    # Reward head MLP
    reward_head_dims: List[int] = field(default_factory=lambda: [768, 256, 64, 1])

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_layers: int = 4  # last N attention layers

    # Training
    lr: float = 1e-4
    epochs: int = 10
    batch_size: int = 32
    margin: float = 1.0  # Bradley-Terry margin


# ── GRPO Controller ─────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    """Hyperparameters from conference paper Table II, scaled for journal."""
    # Scale
    total_steps: int = 3000         # conference: 1500, journal: 3000
    batch_size: int = 4             # B — prompts per step (conference: 2)
    group_size: int = 8             # G — completions per prompt (conference: 6)
    prompt_len: int = 64            # tokens
    completion_len: int = 128       # tokens

    # PPO-style clipping
    clip_range: float = 0.2         # ε

    # Regularization
    kl_coeff: float = 0.03          # β
    entropy_coeff: float = 0.02     # λ
    entropy_target: float = 0.55    # H_target

    # Reward weighting
    tracking_weight: float = 1.0    # weight on tracking error reward
    quality_weight: float = 0.1     # α — weight on MidiBERT quality reward

    # Optimizer
    lr_init: float = 7.5e-6
    lr_final: float = 7.5e-7
    lr_schedule: str = "cosine"
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # Sampling
    top_p: float = 0.95             # nucleus sampling
    temperature: float = 1.0


# ── Pretraining ─────────────────────────────────────────────────────

@dataclass
class PretrainConfig:
    epochs: int = 100
    batch_size: int = 64            # A5000 24GB can handle this
    lr: float = 3e-4
    lr_schedule: str = "cosine"
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    fp16: bool = True               # mixed precision
    gradient_accumulation_steps: int = 2
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 2000
    log_every_n_steps: int = 100


# ── Musical Feature Trajectories ────────────────────────────────────

@dataclass
class MFTConfig:
    """Configuration for the 4 measurable output channels."""
    # Feature normalization ranges (for weighted tracking error)
    nd_range: Tuple[float, float] = (0.0, 20.0)    # note density per bar
    pc_range: Tuple[float, float] = (21.0, 108.0)   # MIDI pitch
    tt_range: Tuple[float, float] = (0.0, 1.0)      # tonal tension
    rc_range: Tuple[float, float] = (0.0, 1.0)      # rhythmic complexity

    # Tracking error weights W = diag(w_nd, w_pc, w_tt, w_rc)
    weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])


# ── Evaluation ──────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    num_eval_samples: int = 100
    eval_batch_size: int = 8
    # Control metrics
    settling_band: float = 0.05       # 5% band for settling time
    rise_threshold: float = 0.9       # 90% for rise time
    steady_state_fraction: float = 0.25  # last 25% of bars


# ── Master Config ───────────────────────────────────────────────────

@dataclass
class Config:
    # Sub-configs
    vocab: VocabConfig = field(default_factory=VocabConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    mft: MFTConfig = field(default_factory=MFTConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Global
    seed: int = 52
    device: str = "cuda"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    wandb_project: str = "music-ref-tracking"
    wandb_enabled: bool = True

    def ensure_dirs(self):
        for d in [self.output_dir, self.checkpoint_dir, self.log_dir,
                  self.data.data_dir, self.data.processed_dir]:
            os.makedirs(d, exist_ok=True)


def get_config(**overrides) -> Config:
    """Create config with optional overrides for quick experiments."""
    cfg = Config()
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    cfg.ensure_dirs()
    return cfg
