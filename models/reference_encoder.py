import torch
import torch.nn as nn
from configs.config import Config, get_config

class ReferenceEncoder(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        if config is None:
            config = get_config()
            
        self.d_model = config.transformer.d_model
        
        # 4 embedding tables for the discretized reference features
        self.nd_embed = nn.Embedding(21, self.d_model // 4)
        self.pc_embed = nn.Embedding(22, self.d_model // 4)
        self.tt_embed = nn.Embedding(11, self.d_model // 4)
        self.rc_embed = nn.Embedding(11, self.d_model // 4)
        
        self.embed_dim = self.d_model // 4
        self.proj = nn.Linear(4 * self.embed_dim, self.d_model)
        
        self._log_params()
        
    def _log_params(self):
        print(f"ReferenceEncoder Parameter Count: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def discretize_features(self, y_ref: torch.Tensor) -> dict:
        """
        Discretize continuous/discrete features into bin indices.
        y_ref: (batch, num_bars, 4) - [note_density, pitch_centroid, tonal_tension, rhythmic_complexity]
        
        IMPORTANT: Expects RAW (unnormalized) values:
            - note_density: 0-20 (notes per bar)
            - pitch_centroid: 21-108 (MIDI pitch)
            - tonal_tension: 0.0-1.0
            - rhythmic_complexity: 0.0-1.0
        """
        # note_density: bin into 21 levels (0-20 notes/bar)
        nd = torch.clamp(torch.round(y_ref[..., 0]), 0, 20).long()
        
        # pitch_centroid: bin into 22 levels (quantize MIDI 21-108 into ~4-note bins)
        pc = torch.clamp(y_ref[..., 1], 21.0, 108.0)
        pc_bins = torch.round((pc - 21.0) / 4.0).long()
        pc_bins = torch.clamp(pc_bins, 0, 21)
        
        # tonal_tension: bin into 11 levels (0.0-1.0 in 0.1 steps)
        tt = torch.clamp(torch.round(y_ref[..., 2] * 10.0), 0, 10).long()
        
        # rhythmic_complexity: bin into 11 levels (0.0-1.0 in 0.1 steps)
        rc = torch.clamp(torch.round(y_ref[..., 3] * 10.0), 0, 10).long()
        
        return {'nd': nd, 'pc': pc_bins, 'tt': tt, 'rc': rc}

    def forward(self, y_ref: torch.Tensor) -> torch.Tensor:
        """
        Converts y_ref into tokens that can be prepended to the CP Transformer input.
        Args:
            y_ref: Tensor of shape (batch, num_bars, 4)
        Returns:
            Tensor of shape (batch, num_bars, d_model)
        """
        bins = self.discretize_features(y_ref)
        
        e_nd = self.nd_embed(bins['nd'])
        e_pc = self.pc_embed(bins['pc'])
        e_tt = self.tt_embed(bins['tt'])
        e_rc = self.rc_embed(bins['rc'])
        
        # Concatenate and project
        e_concat = torch.cat([e_nd, e_pc, e_tt, e_rc], dim=-1)
        out = self.proj(e_concat)
        
        return out
