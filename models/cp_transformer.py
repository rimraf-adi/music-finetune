import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import Config, VocabConfig, TransformerConfig, get_config

class CPTransformer(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        if config is None:
            config = get_config()
        self.config = config
        self.vocab = config.vocab
        self.tx_cfg = config.transformer
        
        # 4 separate nn.Embedding tables, one per CP attribute
        self.bar_embed = nn.Embedding(self.vocab.bar_size, self.tx_cfg.attr_embed_dim)
        self.pos_embed = nn.Embedding(self.vocab.position_size, self.tx_cfg.attr_embed_dim)
        self.pitch_embed = nn.Embedding(self.vocab.pitch_size, self.tx_cfg.attr_embed_dim)
        self.dur_embed = nn.Embedding(self.vocab.duration_size, self.tx_cfg.attr_embed_dim)
        
        # Linear projection to d_model
        self.proj_in = nn.Linear(4 * self.tx_cfg.attr_embed_dim, self.tx_cfg.d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(config.data.max_seq_len, self.tx_cfg.d_model)
        
        # Transformer decoder layers
        # Using nn.TransformerEncoder with a causal mask to act as an autoregressive decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.tx_cfg.d_model,
            nhead=self.tx_cfg.n_heads,
            dim_feedforward=self.tx_cfg.d_ff,
            dropout=self.tx_cfg.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.tx_cfg.n_layers)
        
        # 4 independent linear output heads
        self.bar_head = nn.Linear(self.tx_cfg.d_model, self.vocab.bar_size)
        self.pos_head = nn.Linear(self.tx_cfg.d_model, self.vocab.position_size)
        self.pitch_head = nn.Linear(self.tx_cfg.d_model, self.vocab.pitch_size)
        self.dur_head = nn.Linear(self.tx_cfg.d_model, self.vocab.duration_size)
        
        self._log_params()
        
    def _log_params(self):
        print(f"CPTransformer Parameter Count: {self.param_count():,}")
        
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, bar: torch.Tensor, pos: torch.Tensor, pitch: torch.Tensor, dur: torch.Tensor, ref_embeddings: torch.Tensor = None, mask: torch.Tensor = None) -> dict:
        """
        Args:
            bar, pos, pitch, dur: LongTensors of shape (batch_size, seq_len)
            ref_embeddings: Optional Tensor of shape (batch_size, num_bars, d_model)
            mask: Optional causal mask
        Returns:
            dict with logits for each attribute
        """
        device = bar.device
        b_sz, seq_len = bar.shape
        
        # Embeddings
        e_bar = self.bar_embed(bar)
        e_pos = self.pos_embed(pos)
        e_pitch = self.pitch_embed(pitch)
        e_dur = self.dur_embed(dur)
        
        # Concatenate 4 embeddings -> 512-dim, then linear projection to d_model (512)
        e_concat = torch.cat([e_bar, e_pos, e_pitch, e_dur], dim=-1)
        x = self.proj_in(e_concat)
        
        # Learned positional encoding
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(b_sz, seq_len)
        x = x + self.pos_encoding(positions)
        
        num_ref = 0
        if ref_embeddings is not None:
            num_ref = ref_embeddings.size(1)
            x = torch.cat([ref_embeddings, x], dim=1)
            
        total_len = num_ref + seq_len
        
        # Causal mask (upper triangular) to enforce autoregressive generation
        if mask is None:
            mask = self._generate_square_subsequent_mask(total_len, device)
            
        # Pass through Transformer
        x = self.transformer(x, mask=mask, is_causal=True)
        
        if num_ref > 0:
            x = x[:, num_ref:, :]
            
        # Output heads
        logits_bar = self.bar_head(x)
        logits_pos = self.pos_head(x)
        logits_pitch = self.pitch_head(x)
        logits_dur = self.dur_head(x)
        
        return {
            'bar': logits_bar,
            'position': logits_pos,
            'pitch': logits_pitch,
            'duration': logits_dur
        }

    def compute_loss(self, logits_dict: dict, targets_dict: dict) -> dict:
        """
        Args:
            logits_dict: Dict with logits from forward()
            targets_dict: Dict with target tokens
        Returns:
            Dict with per-attribute losses and total loss
        """
        loss_dict = {}
        total_loss = 0.0
        
        for attr in ['bar', 'position', 'pitch', 'duration']:
            logits = logits_dict[attr]
            targets = targets_dict[attr]
            
            # Cross-entropy loss per attribute head, ignore pad_id=0
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1), 
                ignore_index=self.vocab.pad_id
            )
            loss_dict[f'loss_{attr}'] = loss
            total_loss = total_loss + loss
            
        loss_dict['loss_total'] = total_loss
        return loss_dict

    @torch.no_grad()
    def generate(self, prompt_bar: torch.Tensor, prompt_pos: torch.Tensor, prompt_pitch: torch.Tensor, prompt_dur: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_p: float = 0.95, ref_embeddings: torch.Tensor = None) -> dict:
        """
        Autoregressive generation with nucleus sampling
        """
        self.eval()
        bar, pos, pitch, dur = prompt_bar, prompt_pos, prompt_pitch, prompt_dur
        
        for _ in range(max_new_tokens):
            logits_dict = self.forward(bar, pos, pitch, dur, ref_embeddings=ref_embeddings)
            
            next_tokens = {}
            for attr in ['bar', 'position', 'pitch', 'duration']:
                # Get logits for the last token
                logits = logits_dict[attr][:, -1, :] / temperature
                
                # Nucleus sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_tokens[attr] = next_token
                
            # Append generated tokens
            bar = torch.cat([bar, next_tokens['bar']], dim=1)
            pos = torch.cat([pos, next_tokens['position']], dim=1)
            pitch = torch.cat([pitch, next_tokens['pitch']], dim=1)
            dur = torch.cat([dur, next_tokens['duration']], dim=1)
            
        return {
            'bar': bar,
            'position': pos,
            'pitch': pitch,
            'duration': dur
        }

    def log_probs(self, bar: torch.Tensor, pos: torch.Tensor, pitch: torch.Tensor, dur: torch.Tensor, ref_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Compute sum of log-probabilities across all 4 attribute heads.
        Returns: Tensor of shape (batch, seq_len)
        """
        logits_dict = self.forward(bar, pos, pitch, dur, ref_embeddings=ref_embeddings)
        total_log_probs = 0
        
        for attr, tokens in [('bar', bar), ('position', pos), ('pitch', pitch), ('duration', dur)]:
            logits = logits_dict[attr]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather log probs for the actual tokens
            token_log_probs = torch.gather(log_probs, 2, tokens.unsqueeze(2)).squeeze(2)
            total_log_probs = total_log_probs + token_log_probs
            
        return total_log_probs

    def entropy(self, bar: torch.Tensor, pos: torch.Tensor, pitch: torch.Tensor, dur: torch.Tensor, ref_embeddings: torch.Tensor = None) -> dict:
        """
        Per-attribute entropy of the output distribution.
        """
        logits_dict = self.forward(bar, pos, pitch, dur, ref_embeddings=ref_embeddings)
        entropy_dict = {}
        
        vocab_sizes = {
            'bar': self.vocab.bar_size,
            'position': self.vocab.position_size,
            'pitch': self.vocab.pitch_size,
            'duration': self.vocab.duration_size
        }
        
        total_norm_entropy = 0.0
        
        for attr in ['bar', 'position', 'pitch', 'duration']:
            logits = logits_dict[attr]
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # H = -sum(p * log(p))
            h = -torch.sum(probs * log_probs, dim=-1)
            entropy_dict[f'entropy_{attr}'] = h
            
            # Normalize by log(vocab_size)
            norm_h = h / torch.log(torch.tensor(vocab_sizes[attr], dtype=torch.float32, device=h.device))
            total_norm_entropy = total_norm_entropy + norm_h.mean()
            
        entropy_dict['entropy_normalized'] = total_norm_entropy / 4.0
        return entropy_dict
