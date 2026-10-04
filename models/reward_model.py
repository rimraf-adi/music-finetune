import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from peft import get_peft_model, LoraConfig, TaskType

class RewardModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        # Fallback fresh BERT-base config
        bert_config = BertConfig(
            vocab_size=30000, 
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
        )
        
        try:
            # Load pretrained MidiBERT-Piano
            self.bert = BertModel.from_pretrained('wazenmai/MIDI-BERT')
        except Exception:
            print("Could not load wazenmai/MIDI-BERT, falling back to fresh BERT-base model")
            self.bert = BertModel(bert_config)
            
        # Apply LoRA to last 4 attention layers (using peft)
        # target modules: 'query', 'value' in attention for layers 8 to 11
        target_modules = []
        for i in range(8, 12):
            target_modules.extend([
                f"encoder.layer.{i}.attention.self.query",
                f"encoder.layer.{i}.attention.self.value"
            ])
            
        lora_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1
        )
        
        self.bert = get_peft_model(self.bert, lora_config)
        
        # Reward MLP head: 768 -> 256 -> 64 -> 1
        self.reward_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Wrapper to take CP token sequences as input
        from configs.config import get_config
        if config is None:
            config = get_config()
            
        # Simple embeddings to project CP tokens into BERT's 768-dim input space
        self.bar_embed = nn.Embedding(config.vocab.bar_size, 192)
        self.pos_embed = nn.Embedding(config.vocab.position_size, 192)
        self.pitch_embed = nn.Embedding(config.vocab.pitch_size, 192)
        self.dur_embed = nn.Embedding(config.vocab.duration_size, 192)
        
        self.freeze_backbone()
        self._log_params()
        
    def _log_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"RewardModel Parameter Count: {total:,} (Trainable: {trainable:,})")
        
    def freeze_backbone(self):
        """Freeze all except LoRA + reward head."""
        # Note: PEFT handles freezing the backbone for LoRA, but we must ensure 
        # our custom embeddings and reward head are trainable.
        for param in self.reward_head.parameters():
            param.requires_grad = True
        for param in self.bar_embed.parameters():
            param.requires_grad = True
        for param in self.pos_embed.parameters():
            param.requires_grad = True
        for param in self.pitch_embed.parameters():
            param.requires_grad = True
        for param in self.dur_embed.parameters():
            param.requires_grad = True

    def forward(self, bar: torch.Tensor, pos: torch.Tensor, pitch: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bar, pos, pitch, dur: Tensors of shape (batch, seq_len)
        Returns:
            Tensor of shape (batch,) - scalar reward per sequence
        """
        e_bar = self.bar_embed(bar)
        e_pos = self.pos_embed(pos)
        e_pitch = self.pitch_embed(pitch)
        e_dur = self.dur_embed(dur)
        
        # Concatenate 4x192 -> 768-dim
        inputs_embeds = torch.cat([e_bar, e_pos, e_pitch, e_dur], dim=-1)
        
        # Pass through MidiBERT encoder
        outputs = self.bert(inputs_embeds=inputs_embeds)
        
        # Mean-pool the hidden states
        hidden_states = outputs.last_hidden_state # (batch, seq_len, 768)
        pooled = hidden_states.mean(dim=1) # (batch, 768)
        
        # Pass through reward MLP head
        reward = self.reward_head(pooled).squeeze(-1) # (batch,)
        
        return reward
