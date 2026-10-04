import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import math

from configs.config import Config

class CPDataset(Dataset):
    """Dataset for Compound Word tokens."""
    
    def __init__(self, sequences: List[np.ndarray], prompt_len: Optional[int] = None):
        """
        Args:
            sequences: List of tokenized sequences (numpy arrays of shape (N, 4)).
            prompt_len: If provided, splits sequence into prompt and completion for GRPO.
        """
        self.sequences = sequences
        self.prompt_len = prompt_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Any:
        seq = self.sequences[idx]
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        
        if self.prompt_len is not None:
            # Handle prompt/completion split for GRPO
            prompt_length = min(self.prompt_len, len(seq_tensor) // 2)
            if prompt_length == 0:
                prompt_length = 1
                
            prompt = seq_tensor[:prompt_length]
            completion = seq_tensor[prompt_length:]
            
            prompt_dict = {
                'bar': prompt[:, 0],
                'position': prompt[:, 1],
                'pitch': prompt[:, 2],
                'duration': prompt[:, 3]
            }
            completion_dict = {
                'bar': completion[:, 0],
                'position': completion[:, 1],
                'pitch': completion[:, 2],
                'duration': completion[:, 3]
            }
            return prompt_dict, completion_dict
            
        else:
            # Handle pretraining teacher forcing
            inputs = seq_tensor[:-1]
            targets = seq_tensor[1:]
            
            input_dict = {
                'bar': inputs[:, 0],
                'position': inputs[:, 1],
                'pitch': inputs[:, 2],
                'duration': inputs[:, 3]
            }
            target_dict = {
                'bar': targets[:, 0],
                'position': targets[:, 1],
                'pitch': targets[:, 2],
                'duration': targets[:, 3]
            }
            return input_dict, target_dict

def collate_fn_pad(batch: List[Any]) -> Any:
    """Collate function that handles padding."""
    if len(batch[0]) == 2 and isinstance(batch[0][0], dict):
        is_grpo = 'prompt' in batch[0] # Not perfectly accurate, we just return Tuple of batched dicts
        
        def pad_dicts(dict_list):
            batched = {}
            for key in ['bar', 'position', 'pitch', 'duration']:
                tensors = [d[key] for d in dict_list]
                lengths = [t.size(0) for t in tensors]
                max_len = max(lengths)
                
                padded = []
                for t in tensors:
                    if max_len > t.size(0):
                        pad_tensor = torch.zeros(max_len - t.size(0), dtype=t.dtype)
                        padded.append(torch.cat([t, pad_tensor]))
                    else:
                        padded.append(t)
                batched[key] = torch.stack(padded)
            return batched
            
        part1 = pad_dicts([item[0] for item in batch])
        part2 = pad_dicts([item[1] for item in batch])
        return part1, part2
    return torch.utils.data.dataloader.default_collate(batch)

def build_dataloaders(config: Config, pretrain_seqs: List[np.ndarray], grpo_seqs: List[np.ndarray]) -> Dict[str, DataLoader]:
    """
    Build train, val, and grpo DataLoaders.
    
    Args:
        config: Master Config object.
        pretrain_seqs: Sequences for pretraining.
        grpo_seqs: Sequences for GRPO.
        
    Returns:
        Dict with 'train', 'val', and 'grpo' DataLoaders.
    """
    # Split pretrain into train/val (90/10)
    split_idx = int(len(pretrain_seqs) * 0.9)
    train_seqs = pretrain_seqs[:split_idx]
    val_seqs = pretrain_seqs[split_idx:]
    
    train_dataset = CPDataset(train_seqs)
    val_dataset = CPDataset(val_seqs)
    grpo_dataset = CPDataset(grpo_seqs, prompt_len=config.data.max_seq_len // 4)
    
    # Typically you get batch size from config, assuming 4 if not present
    batch_size = 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_pad
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_pad
    )
    grpo_loader = DataLoader(
        grpo_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_pad
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'grpo': grpo_loader
    }
