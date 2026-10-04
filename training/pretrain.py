import argparse
import pathlib
import time
import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional

from configs.config import Config, get_config
from utils.logging import ExperimentLogger
from utils.metrics_store import PretrainMetricsStore
from utils.checkpoint import CheckpointManager
from data.dataset import build_dataloaders
from models.cp_transformer import CPTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: MLE Pretraining for CP Transformer")
    parser.add_argument("--config", type=str, default=None, help="Path to config override")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run ID for logging")
    return parser.parse_args()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Create a learning rate schedule with warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(config: Config, args: argparse.Namespace):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    run_id = args.run_id if args.run_id else f"pretrain_{int(time.time())}"
    
    # 1. Setup logging and metrics
    log_dir = pathlib.Path("logs") / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(str(log_dir))
    metrics_store = PretrainMetricsStore(str(log_dir))
    ckpt_manager = CheckpointManager(str(pathlib.Path("checkpoints") / run_id))
    
    logger.log_info(f"Starting pretraining run: {run_id} on {device}")
    
    # 2. DataLoaders
    train_loader, val_loader = build_dataloaders(config.data, batch_size=config.training.batch_size)
    logger.log_info(f"Loaded {len(train_loader)} training batches and {len(val_loader)} validation batches.")
    
    # 3. Model
    model = CPTransformer(config.transformer, config.vocab, config.data)
    model.to(device)
    
    # 4. Optimizer and Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, 
                                  weight_decay=config.training.weight_decay)
    scaler = GradScaler()
    
    # 5. LR Schedule
    total_steps = config.training.max_steps
    warmup_steps = config.training.warmup_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # 6. Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        logger.log_info(f"Resuming from checkpoint: {args.resume}")
        start_step = ckpt_manager.load(args.resume, model, optimizer, scheduler, scaler)
        
    # 7. Training loop
    model.train()
    step = start_step
    data_iter = iter(train_loader)
    
    logger.log_info("Starting training loop...")
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        step_start_time = time.time()
        
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward and Loss
        optimizer.zero_grad()
        with autocast():
            # Assume model.compute_loss returns total_loss and a dict of individual losses
            total_loss, loss_dict = model.compute_loss(batch)
            
        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        step_time = time.time() - step_start_time
        current_lr = scheduler.get_last_lr()[0]
        
        # Log every step
        metrics = {
            "loss_total": total_loss.item(),
            "loss_bar": loss_dict.get("loss_bar", 0.0),
            "loss_position": loss_dict.get("loss_position", 0.0),
            "loss_pitch": loss_dict.get("loss_pitch", 0.0),
            "loss_duration": loss_dict.get("loss_duration", 0.0),
            "lr": current_lr,
            "grad_norm": grad_norm.item(),
            "step_time": step_time
        }
        
        metrics_store.add_train_step(step, metrics)
        logger.log_metrics(step, metrics)
        
        # Evaluate
        if step > 0 and step % config.training.eval_every_n_steps == 0:
            model.eval()
            val_loss = 0.0
            val_loss_dict = {"loss_bar": 0.0, "loss_position": 0.0, "loss_pitch": 0.0, "loss_duration": 0.0}
            num_val_batches = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    with autocast():
                        v_loss, v_ldict = model.compute_loss(val_batch)
                    val_loss += v_loss.item()
                    for k in val_loss_dict:
                        val_loss_dict[k] += v_ldict.get(k, 0.0)
                    num_val_batches += 1
                    
            if num_val_batches > 0:
                val_loss /= num_val_batches
                for k in val_loss_dict:
                    val_loss_dict[k] /= num_val_batches
                    
                val_metrics = {"val_loss_total": val_loss}
                val_metrics.update({f"val_{k}": v for k, v in val_loss_dict.items()})
                
                metrics_store.add_eval_step(step, val_metrics)
                logger.log_metrics(step, val_metrics, prefix="Eval")
                
            model.train()
            
        # Save Checkpoint
        if step > 0 and step % config.training.save_every_n_steps == 0:
            ckpt_manager.save(step, model, optimizer, scheduler, scaler)
            metrics_store.save()
            logger.log_info(f"Saved checkpoint and metrics at step {step}")
            
        step += 1
        
    # Final save
    ckpt_manager.save(total_steps, model, optimizer, scheduler, scaler)
    metrics_store.save()
    logger.log_info("Pretraining completed successfully.")

if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    # Apply potential config overrides here if needed
    train(cfg, args)
