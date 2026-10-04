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
    logger = ExperimentLogger(run_id=run_id, log_dir="logs", config=config)
    metrics_store = PretrainMetricsStore()
    ckpt_manager = CheckpointManager(str(pathlib.Path("checkpoints") / run_id))
    
    print(f"Starting pretraining run: {run_id} on {device}")
    
    # 2. DataLoaders
    import pickle
    with open(pathlib.Path(config.data.processed_dir) / "pretrain.pkl", "rb") as f:
        pretrain_seqs = pickle.load(f)
    
    # We only need pretrain seqs here, but build_dataloaders expects both
    loaders = build_dataloaders(config, pretrain_seqs, [])
    train_loader = loaders['train']
    val_loader = loaders['val']
    print(f"Loaded {len(train_loader)} training batches and {len(val_loader)} validation batches.")
    
    # 3. Model
    model = CPTransformer(config)
    model.to(device)
    
    # 4. Optimizer and Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain.lr, 
                                  weight_decay=config.pretrain.weight_decay)
    scaler = GradScaler()
    
    # 5. LR Schedule
    total_steps = config.pretrain.epochs * max(1, len(train_loader))
    warmup_steps = config.pretrain.warmup_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # 6. Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        meta = ckpt_manager.load(args.resume, model, optimizer, scheduler, device=str(device))
        start_step = meta.get("step", 0)
        
    # 7. Training loop
    model.train()
    step = start_step
    data_iter = iter(train_loader)
    
    print("Starting training loop...")
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        step_start_time = time.time()
        
        # Move batch to device (batch is a tuple of input_dict, target_dict)
        input_dict, target_dict = batch
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        target_dict = {k: v.to(device) for k, v in target_dict.items()}
        
        # Forward and Loss
        optimizer.zero_grad()
        with autocast():
            # Assume model.compute_loss returns total_loss and a dict of individual losses
            logits_dict = model(input_dict['bar'], input_dict['position'], input_dict['pitch'], input_dict['duration'])
            loss_dict = model.compute_loss(logits_dict, target_dict)
            total_loss = loss_dict['loss_total']
            
        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.pretrain.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        step_time = time.time() - step_start_time
        current_lr = scheduler.get_last_lr()[0]
        
        # Log every step
        metrics = {
            "loss_total": total_loss.item(),
            "loss_bar": loss_dict["loss_bar"].item() if isinstance(loss_dict.get("loss_bar"), torch.Tensor) else loss_dict.get("loss_bar", 0.0),
            "loss_position": loss_dict["loss_position"].item() if isinstance(loss_dict.get("loss_position"), torch.Tensor) else loss_dict.get("loss_position", 0.0),
            "loss_pitch": loss_dict["loss_pitch"].item() if isinstance(loss_dict.get("loss_pitch"), torch.Tensor) else loss_dict.get("loss_pitch", 0.0),
            "loss_duration": loss_dict["loss_duration"].item() if isinstance(loss_dict.get("loss_duration"), torch.Tensor) else loss_dict.get("loss_duration", 0.0),
            "lr": current_lr,
            "grad_norm": grad_norm.item(),
            "step_time": step_time
        }
        
        metrics_store.record(step, **metrics)
        logger.pretrain_train.log(metrics, step=step)
        
        # Evaluate
        if step > 0 and step % config.pretrain.eval_every_n_steps == 0:
            model.eval()
            val_loss = 0.0
            val_loss_dict = {"loss_bar": 0.0, "loss_position": 0.0, "loss_pitch": 0.0, "loss_duration": 0.0}
            num_val_batches = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    v_input, v_target = val_batch
                    v_input = {k: v.to(device) for k, v in v_input.items()}
                    v_target = {k: v.to(device) for k, v in v_target.items()}
                    with autocast():
                        v_logits = model(v_input['bar'], v_input['position'], v_input['pitch'], v_input['duration'])
                        v_ldict = model.compute_loss(v_logits, v_target)
                        v_loss = v_ldict['loss_total']
                    val_loss += v_loss.item()
                    for k in val_loss_dict:
                        val_loss_dict[k] += v_ldict.get(k, 0.0).item() if isinstance(v_ldict.get(k, 0.0), torch.Tensor) else v_ldict.get(k, 0.0)
                    num_val_batches += 1
                    
            if num_val_batches > 0:
                val_loss /= num_val_batches
                for k in val_loss_dict:
                    val_loss_dict[k] /= num_val_batches
                    
                val_metrics = {"val_loss_total": val_loss}
                val_metrics.update({f"val_{k}": v for k, v in val_loss_dict.items()})
                
                metrics_store.record(step, **val_metrics)
                logger.pretrain_eval.log(val_metrics, step=step)
                
            model.train()
            
        # Save Checkpoint
        if step > 0 and step % config.pretrain.save_every_n_steps == 0:
            ckpt_manager.save(step, model, optimizer, scheduler)
            metrics_store.save(str(log_dir / "metrics.json"))
            print(f"Saved checkpoint and metrics at step {step}")
            
        step += 1
        
    # Final save
    if total_steps > 0:
        ckpt_manager.save(total_steps, model, optimizer, scheduler)
    metrics_store.save(str(log_dir / "metrics.json"))
    print("Pretraining completed successfully.")

if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    # Apply potential config overrides here if needed
    train(cfg, args)
