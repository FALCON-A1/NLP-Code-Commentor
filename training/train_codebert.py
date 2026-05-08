"""
CodeBERT Training Script — Kaggle Optimized
=============================================
Fine-tunes CodeBERT Encoder-Decoder for code summarization
on the CodeSearchNet dataset.

Optimizations for Kaggle 2x T4 GPUs:
- AMP (FP16 mixed precision) for ~2x speed + half memory
- cuDNN benchmark for optimized kernels
- Pinned memory + multi-worker DataLoaders
- DataParallel across multiple GPUs
- Gradient accumulation for effective larger batch
"""

import os
import sys
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.codebert_summarizer import build_codebert_model
from src.dataset import CodeBERTDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, scheduler, device, grad_accum_steps, scaler, use_amp):
    """Run one training epoch with AMP support."""
    model.train()
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            # Handle DataParallel (loss is averaged per GPU, need mean across GPUs)
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / grad_accum_steps

        # Scaled backward pass
        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * grad_accum_steps

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item() * grad_accum_steps:.4f}")

    return epoch_loss / len(loader)


def evaluate(model, loader, device, use_amp):
    """Evaluate model on validation set with AMP."""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
                if loss.dim() > 0:
                    loss = loss.mean()
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT")
    parser.add_argument("--train_data", type=str, default="data/processed/train.json")
    parser.add_argument("--val_data", type=str, default="data/processed/validation.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--freeze_encoder", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # GPU detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    use_amp = torch.cuda.is_available() and not args.no_amp

    if torch.cuda.is_available():
        # Enable cuDNN benchmark for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        logger.info(f"Found {n_gpus} GPU(s):")
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")
        logger.info(f"Mixed Precision (AMP): {'ON' if use_amp else 'OFF'}")
        logger.info(f"cuDNN Benchmark: ON")
    else:
        logger.info("No GPU found, using CPU (AMP disabled)")

    # Build model
    logger.info("Building CodeBERT model...")
    model = build_codebert_model(
        device=device,
        decoder_layers=args.decoder_layers,
        freeze_encoder_layers=args.freeze_encoder,
    )
    tokenizer = model.get_tokenizer()

    # Multi-GPU: wrap with DataParallel if more than 1 GPU
    if n_gpus > 1:
        logger.info(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    # Datasets with pinned memory and multiple workers
    pin_memory = torch.cuda.is_available()
    train_ds = CodeBERTDataset(args.train_data, tokenizer)
    val_ds = CodeBERTDataset(args.val_data, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0
    )
    logger.info(f"DataLoader: {args.num_workers} workers, pin_memory={pin_memory}")

    # Optimizer with higher weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                     if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                     if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=args.lr)

    total_steps = max(1, len(train_loader) * args.epochs // args.grad_accum)
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)

    # AMP gradient scaler
    scaler = GradScaler(enabled=use_amp)

    # Training with early stopping
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    effective_batch = args.batch_size * args.grad_accum * max(1, n_gpus)
    logger.info(f"Starting CodeBERT fine-tuning...")
    logger.info(f"Effective batch size: {effective_batch} (batch={args.batch_size} x accum={args.grad_accum} x gpus={max(1, n_gpus)})")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                  device, args.grad_accum, scaler, use_amp)
        val_loss = evaluate(model, val_loader, device, use_amp)
        elapsed = time.time() - start

        gap = val_loss - train_loss
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Gap: {gap:.4f} | Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save unwrapped model (without DataParallel wrapper) for portability
            model_to_save = model.module if hasattr(model, 'module') else model
            save_path = os.path.join(args.checkpoint_dir, 'codebert_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, save_path)
            logger.info(f"  ✓ Saved best CodeBERT model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                logger.info(f"  Early stopping at epoch {epoch}!")
                break

    logger.info(f"CodeBERT fine-tuning complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
