"""
CodeBERT Training Script
=========================
Fine-tunes CodeBERT Encoder-Decoder for code summarization
on the CodeSearchNet dataset.

Improvements applied:
- Freeze early encoder layers (CodeBERT already knows code)
- Lower learning rate (1e-5) to avoid forgetting pre-trained knowledge
- Early stopping to prevent overfitting
- Higher weight decay (0.05) for regularization
- Fewer decoder layers (4) to reduce model capacity
"""

import os
import sys
import time
import logging
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.codebert_summarizer import build_codebert_model
from src.dataset import CodeBERTDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, scheduler, device, grad_accum_steps=1):
    """Run one training epoch."""
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"] / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_loss += outputs["loss"].item()

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)} - Loss: {outputs['loss'].item():.4f}")

    return epoch_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels)
            epoch_loss += outputs["loss"].item()

    return epoch_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT")
    parser.add_argument("--train_data", type=str, default="data/processed/train.json")
    parser.add_argument("--val_data", type=str, default="data/processed/validation.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--freeze_encoder", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model
    logger.info("Building CodeBERT model...")
    model = build_codebert_model(
        device=device,
        decoder_layers=args.decoder_layers,
        freeze_encoder_layers=args.freeze_encoder,
    )
    tokenizer = model.get_tokenizer()

    # Datasets
    train_ds = CodeBERTDataset(args.train_data, tokenizer)
    val_ds = CodeBERTDataset(args.val_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

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

    # Training with early stopping
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("Starting CodeBERT fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                  device, args.grad_accum)
        val_loss = evaluate(model, val_loader, device)
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
            save_path = os.path.join(args.checkpoint_dir, 'codebert_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
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
