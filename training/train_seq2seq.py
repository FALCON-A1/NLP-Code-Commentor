"""
Seq2Seq Training Script — Kaggle Optimized
============================================
Trains the Bidirectional LSTM Seq2Seq + Attention model
on the CodeSearchNet dataset for code summarization.

Optimizations for Kaggle 2x T4 GPUs:
- AMP (FP16 mixed precision) for ~2x speed + half memory
- cuDNN benchmark for optimized kernels
- Pinned memory + multi-worker DataLoaders
- DataParallel across multiple GPUs
- Larger batch size (fits in FP16 memory)
"""

import os
import sys
import json
import time
import math
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seq2seq import build_seq2seq_model
from src.preprocessing import Vocabulary, CodePreprocessor, TextPreprocessor
from src.code_parser import CodeTokenizer
from src.dataset import Seq2SeqDataset, collate_seq2seq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_vocabularies(train_path: str, min_freq: int = 2):
    """Build source (code) and target (docstring) vocabularies."""
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    code_tokenizer = CodeTokenizer()
    text_prep = TextPreprocessor()

    src_tokens = [code_tokenizer.tokenize_code(d["code"]) for d in data]
    tgt_tokens = [text_prep.tokenize_text(d["docstring"]) for d in data]

    src_vocab = Vocabulary(min_freq=min_freq, max_size=50000)
    tgt_vocab = Vocabulary(min_freq=min_freq, max_size=30000)
    src_vocab.build_vocab(src_tokens)
    tgt_vocab.build_vocab(tgt_tokens)

    return src_vocab, tgt_vocab


def train_epoch(model, loader, optimizer, criterion, clip, device, teacher_forcing, scaler, use_amp):
    """Run one training epoch with AMP support."""
    model.train()
    epoch_loss = 0

    for batch_idx, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        # Mixed precision forward pass
        with autocast('cuda', enabled=use_amp):
            output = model(src, tgt, teacher_forcing)
            output = output[:, 1:, :].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}")

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device, use_amp):
    """Evaluate model on validation/test set with AMP."""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            with autocast('cuda', enabled=use_amp):
                output = model(src, tgt, teacher_forcing_ratio=0)
                output = output[:, 1:, :].reshape(-1, output.shape[-1])
                tgt = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train Seq2Seq Model")
    parser.add_argument("--train_data", type=str, default="data/processed/train.json")
    parser.add_argument("--val_data", type=str, default="data/processed/validation.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--teacher_forcing", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=7)
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

    # Build vocabularies
    logger.info("Building vocabularies...")
    src_vocab, tgt_vocab = build_vocabularies(args.train_data)

    # Tokenizers
    code_tokenizer = CodeTokenizer()
    text_prep = TextPreprocessor()

    # Datasets with pinned memory and multiple workers for fast loading
    from torch.utils.data import DataLoader
    pin_memory = torch.cuda.is_available()
    train_ds = Seq2SeqDataset(
        args.train_data, src_vocab, tgt_vocab,
        code_tokenizer.tokenize_code, text_prep.tokenize_text
    )
    val_ds = Seq2SeqDataset(
        args.val_data, src_vocab, tgt_vocab,
        code_tokenizer.tokenize_code, text_prep.tokenize_text
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_seq2seq, num_workers=args.num_workers,
        pin_memory=pin_memory, persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_seq2seq, num_workers=args.num_workers,
        pin_memory=pin_memory, persistent_workers=args.num_workers > 0
    )
    logger.info(f"DataLoader: {args.num_workers} workers, pin_memory={pin_memory}")

    # Build model
    model = build_seq2seq_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        device=device,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    # Multi-GPU: wrap with DataParallel if more than 1 GPU
    if n_gpus > 1:
        logger.info(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(
        ignore_index=Vocabulary.PAD_IDX,
        label_smoothing=args.label_smoothing
    )

    # AMP gradient scaler (only active when use_amp=True)
    scaler = GradScaler('cuda', enabled=use_amp)

    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop with early stopping
    logger.info(f"Starting training... (batch_size={args.batch_size}, epochs={args.epochs})")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tf_ratio = max(0.1, args.teacher_forcing * (0.95 ** epoch))

        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  args.clip, device, tf_ratio, scaler, use_amp)
        val_loss = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step(val_loss)

        elapsed = time.time() - start
        gap = val_loss - train_loss
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Gap: {gap:.4f} | PPL: {math.exp(min(val_loss, 100)):.2f} | "
            f"TF: {tf_ratio:.2f} | Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save unwrapped model (without DataParallel wrapper) for portability
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'args': vars(args),
            }, os.path.join(args.checkpoint_dir, 'seq2seq_best.pth'))
            logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                logger.info(f"  Early stopping at epoch {epoch}!")
                break

    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
