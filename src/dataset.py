"""
PyTorch Dataset Module
======================
Custom Dataset and DataLoader classes for both
Seq2Seq and CodeBERT models.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class Seq2SeqDataset(Dataset):
    """
    Dataset for the Seq2Seq + Attention model.
    Returns tokenized and numericalized code-docstring pairs.
    """

    def __init__(self, data_path: str, src_vocab, tgt_vocab,
                 src_tokenizer, tgt_tokenizer,
                 max_src_len: int = 256, max_tgt_len: int = 128):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        # Tokenize
        src_tokens = self.src_tokenizer(item["code"])[:self.max_src_len]
        tgt_tokens = self.tgt_tokenizer(item["docstring"])[:self.max_tgt_len]

        # Numericalize
        src_indices = self.src_vocab.encode(src_tokens)
        tgt_indices = (
            [self.tgt_vocab.SOS_IDX]
            + self.tgt_vocab.encode(tgt_tokens)
            + [self.tgt_vocab.EOS_IDX]
        )

        return torch.tensor(src_indices, dtype=torch.long), \
               torch.tensor(tgt_indices, dtype=torch.long)


class CodeBERTDataset(Dataset):
    """
    Dataset for CodeBERT fine-tuning.
    Uses HuggingFace tokenizer for encoding.
    """

    def __init__(self, data_path: str, tokenizer,
                 max_src_len: int = 256, max_tgt_len: int = 128):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Encode source (code)
        source = self.tokenizer(
            item["code"],
            max_length=self.max_src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Encode target (docstring)
        target = self.tokenizer(
            item["docstring"],
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze(),
        }


def collate_seq2seq(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Custom collate function for Seq2Seq that pads sequences."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded


def get_seq2seq_loaders(
    train_path: str, val_path: str, test_path: str,
    src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for Seq2Seq model."""
    train_ds = Seq2SeqDataset(train_path, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer)
    val_ds = Seq2SeqDataset(val_path, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer)
    test_ds = Seq2SeqDataset(test_path, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_seq2seq)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq2seq)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq2seq)

    return train_loader, val_loader, test_loader


def get_codebert_loaders(
    train_path: str, val_path: str, test_path: str,
    tokenizer, batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for CodeBERT model."""
    train_ds = CodeBERTDataset(train_path, tokenizer)
    val_ds = CodeBERTDataset(val_path, tokenizer)
    test_ds = CodeBERTDataset(test_path, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
