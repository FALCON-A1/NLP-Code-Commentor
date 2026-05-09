"""
CodeBERT Summarizer Model (Model 1) — v2.0
============================================
Fine-tunes Microsoft's CodeBERT for code summarization.
Uses an Encoder-Decoder architecture with CodeBERT as encoder
and a randomly initialized Transformer decoder.

v2.0 Changes:
- Fixed EncoderDecoderModel construction for new transformers API
- Load encoder/decoder separately, combine with EncoderDecoderModel()
- Added version logging to verify correct file is loaded
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version tag — if you see this in Kaggle logs, the correct file is loaded
CODEBERT_MODEL_VERSION = "2.0"
logger.info(f"=== CodeBERT Summarizer v{CODEBERT_MODEL_VERSION} loaded ===")

try:
    from transformers import (
        RobertaTokenizer,
        RobertaModel,
        RobertaConfig,
        EncoderDecoderModel,
        EncoderDecoderConfig,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("HuggingFace Transformers not installed.")


class CodeBERTSummarizer(nn.Module):
    """
    Code Summarization model using CodeBERT as the encoder
    and a randomly initialized Transformer decoder.

    Model 1 in our project — the primary deep learning model.
    """

    MODEL_NAME = "microsoft/codebert-base"

    def __init__(self, decoder_layers: int = 6, max_target_length: int = 128,
                 beam_size: int = 4, freeze_encoder_layers: int = 0):
        """
        Args:
            decoder_layers: Number of decoder transformer layers.
            max_target_length: Max tokens for generated summaries.
            beam_size: Beam search width for generation.
            freeze_encoder_layers: Number of encoder layers to freeze (0 = none).
        """
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Install transformers: pip install transformers")

        self.max_target_length = max_target_length
        self.beam_size = beam_size

        logger.info(f"[v{CODEBERT_MODEL_VERSION}] Loading CodeBERT from {self.MODEL_NAME}...")

        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.MODEL_NAME)

        # ============================================================
        # v2.0 FIX: Build encoder and decoder SEPARATELY, then combine.
        # Old API (broken): EncoderDecoderModel.from_encoder_decoder_pretrained()
        # New API (works):  EncoderDecoderModel(encoder=..., decoder=...)
        # ============================================================

        # Step 1: Load pre-trained CodeBERT as encoder
        logger.info(f"[v{CODEBERT_MODEL_VERSION}] Loading pre-trained encoder...")
        encoder = RobertaModel.from_pretrained(self.MODEL_NAME)

        # Step 2: Create fresh decoder with cross-attention
        logger.info(f"[v{CODEBERT_MODEL_VERSION}] Creating decoder with {decoder_layers} layers...")
        decoder_config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size,
            num_hidden_layers=decoder_layers,
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            is_decoder=True,
            add_cross_attention=True,
        )
        decoder = RobertaModel(decoder_config)

        # Step 3: Combine into Encoder-Decoder model
        logger.info(f"[v{CODEBERT_MODEL_VERSION}] Combining encoder + decoder...")
        self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        # Set generation config
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.max_length = max_target_length

        # Optionally freeze early encoder layers
        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[v{CODEBERT_MODEL_VERSION}] CodeBERT Summarizer: {total_params:,} total, {trainable:,} trainable")

    def _freeze_encoder_layers(self, n_layers: int):
        """Freeze the first n encoder layers."""
        for i, layer in enumerate(self.model.encoder.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        logger.info(f"Froze first {n_layers} encoder layers.")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass for training.

        Args:
            input_ids: (batch, src_len) tokenized source code.
            attention_mask: (batch, src_len) attention mask.
            labels: (batch, tgt_len) target docstring token ids.

        Returns:
            Dict with 'loss' and 'logits'.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def generate_summary(self, code: str, max_length: Optional[int] = None) -> str:
        """
        Generate a docstring/summary for a code snippet.

        Args:
            code: Source code string.
            max_length: Override max generation length.

        Returns:
            Generated documentation string.
        """
        self.eval()
        max_len = max_length or self.max_target_length

        inputs = self.tokenizer(
            code,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_len,
                num_beams=self.beam_size,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        summary = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return summary.strip()

    def get_tokenizer(self):
        """Return the tokenizer for external use (e.g., dataset creation)."""
        return self.tokenizer


def build_codebert_model(
    device: torch.device,
    decoder_layers: int = 6,
    freeze_encoder_layers: int = 0,
    max_target_length: int = 128
) -> CodeBERTSummarizer:
    """
    Factory function to build the CodeBERT Summarizer model.

    Args:
        device: torch device.
        decoder_layers: Number of decoder layers.
        freeze_encoder_layers: Encoder layers to freeze.
        max_target_length: Max generation length.

    Returns:
        Initialized CodeBERTSummarizer model.
    """
    model = CodeBERTSummarizer(
        decoder_layers=decoder_layers,
        max_target_length=max_target_length,
        freeze_encoder_layers=freeze_encoder_layers,
    )
    return model.to(device)
