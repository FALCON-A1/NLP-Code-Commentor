"""
Inference Pipeline
==================
Loads trained models and generates documentation for input code.
Supports both Seq2Seq and CodeBERT models.
"""

import os
import sys
import torch
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeDocGenerator:
    """
    Unified inference interface for generating code documentation.
    Supports loading either the Seq2Seq or CodeBERT model.
    """

    def __init__(self, model_type: str = "codebert", checkpoint_path: Optional[str] = None):
        """
        Args:
            model_type: 'codebert' or 'seq2seq'.
            checkpoint_path: Path to saved model checkpoint.
        """
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if checkpoint_path:
            self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: str):
        """Load a trained model from checkpoint."""
        logger.info(f"Loading {self.model_type} model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.model_type == "codebert":
            from models.codebert_summarizer import build_codebert_model
            self.model = build_codebert_model(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        elif self.model_type == "seq2seq":
            from models.seq2seq import build_seq2seq_model
            self.src_vocab = checkpoint['src_vocab']
            self.tgt_vocab = checkpoint['tgt_vocab']
            args = checkpoint['args']
            self.model = build_seq2seq_model(
                src_vocab_size=len(self.src_vocab),
                tgt_vocab_size=len(self.tgt_vocab),
                device=self.device,
                embed_dim=args.get('embed_dim', 300),
                hidden_dim=args.get('hidden_dim', 512),
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        logger.info("Model loaded successfully.")

    def generate(self, code: str) -> str:
        """
        Generate documentation for a code snippet.

        Args:
            code: Python source code string.

        Returns:
            Generated documentation string.
        """
        if self.model is None:
            return "Error: No model loaded. Call load_model() first."

        if self.model_type == "codebert":
            return self.model.generate_summary(code)

        elif self.model_type == "seq2seq":
            from src.code_parser import CodeTokenizer
            from src.preprocessing import Vocabulary

            tokenizer = CodeTokenizer()
            tokens = tokenizer.tokenize_code(code)[:256]
            indices = self.src_vocab.encode(tokens)
            src_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)

            gen_tokens, _ = self.model.generate(
                src_tensor,
                sos_idx=Vocabulary.SOS_IDX,
                eos_idx=Vocabulary.EOS_IDX,
            )
            words = self.tgt_vocab.decode(gen_tokens)
            # Remove special tokens
            words = [w for w in words if w not in
                     {Vocabulary.PAD_TOKEN, Vocabulary.SOS_TOKEN,
                      Vocabulary.EOS_TOKEN, Vocabulary.UNK_TOKEN}]
            return " ".join(words)

        return "Error: Unknown model type."


# Singleton for the web app
_generator = None

def get_generator(model_type: str = "codebert",
                  checkpoint: Optional[str] = None) -> CodeDocGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = CodeDocGenerator(model_type, checkpoint)
    return _generator
