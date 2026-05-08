"""
Embeddings Module
==================
Generates and manages code embeddings using FastText (for Seq2Seq)
and CodeBERT (for the Transformer model).
"""

import os
import json
import logging
import numpy as np
from typing import List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from gensim.models import FastText as FastTextModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


class FastTextEmbeddings:
    """
    Trains FastText embeddings on the code corpus
    for use with the Seq2Seq model.
    """

    def __init__(self, embed_dim: int = 300, min_count: int = 2,
                 window: int = 5, epochs: int = 10):
        self.embed_dim = embed_dim
        self.min_count = min_count
        self.window = window
        self.epochs = epochs
        self.model = None

    def train(self, token_lists: List[List[str]]) -> None:
        """Train FastText on tokenized code corpus."""
        if not GENSIM_AVAILABLE:
            raise ImportError("Install gensim: pip install gensim")

        logger.info(f"Training FastText ({self.embed_dim}d) on {len(token_lists)} sequences...")
        self.model = FastTextModel(
            sentences=token_lists,
            vector_size=self.embed_dim,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=4,
            sg=1,  # Skip-gram
        )
        logger.info("FastText training complete.")

    def get_embedding(self, token: str) -> np.ndarray:
        """Get embedding vector for a token."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.wv[token]

    def get_embedding_matrix(self, vocab) -> np.ndarray:
        """
        Build an embedding matrix aligned with the vocabulary.

        Args:
            vocab: Vocabulary object with word2idx mapping.

        Returns:
            (vocab_size, embed_dim) numpy array.
        """
        matrix = np.random.normal(0, 0.1, (len(vocab), self.embed_dim))
        found = 0
        for word, idx in vocab.word2idx.items():
            try:
                matrix[idx] = self.model.wv[word]
                found += 1
            except KeyError:
                pass
        logger.info(f"Embedded {found}/{len(vocab)} tokens from FastText.")
        return matrix

    def save(self, path: str):
        """Save the trained model."""
        if self.model:
            self.model.save(path)
            logger.info(f"FastText model saved to {path}")

    def load(self, path: str):
        """Load a trained model."""
        if not GENSIM_AVAILABLE:
            raise ImportError("Install gensim: pip install gensim")
        self.model = FastTextModel.load(path)
        logger.info(f"FastText model loaded from {path}")


def build_fasttext_embeddings(
    train_data_path: str,
    save_path: str = "data/embeddings/fasttext_code.model",
    embed_dim: int = 300
) -> FastTextEmbeddings:
    """
    Build FastText embeddings from training data.

    Args:
        train_data_path: Path to processed training JSON.
        save_path: Where to save the trained FastText model.
        embed_dim: Embedding dimension.

    Returns:
        Trained FastTextEmbeddings instance.
    """
    from src.code_parser import CodeTokenizer

    with open(train_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = CodeTokenizer()
    code_tokens = [tokenizer.tokenize_code(d["code"]) for d in data]
    doc_tokens = [d["docstring"].lower().split() for d in data]
    all_tokens = code_tokens + doc_tokens

    ft = FastTextEmbeddings(embed_dim=embed_dim)
    ft.train(all_tokens)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ft.save(save_path)

    return ft


if __name__ == "__main__":
    print("Embeddings module loaded.")
    print(f"Gensim available: {GENSIM_AVAILABLE}")
