"""
Preprocessing Module
====================
Handles text cleaning, normalization, and tokenization for both
code and natural language (docstrings/comments).
"""

import re
import logging
from typing import List, Tuple, Optional
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available. Falling back to basic tokenization.")


class TextPreprocessor:
    """Preprocesses natural language text (docstrings, comments)."""

    def __init__(self, max_length: int = 128, min_length: int = 3):
        self.max_length = max_length
        self.min_length = min_length

    def clean_docstring(self, docstring: str) -> str:
        """Clean and normalize a docstring."""
        if not docstring:
            return ""
        text = docstring.strip().strip('"""').strip("'''")
        # Remove parameter sections
        text = re.split(
            r'\n\s*(Args|Parameters|Returns|Raises|Yields|Examples?|Notes?):',
            text, flags=re.IGNORECASE
        )[0]
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'https?://\S+', '', text)
        return text.strip()

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize natural language text."""
        if SPACY_AVAILABLE:
            doc = nlp(text)
            return [token.text.lower() for token in doc if not token.is_space]
        return text.lower().split()

    def is_valid_pair(self, code: str, docstring: str) -> bool:
        """Check if a code-docstring pair meets quality criteria."""
        if not code or not docstring:
            return False
        doc_tokens = docstring.split()
        code_tokens = code.split()
        if len(doc_tokens) < self.min_length or len(doc_tokens) > self.max_length:
            return False
        if len(code_tokens) < self.min_length:
            return False
        boilerplate = [r'^todo', r'^fixme', r'^auto-generated', r'^getter for', r'^setter for']
        cleaned = docstring.strip().lower()
        for pattern in boilerplate:
            if re.match(pattern, cleaned):
                return False
        return True


class CodePreprocessor:
    """Preprocesses source code for model input."""

    def __init__(self, max_code_length: int = 256):
        self.max_code_length = max_code_length

    def clean_code(self, code: str) -> str:
        """Clean and normalize source code."""
        if not code:
            return ""
        code = code.replace('\t', '    ')
        code = re.sub(r'\n{3,}', '\n\n', code)
        code = '\n'.join(line.rstrip() for line in code.split('\n'))
        return code.strip()


class Vocabulary:
    """Builds and manages vocabulary for the Seq2Seq model."""

    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

    def __init__(self, min_freq: int = 2, max_size: Optional[int] = 50000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {self.PAD_TOKEN: 0, self.SOS_TOKEN: 1, self.EOS_TOKEN: 2, self.UNK_TOKEN: 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()

    def build_vocab(self, token_lists: List[List[str]]) -> None:
        """Build vocabulary from token sequences."""
        for tokens in token_lists:
            self.word_freq.update(tokens)
        valid = sorted(
            [(w, f) for w, f in self.word_freq.items() if f >= self.min_freq],
            key=lambda x: x[1], reverse=True
        )
        if self.max_size:
            valid = valid[:self.max_size - len(self.word2idx)]
        for word, _ in valid:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        logger.info(f"Vocabulary: {len(self.word2idx)} tokens")

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.word2idx.get(t, self.UNK_IDX) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(i, self.UNK_TOKEN) for i in indices]

    def __len__(self) -> int:
        return len(self.word2idx)


def preprocess_pipeline(code: str, docstring: str) -> Optional[Tuple[str, str]]:
    """Full preprocessing pipeline for a single code-docstring pair."""
    cp = CodePreprocessor()
    tp = TextPreprocessor()
    cleaned_code = cp.clean_code(code)
    cleaned_doc = tp.clean_docstring(docstring)
    if tp.is_valid_pair(cleaned_code, cleaned_doc):
        return cleaned_code, cleaned_doc
    return None
