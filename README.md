# CodeCommentator NLP: Automated Code Documentation Generator

## Overview
This project analyzes source code and automatically generates meaningful comments and documentation using NLP-based techniques. It helps developers understand legacy code and improves maintainability.

## Deep Learning Models

### Model 1: CodeBERT (Fine-tuned Transformer)
- **Architecture**: Encoder-Decoder with CodeBERT (pre-trained on code + NL)
- **Framework**: PyTorch + HuggingFace Transformers
- **Role**: Primary high-performance model

### Model 2: Seq2Seq + Bahdanau Attention (LSTM)
- **Architecture**: Bidirectional LSTM Encoder + LSTM Decoder with Attention
- **Framework**: PyTorch
- **Role**: Baseline comparison + lightweight alternative

## Dataset
- **CodeSearchNet** (Python subset): ~500K function-docstring pairs

## Evaluation Metrics
- BLEU Score (1-4)
- ROUGE Score (1, 2, L)
- Exact Match
- Human Evaluation

## Project Structure
```
NLP/
├── data/               # Raw and processed datasets
├── models/             # Model architectures (Seq2Seq, CodeBERT)
├── src/                # Data collection, parsing, preprocessing
├── training/           # Training scripts for both models
├── evaluation/         # Metrics computation
├── deployment/         # FastAPI web app + inference
├── checkpoints/        # Saved model weights
└── requirements.txt    # Dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Prepare Data
```bash
python src/data_collection.py
```

### 3. Train Models
```bash
# Seq2Seq model
python training/train_seq2seq.py --epochs 30 --batch_size 64

# CodeBERT model (requires GPU)
python training/train_codebert.py --epochs 10 --batch_size 16
```

### 4. Evaluate
```bash
python evaluation/evaluate.py
```

### 5. Deploy Web App
```bash
cd deployment && uvicorn app:app --reload --port 8000
```

## Tools & Libraries
- Python AST, SpaCy, PyTorch, HuggingFace Transformers
- Gensim, NLTK, rouge-score, FastAPI
