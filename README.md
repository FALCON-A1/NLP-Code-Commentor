<div align="center">
  <h1>🤖 CodeCommentator NLP</h1>
  <p><strong>Automated Code Documentation Generator using Deep Learning</strong></p>
</div>

## 📖 Overview
CodeCommentator NLP is an advanced machine learning project designed to analyze Python source code and automatically generate meaningful, human-readable docstrings and comments. By leveraging Natural Language Processing (NLP), this tool assists developers in understanding legacy code, accelerating documentation processes, and significantly improving overall code maintainability.

## ✨ Key Features
- **Dual Model Support:** Choose between a state-of-the-art pre-trained Transformer (CodeBERT) or a custom Bidirectional LSTM with Bahdanau Attention.
- **Automated Data Pipeline:** Built-in tools for downloading, parsing, and cleaning data from the massive HuggingFace CodeSearchNet dataset.
- **Kaggle-Optimized Training:** Highly optimized training scripts featuring Mixed Precision (AMP), multi-GPU DataParallel, and gradient accumulation.
- **FastAPI Web Interface:** A sleek, interactive web application to input code and instantly receive generated documentation.
- **Comprehensive Evaluation:** Built-in evaluation pipeline calculating BLEU, ROUGE, and Exact Match metrics to accurately compare model performances.

## 🧠 Architecture & Models

### 1. CodeBERT Summarizer (Primary Model)
- **Architecture:** Encoder-Decoder using Microsoft's `codebert-base` as the encoder and a custom cross-attention Roberta decoder.
- **Strengths:** High performance, contextual understanding of both natural language and code semantics.
- **Framework:** PyTorch + HuggingFace Transformers.

### 2. Seq2Seq + Bahdanau Attention (Baseline Model)
- **Architecture:** Bidirectional LSTM Encoder + LSTM Decoder enhanced with Additive (Bahdanau) Attention.
- **Strengths:** Lightweight, fast inference, and serves as an excellent baseline to measure the performance gains of the transformer model.
- **Framework:** PyTorch natively.

## 📂 Project Structure
```text
NLP-Code-Commentor-main/
├── data/               # Raw and processed datasets (CodeSearchNet)
├── models/             # PyTorch Model architectures (CodeBERT, Seq2Seq)
├── src/                # Data collection, AST code parsing, text preprocessing
├── training/           # Kaggle-optimized training scripts
├── evaluation/         # Automated metrics (BLEU, ROUGE)
├── deployment/         # FastAPI web app and inference pipeline
├── checkpoints/        # Saved model weights
└── requirements.txt    # Project dependencies
```

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Data Preparation
Download and preprocess the CodeSearchNet dataset.
```bash
python src/data_collection.py
```

### 3. Training
Train the models using our optimized scripts. The CodeBERT model is highly recommended to be trained on a GPU.

**Train CodeBERT (Requires GPU):**
```bash
python training/train_codebert.py --epochs 10 --batch_size 16
```

**Train Seq2Seq Baseline:**
```bash
python training/train_seq2seq.py --epochs 30 --batch_size 64
```

### 4. Evaluation
Evaluate and compare the models against the test dataset to compute BLEU and ROUGE scores.
```bash
python evaluation/evaluate.py
```

### 5. Deployment
Launch the FastAPI web interface to interactively generate comments for your code snippets.
```bash
cd deployment
uvicorn app:app --reload --port 8000
```
Visit `http://localhost:8000` in your browser to access the web UI.

## 📊 Evaluation Metrics
To ensure the highest quality of generated comments, this project evaluates output using:
- **BLEU (1-4):** Measures precision of n-gram overlap between generated and reference docstrings.
- **ROUGE (1, 2, L):** Measures recall and longest common sub-sequence.
- **Exact Match:** Percentage of perfectly generated docstrings.

## 🛠️ Technologies Used
- **Deep Learning:** PyTorch, HuggingFace Transformers (`codebert-base`)
- **NLP Processing:** SpaCy, Gensim (FastText), NLTK, rouge-score
- **Code Parsing:** Python `ast` module
- **Web App:** FastAPI, Uvicorn, Jinja2
- **Data Handling:** HuggingFace Datasets, Pandas, NumPy

## 📝 License
This project is open-source and available under the MIT License.
