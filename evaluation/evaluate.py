"""
Evaluation Module
==================
Computes BLEU, ROUGE, and Exact Match metrics for both models.
Supports side-by-side model comparison.
"""

import json
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not available.")


class Evaluator:
    """Computes automated metrics for code summarization."""

    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if NLTK_AVAILABLE else None
        self.rouge = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        ) if ROUGE_AVAILABLE else None

    def compute_bleu(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Compute BLEU-1 through BLEU-4 scores."""
        if not NLTK_AVAILABLE:
            return {"error": "NLTK not installed"}

        refs_tokenized = [[ref.split()] for ref in references]
        preds_tokenized = [pred.split() for pred in predictions]

        scores = {}
        for n in range(1, 5):
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            try:
                score = corpus_bleu(refs_tokenized, preds_tokenized,
                                     weights=weights,
                                     smoothing_function=self.smoothing)
                scores[f"BLEU-{n}"] = round(score * 100, 2)
            except Exception as e:
                scores[f"BLEU-{n}"] = 0.0
                logger.warning(f"BLEU-{n} error: {e}")

        return scores

    def compute_rouge(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
        if not ROUGE_AVAILABLE:
            return {"error": "rouge-score not installed"}

        rouge1_f, rouge2_f, rougeL_f = [], [], []

        for ref, pred in zip(references, predictions):
            scores = self.rouge.score(ref, pred)
            rouge1_f.append(scores['rouge1'].fmeasure)
            rouge2_f.append(scores['rouge2'].fmeasure)
            rougeL_f.append(scores['rougeL'].fmeasure)

        return {
            "ROUGE-1": round(sum(rouge1_f) / len(rouge1_f) * 100, 2),
            "ROUGE-2": round(sum(rouge2_f) / len(rouge2_f) * 100, 2),
            "ROUGE-L": round(sum(rougeL_f) / len(rougeL_f) * 100, 2),
        }

    def compute_exact_match(self, references: List[str], predictions: List[str]) -> float:
        """Compute Exact Match percentage."""
        matches = sum(1 for r, p in zip(references, predictions)
                      if r.strip().lower() == p.strip().lower())
        return round(matches / len(references) * 100, 2)

    def evaluate_all(self, references: List[str], predictions: List[str]) -> Dict:
        """Compute all metrics."""
        results = {}
        results.update(self.compute_bleu(references, predictions))
        results.update(self.compute_rouge(references, predictions))
        results["Exact Match"] = self.compute_exact_match(references, predictions)
        return results

    def compare_models(self, references: List[str],
                       pred_model1: List[str], pred_model2: List[str],
                       name1: str = "CodeBERT", name2: str = "Seq2Seq") -> Dict:
        """Side-by-side comparison of two models."""
        results1 = self.evaluate_all(references, pred_model1)
        results2 = self.evaluate_all(references, pred_model2)

        comparison = {"metrics": {}}
        for metric in results1:
            comparison["metrics"][metric] = {
                name1: results1[metric],
                name2: results2[metric],
                "winner": name1 if results1[metric] > results2[metric] else name2,
            }

        logger.info(f"\n{'Metric':<15} | {name1:<12} | {name2:<12} | Winner")
        logger.info("-" * 55)
        for metric, vals in comparison["metrics"].items():
            logger.info(f"{metric:<15} | {vals[name1]:<12} | {vals[name2]:<12} | {vals['winner']}")

        return comparison


def evaluate_from_files(test_data_path: str, predictions_path: str) -> Dict:
    """Evaluate predictions from saved files."""
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    references = [d["docstring"] for d in test_data]
    preds = [p["prediction"] for p in predictions]

    evaluator = Evaluator()
    return evaluator.evaluate_all(references, preds)


if __name__ == "__main__":
    # Demo with sample data
    refs = [
        "calculate the average of a list of numbers",
        "return the maximum element in the array",
        "sort the list in ascending order",
    ]
    preds = [
        "compute the mean of a list of values",
        "find the maximum element in the array",
        "sort the list in ascending order",
    ]

    evaluator = Evaluator()
    results = evaluator.evaluate_all(refs, preds)
    print("\nEvaluation Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score}")
