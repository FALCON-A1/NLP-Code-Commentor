"""
Data Collection Module
======================
Handles downloading and loading the CodeSearchNet dataset
and optionally scraping GitHub repositories.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning("HuggingFace datasets not installed.")


class CodeSearchNetLoader:
    """
    Loads the CodeSearchNet dataset from HuggingFace.
    Provides ~500K Python function-docstring pairs.
    """

    def __init__(self, data_dir: str = "data/raw", language: str = "python"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.language = language

    def load(self, split: str = "train", max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load CodeSearchNet dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test').
            max_samples: Optional limit on number of samples.

        Returns:
            List of dicts with 'code' and 'docstring' keys.
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("Install 'datasets': pip install datasets")

        logger.info(f"Loading CodeSearchNet ({self.language}) - {split} split...")
        dataset = load_dataset("code_search_net", self.language, split=split)

        pairs = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            pairs.append({
                "code": item.get("func_code_string", ""),
                "docstring": item.get("func_documentation_string", ""),
                "language": self.language,
                "repo": item.get("repository_name", ""),
                "path": item.get("func_path_in_repo", ""),
                "func_name": item.get("func_name", ""),
            })

        logger.info(f"Loaded {len(pairs)} pairs from {split} split.")
        return pairs

    def load_all_splits(self, max_samples_per_split: Optional[int] = None) -> Dict[str, List[Dict]]:
        """Load all dataset splits."""
        splits = {}
        for split_name in ["train", "validation", "test"]:
            try:
                splits[split_name] = self.load(split_name, max_samples_per_split)
            except Exception as e:
                logger.error(f"Error loading {split_name}: {e}")
                splits[split_name] = []
        return splits

    def save_processed(self, pairs: List[Dict], filename: str) -> str:
        """Save processed pairs to JSON."""
        output_path = self.data_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(pairs)} pairs to {output_path}")
        return str(output_path)


class GitHubScraper:
    """
    Scrapes Python repositories from GitHub for additional training data.
    Requires a GitHub API token for higher rate limits.
    """

    GITHUB_API = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.headers = {"Authorization": f"token {self.token}"} if self.token else {}

    def search_repos(self, query: str = "language:python stars:>100",
                     max_repos: int = 10) -> List[Dict]:
        """Search GitHub for Python repositories."""
        try:
            import requests
        except ImportError:
            logger.error("Install requests: pip install requests")
            return []

        url = f"{self.GITHUB_API}/search/repositories"
        params = {"q": query, "sort": "stars", "per_page": min(max_repos, 100)}
        resp = requests.get(url, headers=self.headers, params=params)

        if resp.status_code != 200:
            logger.error(f"GitHub API error: {resp.status_code}")
            return []

        return [
            {"name": r["full_name"], "url": r["html_url"], "stars": r["stargazers_count"]}
            for r in resp.json().get("items", [])[:max_repos]
        ]


def prepare_dataset(
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    save_dir: str = "data/processed"
) -> Dict[str, str]:
    """
    Main function to download, process, and save the dataset.

    Returns:
        Dict mapping split names to saved file paths.
    """
    from src.preprocessing import TextPreprocessor, CodePreprocessor

    loader = CodeSearchNetLoader()
    code_prep = CodePreprocessor()
    text_prep = TextPreprocessor()

    saved_paths = {}
    limits = {"train": max_train, "validation": max_val, "test": max_test}

    for split_name, max_samples in limits.items():
        raw_pairs = loader.load(split_name, max_samples)

        # Preprocess
        processed = []
        for pair in raw_pairs:
            cleaned_code = code_prep.clean_code(pair["code"])
            cleaned_doc = text_prep.clean_docstring(pair["docstring"])
            if text_prep.is_valid_pair(cleaned_code, cleaned_doc):
                processed.append({
                    "code": cleaned_code,
                    "docstring": cleaned_doc,
                    "func_name": pair.get("func_name", ""),
                })

        logger.info(f"{split_name}: {len(processed)}/{len(raw_pairs)} pairs kept after filtering.")

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{split_name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        saved_paths[split_name] = path

    return saved_paths


if __name__ == "__main__":
    # Quick test with small subset
    paths = prepare_dataset(max_train=1000, max_val=200, max_test=200)
    print("Saved datasets:", paths)
