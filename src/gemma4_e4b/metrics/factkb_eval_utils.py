"""Helpers for running FACTKB factuality scoring.

FACTKB (<external-repository-url>) is a binary sequence classifier
scoring ``(summary, article)`` pairs. Class index 1 is the factual class, so
we report the factual probability in [0, 1].

Per the official README, the tokenizer is loaded from ``roberta-base`` (not
from the FactKB checkpoint), and inputs are fed as ``[[summary, article]]``
pairs with ``padding="max_length"`` + ``truncation=True``. This is a pure
HuggingFace model, no subprocess.
"""

from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FACTKB_MODEL_NAME = "bunsenfeng/FactKB"
FACTKB_TOKENIZER_NAME = "roberta-base"


def _factual_label_index(model) -> int:
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, name in id2label.items():
        if isinstance(name, str) and name.strip().lower() in {
            "factual", "1", "correct", "consistent", "true", "label_1"
        }:
            try:
                return int(idx)
            except (TypeError, ValueError):
                continue
    num_labels = int(getattr(model.config, "num_labels", 2))
    return 1 if num_labels >= 2 else 0


def load_factkb_model(
    device,
    model_name: str = FACTKB_MODEL_NAME,
    tokenizer_name: str = FACTKB_TOKENIZER_NAME,
):
    """Load FACTKB. Returns (tokenizer, model, factual_label_index)."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)
    model.eval()
    return tokenizer, model, _factual_label_index(model)


def compute_factkb_summary_scores(
    generated_summaries: Iterable[str],
    articles: Iterable[str],
    tokenizer,
    model,
    device,
    factual_idx: int,
    batch_size: int = 8,
    max_length: int = 512,
) -> list[float]:
    """Return one factual probability per (summary, article) pair.

    Empty summary or empty article → score 0.0 for that sample (no crash).
    """
    generated_summaries = list(generated_summaries)
    articles = list(articles)
    if len(generated_summaries) != len(articles):
        raise ValueError(
            f"generated_summaries ({len(generated_summaries)}) and articles ({len(articles)}) "
            "must be the same length."
        )

    scores: list[float] = [0.0] * len(generated_summaries)
    valid_indices: list[int] = []
    valid_pairs: list[list[str]] = []
    for i, (summary, article) in enumerate(zip(generated_summaries, articles)):
        summary_str, article_str = str(summary), str(article)
        if not summary_str.strip() or not article_str.strip():
            continue
        valid_indices.append(i)
        valid_pairs.append([summary_str, article_str])

    for start in range(0, len(valid_indices), batch_size):
        batch_pairs = valid_pairs[start:start + batch_size]
        batch_map = valid_indices[start:start + batch_size]
        inputs = tokenizer(
            batch_pairs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, factual_idx]
        for local_i, prob in enumerate(probs.detach().cpu().tolist()):
            scores[batch_map[local_i]] = float(prob)

    return scores
