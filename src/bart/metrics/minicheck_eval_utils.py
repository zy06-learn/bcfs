"""Helpers for loading and scoring MiniCheck."""

from __future__ import annotations

import os

from assets.loader import ensure_asset_repo_on_sys_path, get_asset_dir


ensure_asset_repo_on_sys_path("MiniCheck-main")

from minicheck import inference as minicheck_inference
from minicheck.minicheck import MiniCheck


MINICHECK_MODEL_NAME = "roberta-large"
MINICHECK_CACHE_DIR = str(get_asset_dir("minicheck_ckpts"))


def _disable_minicheck_progress_bars() -> None:
    def _passthrough(iterable, *args, **kwargs):
        return iterable

    minicheck_inference.tqdm = _passthrough


def _should_disable_minicheck_progress_bars() -> bool:
    raw = os.environ.get("NLM_ENABLE_MINICHECK_PROGRESS", "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def load_minicheck_model(
    device=None,
    model_name: str = MINICHECK_MODEL_NAME,
    batch_size: int = 16,
    cache_dir: str = MINICHECK_CACHE_DIR,
):
    _ = device
    if _should_disable_minicheck_progress_bars():
        _disable_minicheck_progress_bars()
    return MiniCheck(model_name=model_name, batch_size=batch_size, cache_dir=cache_dir)


def compute_minicheck_sentence_scores(article, summary, scorer, segmenter):
    sentences = [sent.strip() for sent in segmenter.segment(summary) if sent.strip()]
    if not sentences and summary.strip():
        sentences = [summary.strip()]
    if not sentences:
        return []

    try:
        _, raw_prob, _, _ = scorer.score(
            docs=[article] * len(sentences),
            claims=sentences,
        )
        return [float(prob) for prob in raw_prob]
    except Exception:
        # Batch failed (e.g. empty tensor after tokenisation). Fall back to
        # scoring one sentence at a time; skip any that still crash.
        results = []
        for sent in sentences:
            try:
                _, raw_prob, _, _ = scorer.score(docs=[article], claims=[sent])
                results.append(float(raw_prob[0]))
            except Exception:
                pass
        return results


def compute_minicheck_summary_scores(generated_summaries, articles, scorer, segmenter):
    summary_scores = []
    total = len(generated_summaries)
    for index, (summary, article) in enumerate(zip(generated_summaries, articles), start=1):
        sentence_scores = compute_minicheck_sentence_scores(article, summary, scorer, segmenter)
        if sentence_scores:
            summary_scores.append(sum(sentence_scores) / len(sentence_scores))
        else:
            summary_scores.append(0.0)
        if index == 1 or index % 50 == 0 or index == total:
            print(f"  [MiniCheck] scored {index}/{total} summaries", flush=True)
    return summary_scores
