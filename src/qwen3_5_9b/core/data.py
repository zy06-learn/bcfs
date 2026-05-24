"""Unified summarization dataset loader for cnn_dailymail and multi_news.

Returns a ``DatasetDict`` whose splits expose the canonical fields used by
the existing pipeline (`article`, `highlights`, `id`). multi_news's
`document` and `summary` are renamed in-place so the orchestration layer
stays unchanged.
"""

from __future__ import annotations

from datasets import DatasetDict, load_dataset


SUPPORTED_DATASETS = ("cnn_dailymail", "multi_news")


def _load_multi_news() -> DatasetDict:
    ds = load_dataset("alexfabbri/multi_news")
    result = DatasetDict()
    for split_name, split_ds in ds.items():
        result[split_name] = split_ds.rename_columns(
            {"document": "article", "summary": "highlights"}
        ).map(
            lambda examples, indices: {"id": [f"multi_news_{split_name}_{idx}" for idx in indices]},
            with_indices=True,
            batched=True,
        )
    return result


def load_summarization_dataset(name: str) -> DatasetDict:
    if name == "cnn_dailymail":
        return load_dataset("cnn_dailymail", "3.0.0")
    if name == "multi_news":
        return _load_multi_news()
    raise ValueError(
        f"unknown dataset: {name!r}. supported: {SUPPORTED_DATASETS}"
    )
