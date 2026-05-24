"""Dataset loader for the Multi-News-only PRIMERA entrypoint."""

from __future__ import annotations

from datasets import DatasetDict, load_dataset


SUPPORTED_DATASETS = ("multi_news",)


def load_summarization_dataset(name: str) -> DatasetDict:
    if name != "multi_news":
        raise ValueError(f"unknown dataset: {name!r}. supported: {SUPPORTED_DATASETS}")

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
