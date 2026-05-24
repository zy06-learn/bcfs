"""Dataset loader for the CNN/DailyMail-only BART entrypoint."""

from __future__ import annotations

from datasets import DatasetDict, load_dataset


SUPPORTED_DATASETS = ("cnn_dailymail",)


def load_summarization_dataset(name: str) -> DatasetDict:
    if name == "cnn_dailymail":
        return load_dataset("cnn_dailymail", "3.0.0")
    raise ValueError(
        f"unknown dataset: {name!r}. supported: {SUPPORTED_DATASETS}"
    )
