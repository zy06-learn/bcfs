"""Helpers for loading and scoring AlignScore."""

from __future__ import annotations

import os
import sys
from typing import Iterable, List

import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from assets.loader import ensure_path_on_sys_path, require_asset_dir


ALIGNSCORE_MODEL_NAME = "roberta-base"
ALIGNSCORE_EVAL_MODE = "nli_sp"
ALIGNSCORE_SRC = str(require_asset_dir("AlignScore-main") / "src")
ALIGNSCORE_CKPT = str(require_asset_dir("alignscore_ckpt") / "AlignScore-base.ckpt")


def _device_to_alignscore(device) -> str:
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return f"cuda:{device.index}" if device.index is not None else "cuda:0"
        return "cpu"

    device_str = str(device)
    if device_str.startswith("cuda"):
        return device_str if ":" in device_str else "cuda:0"
    return "cpu"


def _patch_alignscore_compat() -> None:
    if getattr(_patch_alignscore_compat, "_done", False):
        return

    # `evaluate.load(...)` can replace `sys.modules["transformers"]` with a fresh
    # lazy module object. Patch the live module instance rather than the stale
    # module object captured at import time so AlignScore sees AdamW reliably.
    live_transformers = sys.modules.get("transformers", transformers)
    live_transformers.AdamW = torch.optim.AdamW
    live_transformers.__dict__["AdamW"] = torch.optim.AdamW
    ensure_path_on_sys_path(ALIGNSCORE_SRC)

    from alignscore.model import BERTAlignModel
    import alignscore.inference as inference_mod

    def patched_init(self, ckpt_path, model="bert-base-uncased", batch_size=32, device="cuda", verbose=True):
        self.device = device
        if ckpt_path is not None:
            self.model = BERTAlignModel.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                strict=False,
                model=model,
            ).to(self.device)
        else:
            self.model = BERTAlignModel(model=model).to(self.device)

        self.model.eval()
        self.batch_size = batch_size
        self.config = AutoConfig.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        import spacy

        self.spacy = spacy.load("en_core_web_sm")
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.softmax = nn.Softmax(dim=-1)
        self.smart_type = "smart-n"
        self.smart_n_metric = "f1"
        self.disable_progress_bar_in_inference = False
        self.nlg_eval_mode = None
        self.verbose = verbose

    inference_mod.Inferencer.__init__ = patched_init
    _patch_alignscore_compat._done = True


def load_alignscore_model(
    device,
    model_name: str = ALIGNSCORE_MODEL_NAME,
    batch_size: int = 8,
    ckpt_path: str = ALIGNSCORE_CKPT,
    evaluation_mode: str = ALIGNSCORE_EVAL_MODE,
    verbose: bool = False,
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"AlignScore checkpoint not found: {ckpt_path}")

    _patch_alignscore_compat()

    from alignscore import AlignScore

    return AlignScore(
        model=model_name,
        batch_size=batch_size,
        device=_device_to_alignscore(device),
        ckpt_path=ckpt_path,
        evaluation_mode=evaluation_mode,
        verbose=verbose,
    )


def compute_alignscore_summary_scores(
    generated_summaries: Iterable[str],
    articles: Iterable[str],
    scorer,
    batch_size: int = 8,
) -> List[float]:
    """Compute per-summary AlignScore, robust to per-batch failures.

    Rationale: in ``nli_sp`` mode AlignScore splits the *claim* (summary) into
    sentences and then the *context* (article) into chunks. When a whole batch
    has its chunks truncated away by the tokenizer budget, AlignScore falls
    into a ``torch.cat(): expected a non-empty list of Tensors`` crash, which
    aborts the entire metric. We isolate failures to the smallest unit so a
    handful of pathological samples cannot wipe out an 11k-sample run.

    Degenerate samples (empty, whitespace, or too short) are scored as 0.0 so
    the mean stays comparable when the upstream pipeline did not pre-filter.
    """
    generated_summaries = list(generated_summaries)
    articles = list(articles)
    scores: List[float] = []
    failures: list[tuple[int, str]] = []

    def _score_batch(start: int, size: int) -> list[float] | None:
        end = min(start + size, len(generated_summaries))
        batch_s = generated_summaries[start:end]
        batch_a = articles[start:end]
        try:
            return [float(x) for x in scorer.score(contexts=batch_a, claims=batch_s)]
        except Exception:
            return None

    i = 0
    while i < len(generated_summaries):
        batch_scores = _score_batch(i, batch_size)
        if batch_scores is not None:
            scores.extend(batch_scores)
            i += batch_size
            continue
        # Batch failed: retry one-by-one to salvage the good ones.
        for j in range(i, min(i + batch_size, len(generated_summaries))):
            single = _score_batch(j, 1)
            if single is None:
                # Hard failure on this single sample: score as 0.0 and record.
                failures.append((j, (generated_summaries[j] or "")[:80]))
                scores.append(0.0)
            else:
                scores.extend(single)
        i += batch_size

    if failures:
        import sys
        print(
            f"  [alignscore] {len(failures)} samples failed individually "
            f"(scored as 0.0); first few indices: "
            f"{[idx for idx, _ in failures[:5]]}",
            file=sys.stderr,
        )
    return scores
