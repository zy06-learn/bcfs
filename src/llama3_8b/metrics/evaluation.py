"""Shared evaluation pipeline for summarization experiments."""

from __future__ import annotations

import json
import os
from time import perf_counter

import torch
from rouge_score import rouge_scorer

from assets.loader import ensure_asset_repo_on_sys_path

ensure_asset_repo_on_sys_path("bert_score-master")

from bert_score import score as bert_score_fn

from metrics.alignscore_eval_utils import compute_alignscore_summary_scores, load_alignscore_model
from metrics.factcc_eval_utils import compute_factcc_summary_scores, load_factcc_eval_model
from metrics.factkb_eval_utils import compute_factkb_summary_scores, load_factkb_model
from metrics.minicheck_eval_utils import compute_minicheck_summary_scores, load_minicheck_model

try:
    import nltk
except ImportError:
    nltk = None


SUPPORTED_METRIC_NAMES = {
    "rouge",
    "bertscore",
    "factcc",
    "minicheck",
    "alignscore",
    "factkb",
}

DEFAULT_table_metric_names = ["rouge", "bertscore"]
DEFAULT_EXTRA_METRIC_NAMES = ["factcc", "minicheck", "alignscore", "factkb"]


def split_sentences_for_rouge(text, segmenter, sentence_split_for_rouge="nltk"):
    """Sentence splitting for ROUGE formatting.

    - `pysbd`: project-main experiment protocol
    - `nltk`: HF baseline-aligned protocol
    """
    if sentence_split_for_rouge == "nltk" and nltk is not None:
        try:
            sentences = nltk.sent_tokenize(text.strip())
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                sentences = nltk.sent_tokenize(text.strip())
            except Exception:
                sentences = [segment.strip() for segment in segmenter.segment(text) if segment.strip()]
    else:
        sentences = [segment.strip() for segment in segmenter.segment(text) if segment.strip()]
    return "\n".join(sentences)


def normalize_metric_names(metric_names):
    names = list(metric_names or [])
    invalid = sorted(set(names) - SUPPORTED_METRIC_NAMES)
    if invalid:
        raise ValueError(f"Unsupported metric names: {invalid}. Supported names: {sorted(SUPPORTED_METRIC_NAMES)}")
    return names


def _is_multi_reference_batch(references):
    return bool(references) and isinstance(references[0], (list, tuple))


def _normalize_reference_entry(reference_entry):
    if isinstance(reference_entry, (list, tuple)):
        return [str(reference).strip() for reference in reference_entry if str(reference).strip()]

    reference_text = str(reference_entry).strip()
    return [reference_text] if reference_text else []



def _resolve_metric_batch_sizes(eval_batch_size: int | None):
    if eval_batch_size is None or eval_batch_size < 1:
        eval_batch_size = 16

    return {
        "bertscore": eval_batch_size,
        "factcc": eval_batch_size,
        "minicheck": eval_batch_size,
        "alignscore": eval_batch_size,
        "factkb": eval_batch_size,
    }


def _run_rouge(
    generated_summaries,
    references,
    segmenter,
    num_samples,
    rouge_impl,
    use_table_sentence_split,
    sentence_split_for_rouge="nltk",
):
    metrics = {}
    multi_reference = _is_multi_reference_batch(references)
    effective_rouge_impl = rouge_impl
    if multi_reference and rouge_impl == "hf":
        effective_rouge_impl = "local"

    if multi_reference:
        metrics["rouge_impl"] = "rouge_score_multi"
    else:
        metrics["rouge_impl"] = "huggingface_evaluate" if effective_rouge_impl == "hf" else "rouge_score"
    metrics["sentence_split_for_rouge"] = sentence_split_for_rouge

    print(f"\n{'=' * 60}")
    print(f"ROUGE Evaluation ({num_samples} samples, impl={metrics['rouge_impl']})")
    print(f"{'=' * 60}")
    if multi_reference and rouge_impl == "hf":
        print("  Multi-reference ROUGE detected; using rouge_score score_multi instead of huggingface_evaluate.")

    if effective_rouge_impl == "hf":
        try:
            import evaluate
        except ImportError:
            print("  HuggingFace evaluate is unavailable in this environment; falling back to local rouge_score.")
            effective_rouge_impl = "local"
            metrics["rouge_impl"] = "rouge_score"

    if effective_rouge_impl == "hf":

        if use_table_sentence_split:
            predictions_for_rouge = [
                split_sentences_for_rouge(
                    generated,
                    segmenter,
                    sentence_split_for_rouge=sentence_split_for_rouge,
                )
                for generated in generated_summaries
            ]
            references_for_rouge = [
                split_sentences_for_rouge(
                    reference,
                    segmenter,
                    sentence_split_for_rouge=sentence_split_for_rouge,
                )
                for reference in references
            ]
        else:
            predictions_for_rouge = [
                split_sentences_for_rouge(
                    generated,
                    segmenter,
                    sentence_split_for_rouge=sentence_split_for_rouge,
                )
                for generated in generated_summaries
            ]
            references_for_rouge = [
                split_sentences_for_rouge(
                    reference,
                    segmenter,
                    sentence_split_for_rouge=sentence_split_for_rouge,
                )
                for reference in references
            ]

        rouge_scores = evaluate.load("rouge").compute(
            predictions=predictions_for_rouge,
            references=references_for_rouge,
            use_stemmer=True,
        )
        metrics["rouge_scores"] = {
            key: float(rouge_scores[key]) * 100 for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        }
        metrics["rouge_totals"] = None

        for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            print(f"  {key:8s}  F1={metrics['rouge_scores'][key]:.2f}%")
    else:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
        totals = {
            "rouge1": [0, 0, 0],
            "rouge2": [0, 0, 0],
            "rougeL": [0, 0, 0],
            "rougeLsum": [0, 0, 0],
        }

        for generated, reference in zip(generated_summaries, references):
            if use_table_sentence_split:
                generated_lsum = split_sentences_for_rouge(
                    generated,
                    segmenter,
                    sentence_split_for_rouge=sentence_split_for_rouge,
                )
                if multi_reference:
                    reference_lsum = [
                        split_sentences_for_rouge(
                            reference_text,
                            segmenter,
                            sentence_split_for_rouge=sentence_split_for_rouge,
                        )
                        for reference_text in _normalize_reference_entry(reference)
                    ]
                else:
                    reference_lsum = split_sentences_for_rouge(
                        reference,
                        segmenter,
                        sentence_split_for_rouge=sentence_split_for_rouge,
                    )
            else:
                generated_lsum = split_sentences_for_rouge(
                    generated,
                    segmenter,
                    sentence_split_for_rouge=sentence_split_for_rouge,
                )
                if multi_reference:
                    reference_lsum = [
                        split_sentences_for_rouge(
                            reference_text,
                            segmenter,
                            sentence_split_for_rouge=sentence_split_for_rouge,
                        )
                        for reference_text in _normalize_reference_entry(reference)
                    ]
                else:
                    reference_lsum = split_sentences_for_rouge(
                        reference,
                        segmenter,
                        sentence_split_for_rouge=sentence_split_for_rouge,
                    )

            if multi_reference:
                scores = scorer.score_multi(reference_lsum, generated_lsum)
            else:
                scores = scorer.score(reference_lsum, generated_lsum)
            for key in totals:
                totals[key][0] += scores[key].precision
                totals[key][1] += scores[key].recall
                totals[key][2] += scores[key].fmeasure

        metrics["rouge_totals"] = totals
        metrics["rouge_scores"] = {
            key: totals[key][2] / num_samples * 100 for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        }

        for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            precision = totals[key][0] / num_samples * 100
            recall = totals[key][1] / num_samples * 100
            f1 = totals[key][2] / num_samples * 100
            print(f"  {key:8s}  Precision={precision:.2f}%  Recall={recall:.2f}%  F1={f1:.2f}%")

    return metrics


def _persist_eval_checkpoint(checkpoint_path, metrics):
    if not checkpoint_path:
        return
    tmp = f"{checkpoint_path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, checkpoint_path)
    except Exception as exc:
        print(f"  [eval-checkpoint] save failed: {exc!r}")


METRIC_RESULT_KEYS = {
    "bertscore": ("bert_P", "bert_R", "bert_F"),
    "factcc": ("factcc",),
    "minicheck": ("minicheck",),
    "alignscore": ("alignscore",),
    "factkb": ("factkb",),
}


def _load_eval_checkpoint(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
    except Exception as exc:
        print(f"  [eval-checkpoint] load failed: {exc!r}")
        return {}

    if not isinstance(metrics, dict):
        print(f"  [eval-checkpoint] ignoring non-dict payload in {checkpoint_path}")
        return {}

    print(f"  [eval-resume] loaded partial checkpoint: {checkpoint_path}")
    return metrics


def _metric_is_complete(metrics, metric_name):
    result_keys = METRIC_RESULT_KEYS.get(metric_name, ())
    if any(key in metrics for key in result_keys):
        return True
    return metric_name in metrics.get("metric_errors", {})


def _record_metric_seconds(metrics, metric_name, seconds):
    metric_seconds = metrics.setdefault("metric_seconds", {})
    metric_seconds[metric_name] = round(float(seconds), 2)


def _merge_eval_checkpoint(metrics, restored_metrics, requested_metric_names):
    if not restored_metrics:
        return

    metric_errors = metrics.setdefault("metric_errors", {})
    metric_errors.update(restored_metrics.get("metric_errors", {}))

    restored_metric_seconds = restored_metrics.get("metric_seconds")
    if isinstance(restored_metric_seconds, dict):
        metrics["metric_seconds"] = dict(restored_metric_seconds)

    for metric_name in requested_metric_names:
        for key in METRIC_RESULT_KEYS.get(metric_name, ()):
            if key in restored_metrics:
                metrics[key] = restored_metrics[key]


def _print_reused_metric(metric_name, metrics):
    metric_errors = metrics.get("metric_errors", {})
    if metric_name in metric_errors:
        print(f"  [eval-resume] reusing previous {metric_name} failure: {metric_errors[metric_name]}")
        return

    if metric_name == "bertscore" and "bert_F" in metrics:
        print(
            "  [eval-resume] reusing previous BERTScore: "
            f"Precision={metrics['bert_P']:.2f}%  "
            f"Recall={metrics['bert_R']:.2f}%  "
            f"F1={metrics['bert_F']:.2f}%"
        )
    elif metric_name in metrics:
        print(f"  [eval-resume] reusing previous {metric_name}: {metrics[metric_name]:.2f}%")


def run_all_evaluations(
    generated_summaries,
    articles,
    references,
    device,
    segmenter,
    num_samples,
    rouge_impl="hf",
    eval_suite="full",
    table_metric_names=None,
    extra_metric_names=None,
    sentence_split_for_rouge="nltk",
    eval_batch_size=None,
    checkpoint_path=None,
):
    generated_summaries = list(generated_summaries)
    articles = list(articles)
    references = list(references)
    num_samples = len(generated_summaries) if num_samples is None else int(num_samples)

    use_extended_mode = table_metric_names is not None or extra_metric_names is not None or eval_suite != "full"

    if use_extended_mode:
        rouge_started_at = perf_counter()
        metrics = _run_rouge(
            generated_summaries,
            references,
            segmenter,
            num_samples,
            rouge_impl=rouge_impl,
            use_table_sentence_split=True,
            sentence_split_for_rouge=sentence_split_for_rouge,
        )
        _record_metric_seconds(metrics, "rouge", perf_counter() - rouge_started_at)
        metrics["eval_suite"] = eval_suite
        metrics["metric_errors"] = {}

        if table_metric_names is None and extra_metric_names is None:
            if eval_suite == "rouge_only":
                table_metric_names = ["rouge"]
                extra_metric_names = []
            else:
                table_metric_names = list(DEFAULT_table_metric_names)
                extra_metric_names = list(DEFAULT_EXTRA_METRIC_NAMES)
        else:
            table_metric_names = normalize_metric_names(table_metric_names)
            extra_metric_names = normalize_metric_names(extra_metric_names)

        requested_metric_names = set(table_metric_names) | set(extra_metric_names)
        if not requested_metric_names:
            requested_metric_names.add("rouge")
            table_metric_names = ["rouge"]
            extra_metric_names = []

        metrics["table_metric_names"] = table_metric_names
        metrics["extra_metric_names"] = extra_metric_names

        if requested_metric_names == {"rouge"}:
            print(f"\n{'=' * 60}")
            print("Skipping non-ROUGE metrics")
            print(f"{'=' * 60}")
            return metrics
    else:
        rouge_started_at = perf_counter()
        metrics = _run_rouge(
            generated_summaries,
            references,
            segmenter,
            num_samples,
            rouge_impl=rouge_impl,
            use_table_sentence_split=False,
            sentence_split_for_rouge=sentence_split_for_rouge,
        )
        _record_metric_seconds(metrics, "rouge", perf_counter() - rouge_started_at)
        metrics["eval_suite"] = eval_suite
        metrics["metric_errors"] = {}
        metrics["table_metric_names"] = list(DEFAULT_table_metric_names)
        metrics["extra_metric_names"] = list(DEFAULT_EXTRA_METRIC_NAMES)
        requested_metric_names = {
            "bertscore",
            "factcc",
            "minicheck",
            "alignscore",
            "factkb",
        }

    # Filter out degenerate summaries (empty, whitespace-only, or too short)
    # so they don't pollute metric scores or crash downstream tokenizers.
    _MIN_SUMMARY_CHARS = 10
    valid_mask = [len(s.strip()) >= _MIN_SUMMARY_CHARS for s in generated_summaries]
    n_skipped = sum(1 for v in valid_mask if not v)
    if n_skipped:
        generated_summaries = [s for s, v in zip(generated_summaries, valid_mask) if v]
        articles = [a for a, v in zip(articles, valid_mask) if v]
        references = [r for r, v in zip(references, valid_mask) if v]
        num_samples = len(generated_summaries)
        print(f"  [filter] skipped {n_skipped} degenerate summaries (<{_MIN_SUMMARY_CHARS} chars), evaluating {num_samples}")

    metric_batch_sizes = _resolve_metric_batch_sizes(eval_batch_size)
    metrics.setdefault("metric_seconds", {})
    restored_metrics = _load_eval_checkpoint(checkpoint_path)
    _merge_eval_checkpoint(metrics, restored_metrics, requested_metric_names)

    if "bertscore" in requested_metric_names:
        print(f"\n{'=' * 60}")
        print("BERTScore Evaluation")
        print(f"{'=' * 60}")
        if _metric_is_complete(metrics, "bertscore"):
            _print_reused_metric("bertscore", metrics)
        else:
            metric_started_at = perf_counter()
            torch.cuda.empty_cache()
            try:
                bert_precision, bert_recall, bert_f1 = bert_score_fn(
                    generated_summaries,
                    references,
                    model_type="roberta-large",
                    device=str(device),
                    batch_size=metric_batch_sizes["bertscore"],
                    verbose=False,
                )
                metrics["bert_P"] = bert_precision.mean().item() * 100
                metrics["bert_R"] = bert_recall.mean().item() * 100
                metrics["bert_F"] = bert_f1.mean().item() * 100
                print(
                    "  BERTScore  "
                    f"Precision={metrics['bert_P']:.2f}%  "
                    f"Recall={metrics['bert_R']:.2f}%  "
                    f"F1={metrics['bert_F']:.2f}%"
                )
            except Exception as exc:
                metric_errors = metrics.setdefault("metric_errors", {})
                metric_errors["bertscore"] = repr(exc)
                print(f"  BERTScore unavailable: {exc}")
            finally:
                _record_metric_seconds(metrics, "bertscore", perf_counter() - metric_started_at)
                torch.cuda.empty_cache()
                _persist_eval_checkpoint(checkpoint_path, metrics)

    if "factcc" in requested_metric_names:
        print(f"\n{'=' * 60}")
        print("FactCC Evaluation")
        print(f"{'=' * 60}")
        if _metric_is_complete(metrics, "factcc"):
            _print_reused_metric("factcc", metrics)
        else:
            metric_started_at = perf_counter()
            torch.cuda.empty_cache()
            factcc_tokenizer = None
            factcc_model = None
            try:
                factcc_tokenizer, factcc_model, factcc_correct_id = load_factcc_eval_model(device)
                factcc_scores = compute_factcc_summary_scores(
                    generated_summaries,
                    articles,
                    factcc_tokenizer,
                    factcc_model,
                    device,
                    factcc_correct_id,
                    segmenter,
                    batch_size=metric_batch_sizes["factcc"],
                )
                metrics["factcc"] = sum(factcc_scores) / len(factcc_scores) * 100
                print(f"  FactCC (sentence-avg CORRECT prob): {metrics['factcc']:.2f}%")
            except Exception as exc:
                metric_errors = metrics.setdefault("metric_errors", {})
                metric_errors["factcc"] = repr(exc)
                print(f"  FactCC unavailable: {exc}")
            finally:
                if factcc_model is not None:
                    del factcc_model
                if factcc_tokenizer is not None:
                    del factcc_tokenizer
                _record_metric_seconds(metrics, "factcc", perf_counter() - metric_started_at)
                torch.cuda.empty_cache()
                _persist_eval_checkpoint(checkpoint_path, metrics)

    if "minicheck" in requested_metric_names:
        print(f"\n{'=' * 60}")
        print("MiniCheck Evaluation")
        print(f"{'=' * 60}")
        if _metric_is_complete(metrics, "minicheck"):
            _print_reused_metric("minicheck", metrics)
        else:
            metric_started_at = perf_counter()
            torch.cuda.empty_cache()
            minicheck_scorer = None
            try:
                minicheck_scorer = load_minicheck_model(device=device, batch_size=metric_batch_sizes["minicheck"])
                minicheck_scores = compute_minicheck_summary_scores(
                    generated_summaries,
                    articles,
                    minicheck_scorer,
                    segmenter,
                )
                metrics["minicheck"] = sum(minicheck_scores) / len(minicheck_scores) * 100
                print(f"  SummaryAvgConsistent: {metrics['minicheck']:.2f}%")
            except Exception as exc:
                metric_errors = metrics.setdefault("metric_errors", {})
                metric_errors["minicheck"] = repr(exc)
                print(f"  MiniCheck unavailable: {exc}")
            finally:
                if minicheck_scorer is not None:
                    del minicheck_scorer
                _record_metric_seconds(metrics, "minicheck", perf_counter() - metric_started_at)
                torch.cuda.empty_cache()
                _persist_eval_checkpoint(checkpoint_path, metrics)

    if "alignscore" in requested_metric_names:
        print(f"\n{'=' * 60}")
        print("AlignScore Evaluation")
        print(f"{'=' * 60}")
        if _metric_is_complete(metrics, "alignscore"):
            _print_reused_metric("alignscore", metrics)
        else:
            metric_started_at = perf_counter()
            torch.cuda.empty_cache()
            alignscore_scorer = None
            try:
                alignscore_scorer = load_alignscore_model(
                    device,
                    batch_size=metric_batch_sizes["alignscore"],
                    verbose=False,
                )
                alignscore_scores = compute_alignscore_summary_scores(
                    generated_summaries,
                    articles,
                    alignscore_scorer,
                    batch_size=metric_batch_sizes["alignscore"],
                )
                metrics["alignscore"] = sum(alignscore_scores) / len(alignscore_scores) * 100
                print(f"  SummaryAvg: {metrics['alignscore']:.2f}%")
            except Exception as exc:
                metric_errors = metrics.setdefault("metric_errors", {})
                metric_errors["alignscore"] = repr(exc)
                print(f"  AlignScore unavailable: {exc}")
            finally:
                if alignscore_scorer is not None:
                    del alignscore_scorer
                _record_metric_seconds(metrics, "alignscore", perf_counter() - metric_started_at)
                torch.cuda.empty_cache()
                _persist_eval_checkpoint(checkpoint_path, metrics)

    if "factkb" in requested_metric_names:
        print(f"\n{'=' * 60}")
        print("FactKB Evaluation")
        print(f"{'=' * 60}")
        if _metric_is_complete(metrics, "factkb"):
            _print_reused_metric("factkb", metrics)
        else:
            metric_started_at = perf_counter()
            torch.cuda.empty_cache()
            factkb_tokenizer = None
            factkb_model = None
            try:
                factkb_tokenizer, factkb_model, factkb_factual_idx = load_factkb_model(device)
                factkb_scores = compute_factkb_summary_scores(
                    generated_summaries,
                    articles,
                    factkb_tokenizer,
                    factkb_model,
                    device,
                    factkb_factual_idx,
                    batch_size=metric_batch_sizes["factkb"],
                )
                metrics["factkb"] = sum(factkb_scores) / len(factkb_scores) * 100
                print(f"  FactKB (factual prob): {metrics['factkb']:.2f}%")
            except Exception as exc:
                metric_errors = metrics.setdefault("metric_errors", {})
                metric_errors["factkb"] = repr(exc)
                print(f"  FactKB unavailable: {exc}")
            finally:
                if factkb_model is not None:
                    del factkb_model
                if factkb_tokenizer is not None:
                    del factkb_tokenizer
                _record_metric_seconds(metrics, "factkb", perf_counter() - metric_started_at)
                torch.cuda.empty_cache()
                _persist_eval_checkpoint(checkpoint_path, metrics)

    return metrics
