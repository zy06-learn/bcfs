"""Experiment orchestration for gemma-4-E4B generate-then-optimize runs."""

from __future__ import annotations

from contextlib import nullcontext
import os
import sys
import json
from time import perf_counter
from datetime import datetime
import logging

import nltk
import torch
from datasets import load_dataset

from core.data import load_summarization_dataset
from rouge_score import rouge_scorer
from core.config import (
    BUDGET_SENTENCES,
    DEFAULT_ROUGE_IMPL,
    EVAL_BATCH_SIZE,
    GENERATION_BATCH_SIZE,
    OBJECTIVE_VARIANTS,
    GENERATION_DO_SAMPLE,
    GENERATOR_NAME,
    MAX_GENERATION_NEW_TOKENS,
    GENERATION_MIN_P,
    GENERATION_PRESENCE_PENALTY,
    UTILITY_BATCH_SIZE,
    TRI_METRIC_WEIGHTS_BY_METHOD,
)
from core.features import (
    compute_redundancy_matrix,
    compute_tri_metric_utility_scores,
    compute_utility_scores,
    load_utility_model,
)
from metrics.evaluation import run_all_evaluations
from opt_selectors import get_selector
from opt_selectors.tri_metric import normalize_tri_metric_weights
from output.result_saver import save_results
from core.model_generation import SummaryGenerator, get_summary_prompt_version


CHECKPOINT_INTERVAL = 20
ORDERING_CHOICES = {"source_similarity", "none"}
GENERATOR_CHOICES = {GENERATOR_NAME}
_ORDERING_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_TRACE_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    use_stemmer=True,
)


class _NLTKSegmenter:
    def segment(self, text: str) -> list[str]:
        return nltk.sent_tokenize(text)


class _TeeStream:
    """Mirror writes to multiple streams. Used to mirror stdout/stderr to a per-run log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self._streams[0].isatty()
        except Exception:
            return False


def _setup_run_logger(method: str, output_dir: str | None):
    """Open a per-run log file under output_dir (or this model's results dir) and tee stdout/stderr into it.

    Returns (log_path, restore_fn). Caller must invoke restore_fn() in a finally block.
    """
    base_dir = output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(base_dir, f"run_{method}_{timestamp}.log")
    log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
    log_fh.write(f"\n===== run start {datetime.now().isoformat()} method={method} =====\n")
    log_fh.flush()
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, log_fh)
    sys.stderr = _TeeStream(orig_stderr, log_fh)

    def _restore():
        try:
            log_fh.write(f"===== run end {datetime.now().isoformat()} =====\n")
            log_fh.flush()
            log_fh.close()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    return log_path, _restore


def _tri_metric_weight_labels(method: str) -> dict[str, str]:
    return {
        "rouge": "rouge",
        "minicheck": "minicheck",
        "redundancy": "redundancy",
    }


def _format_tri_metric_weights(method: str, weights: dict[str, float], precision: int = 4) -> str:
    labels = _tri_metric_weight_labels(method)
    fmt = f"{{:.{precision}f}}"
    return ", ".join(
        f"{labels[key]}={fmt.format(weights[key])}"
        for key in ("rouge", "minicheck", "redundancy")
    )


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_big_gpu(device: torch.device) -> bool:
    if device.type != "cuda":
        return False

    props = torch.cuda.get_device_properties(device)
    total_gib = props.total_memory / (1024 ** 3)
    name = props.name.lower()
    gpu_keywords = ("gb10", "h100", "h200", "a100", "b100", "b200", "l40", "rtx 6000 ada")
    return total_gib >= 40 or props.major >= 9 or any(keyword in name for keyword in gpu_keywords)


def _resolve_compute_dtype(requested_dtype: str, device: torch.device):
    if device.type != "cuda":
        return None, "fp32"

    if requested_dtype == "fp32":
        return None, "fp32"
    if requested_dtype == "fp16":
        return torch.float16, "fp16"
    if requested_dtype == "bf16":
        return torch.bfloat16, "bf16"

    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    if is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def _autocast_context(device: torch.device, compute_dtype):
    if device.type != "cuda" or compute_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=compute_dtype)


def _resolve_runtime_profile(args, device: torch.device, beam_size: int):
    big_gpu = _is_big_gpu(device)
    auto_generation_batch = GENERATION_BATCH_SIZE
    auto_utility_batch = UTILITY_BATCH_SIZE
    auto_eval_batch = EVAL_BATCH_SIZE

    generation_batch_size = int(args.generation_batch_size or auto_generation_batch)
    utility_batch_size = int(args.utility_batch_size or auto_utility_batch)
    eval_batch_size = int(args.eval_batch_size or auto_eval_batch)
    return {
        "big_gpu": big_gpu,
        "generation_batch_size": max(1, generation_batch_size),
        "utility_batch_size": max(1, utility_batch_size),
        "eval_batch_size": max(1, eval_batch_size),
    }


def resolve_objective(method: str, objective_arg: str | None, tri_metric: bool = False):
    """Resolve the effective objective and selector mode."""
    is_generation_baseline = method == "baseline"
    is_any_baseline = is_generation_baseline

    if is_generation_baseline:
        return "baseline", False, False, "Baseline from the generation backend"
    if tri_metric:
        if is_any_baseline:
            print("Error: --tri-metric is only supported for optimization methods, not baselines.")
            sys.exit(1)
        return "tri_metric", False, True, "Tri-metric unified objective (ROUGE + MiniCheck + Redundancy)"
    if objective_arg is None:
        print("Error: --objective is required for sentence-level methods (ilp, mmr, dpp)")
        sys.exit(1)

    obj_config = OBJECTIVE_VARIANTS[objective_arg]
    return objective_arg, obj_config["use_minicheck_utility"], obj_config["use_rouge_redundancy"], obj_config["description"]


def _decode_candidate_text(payload, tokenizer) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        return str(
            payload.get("postprocessed_summary")
            or payload.get("summary")
            or payload.get("raw_model_output")
            or ""
        ).strip()
    return tokenizer.decode(
        payload,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _candidate_payload_trace_fields(payload) -> dict:
    if not isinstance(payload, dict):
        return {}
    trace_fields = {}
    for key in ("raw_model_output", "postprocessed_summary", "prompt", "prompt_version"):
        if key in payload:
            trace_fields[key] = payload[key]
    return trace_fields


def generate_baseline_summary(candidates, tokenizer, segmenter, is_raw: bool, label: str | None = None):
    """Generate a baseline summary from the top beam."""
    top1_score, top1_payload = candidates[0]
    full_text = _decode_candidate_text(top1_payload, tokenizer)
    sentences = [sentence.strip() for sentence in segmenter.segment(full_text) if sentence.strip()]

    if is_raw:
        final_summary = full_text
        selected_indices = list(range(len(sentences)))
    else:
        final_summary = " ".join(sentences[:BUDGET_SENTENCES]) if sentences else full_text
        selected_indices = list(range(min(len(sentences), BUDGET_SENTENCES)))

    sample_log = {
        "optimization_method": label or ("Baseline" if is_raw else f"Baseline first {BUDGET_SENTENCES} sentences"),
        "beam_candidates": [
            {
                "score": score,
                "summary": _decode_candidate_text(tokens, tokenizer),
                **_candidate_payload_trace_fields(tokens),
            }
            for score, tokens in candidates
        ],
        "selected_beam_index": 0,
        "selected_beam_score": top1_score,
        "total_sentences": len(sentences),
    }
    return final_summary, selected_indices, sample_log


def build_candidate_pool_trace(candidates, tokenizer, segmenter):
    """Decode beam candidates and build the deduplicated sentence pool with provenance."""
    beam_candidates = []
    unique_sentences = []
    seen_to_index = {}
    pool_sources = []

    for beam_index, (beam_score, cand_tokens) in enumerate(candidates):
        full_text = _decode_candidate_text(cand_tokens, tokenizer)
        sentences = [sent.strip() for sent in segmenter.segment(full_text) if sent.strip()]
        beam_candidates.append(
            {
                "beam_index": beam_index,
                "beam_score": float(beam_score),
                "summary": full_text,
                **_candidate_payload_trace_fields(cand_tokens),
                "sentences": list(sentences),
            }
        )

        for sentence_position, sentence in enumerate(sentences):
            if sentence not in seen_to_index:
                seen_to_index[sentence] = len(unique_sentences)
                unique_sentences.append(sentence)
                pool_sources.append([])
            pool_idx = seen_to_index[sentence]
            pool_sources[pool_idx].append(
                {
                    "beam_index": beam_index,
                    "beam_score": float(beam_score),
                    "sentence_position": sentence_position,
                }
            )

    return unique_sentences, beam_candidates, pool_sources


def _source_sentences(article: str, segmenter) -> list[str]:
    return [sent.strip() for sent in segmenter.segment(article) if sent.strip()]


def _best_source_match(selected_sentence: str, source_sentences: list[str]) -> dict:
    best_index = 0
    best_score = -1.0
    best_sentence = ""
    for source_index, source_sentence in enumerate(source_sentences):
        score = _ORDERING_ROUGE_SCORER.score(source_sentence, selected_sentence)["rougeL"].fmeasure
        if score > best_score:
            best_index = source_index
            best_score = float(score)
            best_sentence = source_sentence

    return {
        "source_index": best_index,
        "source_sentence": best_sentence,
        "similarity": best_score,
    }


def order_selected_sentences(
    selected_indices,
    selected_sentences,
    article: str,
    segmenter,
    ordering_method: str = "source_similarity",
):
    """Realize selected content in a coherent order.

    source_similarity maps each selected sentence to its nearest source-document
    sentence by ROUGE-L F1, then orders selected content by that source position.
    """
    method = ordering_method or "source_similarity"
    if method not in ORDERING_CHOICES:
        raise ValueError(f"Unknown ordering method: {method}. Available: {sorted(ORDERING_CHOICES)}")

    selected_indices = list(selected_indices)
    selected_sentences = list(selected_sentences)
    if method == "none" or len(selected_sentences) <= 1:
        matches = [
            {
                "selected_index": idx,
                "selected_sentence": sentence,
                "source_index": None,
                "source_sentence": "",
                "similarity": None,
            }
            for idx, sentence in zip(selected_indices, selected_sentences)
        ]
        return selected_indices, selected_sentences, {"method": method, "matches": matches}

    source_sentences = _source_sentences(article, segmenter)
    if not source_sentences:
        return selected_indices, selected_sentences, {"method": method, "matches": []}

    rows = []
    for selected_position, (selected_index, selected_sentence) in enumerate(zip(selected_indices, selected_sentences)):
        match = _best_source_match(selected_sentence, source_sentences)
        match.update(
            {
                "selected_index": selected_index,
                "selected_position": selected_position,
                "selected_sentence": selected_sentence,
            }
        )
        rows.append(match)

    rows.sort(key=lambda row: (row["source_index"], row["selected_position"]))
    ordered_indices = [row["selected_index"] for row in rows]
    ordered_sentences = [row["selected_sentence"] for row in rows]
    return ordered_indices, ordered_sentences, {"method": method, "matches": rows}


def generate_sentence_level_summary(
    candidates,
    tokenizer,
    segmenter,
    article,
    method,
    objective_desc,
    use_minicheck_utility,
    use_rouge_redundancy,
    minicheck_utility_scorer,
    selector_fn,
    device,
    tri_metric_weights=None,
    redundancy_threshold_override=None,
    ilp_penalty_scale=None,
    ordering_method="source_similarity",
):
    """Generate a summary with a sentence-level selector."""
    co_started_at = perf_counter()
    unique_sentences, beam_candidates, pool_sources = build_candidate_pool_trace(candidates, tokenizer, segmenter)
    if tri_metric_weights is not None:
        utility_scores, tri_metric_metadata = compute_tri_metric_utility_scores(
            unique_sentences,
            article,
            minicheck_utility_scorer,
            tri_metric_weights["rouge"],
            tri_metric_weights["minicheck"],
            tri_metric_weights["redundancy"],
        )
        utility_mode = "tri_metric"
        selector_kwargs = {
            "utility_mode": utility_mode,
            "tri_metric_weights": tri_metric_weights,
        }
    else:
        utility_scores = compute_utility_scores(
            unique_sentences,
            article,
            use_minicheck_utility,
            minicheck_scorer=minicheck_utility_scorer,
            segmenter=segmenter,
            device=device,
        )
        tri_metric_metadata = None
        utility_mode = "legacy"
        selector_kwargs = {}
    if redundancy_threshold_override is not None and method == "ilp":
        selector_kwargs["redundancy_threshold"] = float(redundancy_threshold_override)
    if method == "ilp" and ilp_penalty_scale is not None:
        selector_kwargs["penalty_scale"] = ilp_penalty_scale
    raw_redundancy_matrix = compute_redundancy_matrix(unique_sentences, use_rouge_redundancy)
    if tri_metric_weights is not None and use_rouge_redundancy:
        # Per-sample min-max normalization on off-diagonal redundancy values
        off_diag = [raw_redundancy_matrix[i][j]
                    for i in range(len(unique_sentences))
                    for j in range(len(unique_sentences)) if i != j]
        lo = min(off_diag) if off_diag else 0.0
        hi = max(off_diag) if off_diag else 1.0
        span = hi - lo if hi - lo > 1e-9 else 1.0
        N = len(unique_sentences)
        redundancy_matrix = [[0.0] * N for _ in range(N)]
        for i in range(N):
            redundancy_matrix[i][i] = 1.0
            for j in range(N):
                if i != j:
                    redundancy_matrix[i][j] = (raw_redundancy_matrix[i][j] - lo) / span
    else:
        redundancy_matrix = raw_redundancy_matrix
    selected_indices = selector_fn(
        unique_sentences,
        utility_scores,
        redundancy_matrix,
        BUDGET_SENTENCES,
        **selector_kwargs,
    )
    selected_sentences = [unique_sentences[idx] for idx in selected_indices]
    ordered_indices, ordered_sentences, ordering_log = order_selected_sentences(
        selected_indices,
        selected_sentences,
        article,
        segmenter,
        ordering_method,
    )
    final_summary = " ".join(ordered_sentences)

    sample_log = {
        "optimization_method": f"{method.upper()} ({objective_desc})",
        "beam_candidates": beam_candidates,
        "pool": list(unique_sentences),
        "pool_sources": pool_sources,
        "utility_scores": list(utility_scores),
        "redundancy_shape": (len(unique_sentences), len(unique_sentences)),
        "selected_indices": list(selected_indices),
        "ordered_selected_indices": list(ordered_indices),
        "utility_sum": sum(utility_scores[idx] for idx in selected_indices),
        "redundancy_matrix": redundancy_matrix,
        "utility_mode": utility_mode,
        "selector_kwargs": selector_kwargs,
        "ordering_method": ordering_log["method"],
        "ordering_matches": ordering_log["matches"],
        "pre_order_summary": " ".join(selected_sentences),
        "co_timing_seconds": {
            "total": round(perf_counter() - co_started_at, 6),
        },
    }
    if tri_metric_metadata is not None:
        sample_log["rouge_sentence_scores"] = list(tri_metric_metadata["rouge_scores"])
        sample_log["minicheck_sentence_scores"] = list(tri_metric_metadata["minicheck_scores"])
        sample_log["calibrated_rouge_sentence_scores"] = list(tri_metric_metadata["calibrated_rouge_scores"])
        sample_log["calibrated_minicheck_sentence_scores"] = list(tri_metric_metadata["calibrated_minicheck_scores"])
        sample_log["effective_weights"] = dict(tri_metric_metadata["effective_weights"])
    return final_summary, selected_indices, sample_log


def _summarize_co_timing(all_sample_logs):
    """Aggregate sentence-level CO timing from per-sample logs."""
    timing_rows = [
        log.get("co_timing_seconds")
        for log in all_sample_logs
        if isinstance(log, dict) and isinstance(log.get("co_timing_seconds"), dict)
    ]
    if not timing_rows:
        return {}

    values = [float(row.get("total", 0.0)) for row in timing_rows]
    total = sum(values)
    summary = {
        "samples": len(timing_rows),
        "total": round(total, 2),
        "mean": round(total / len(values), 6),
    }
    return summary


def _build_result_paths(args, method, objective_name, is_any_baseline):
    generator = getattr(args, "generator", GENERATOR_NAME)
    if args.output_dir:
        save_dir = args.output_dir
    else:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            generator,
            getattr(args, "dataset", "cnn_dailymail"),
            f"{method}_{objective_name}",
        )
    os.makedirs(save_dir, exist_ok=True)

    result_name_parts = [method] if is_any_baseline else [f"beam{args.beam_size}", method]
    append_objective_tag = not is_any_baseline
    if append_objective_tag:
        result_name_parts.append(objective_name)
    if not args.use_local_rouge:
        result_name_parts.append("hfrouge")
    if args.split != "test":
        result_name_parts.append(args.split)
    if getattr(args, "sample_mode", "head") != "head":
        result_name_parts.append(args.sample_mode)
        result_name_parts.append(f"seed{int(getattr(args, 'sample_seed', 42))}")
    result_filename = "_".join(result_name_parts) + "_results.txt"
    return os.path.join(save_dir, result_filename)


def _write_progress_checkpoint(
    checkpoint_json_path: str,
    checkpoint_jsonl_path: str,
    method: str,
    objective_desc: str,
    split_name: str,
    sample_mode: str,
    sample_seed: int,
    beam_size: int,
    sentence_split_for_rouge: str,
    ordering_method: str,
    articles,
    references,
    generated_summaries,
    start_index: int,
    end_index: int,
    total_samples: int,
    gen_seconds_so_far: float = 0.0,
    extra_config: dict | None = None,
):
    extra_config = dict(extra_config or {})
    with open(checkpoint_jsonl_path, "a", encoding="utf-8") as handle:
        for idx in range(start_index, end_index):
            handle.write(
                json.dumps(
                    {
                        "sample_index": idx,
                        "article": articles[idx],
                        "reference": references[idx],
                        "summary": generated_summaries[idx],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(checkpoint_json_path, "w", encoding="utf-8") as handle:
        payload = {
            "method": method,
            "objective": objective_desc,
            "split": split_name,
            "sample_mode": sample_mode,
            "sample_seed": sample_seed,
            "beam_size": beam_size,
            "sentence_split_for_rouge": sentence_split_for_rouge,
            "ordering": ordering_method,
            "samples_total": total_samples,
            "samples_completed": end_index,
            "gen_seconds_so_far": round(float(gen_seconds_so_far), 2),
        }
        payload.update(extra_config)
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _stage_trace_path(result_file: str) -> str:
    return result_file.replace("_results.txt", "_stage_outputs.jsonl")


def _prepare_stage_trace_for_resume(stage_trace_jsonl: str, start_index: int) -> list[dict]:
    """Keep stage trace rows aligned with the checkpoint boundary before resuming.

    Stage traces are appended per sample, while checkpoints are written in
    batches. If an interrupted run writes samples after the last checkpoint,
    resuming from that checkpoint would otherwise duplicate those trace rows.
    """
    prior_logs = [{} for _ in range(start_index)]
    if start_index <= 0 or not os.path.exists(stage_trace_jsonl):
        return prior_logs

    rows_by_index: dict[int, dict] = {}
    total_rows = 0
    changed = False
    with open(stage_trace_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            total_rows += 1
            try:
                row = json.loads(line)
                idx = int(row.get("sample_index", -1))
            except (json.JSONDecodeError, TypeError, ValueError):
                changed = True
                continue
            if idx < 0 or idx >= start_index:
                changed = True
                continue
            if idx in rows_by_index:
                changed = True
            rows_by_index[idx] = row

    if len(rows_by_index) != min(start_index, total_rows):
        changed = True

    ordered_rows = [rows_by_index[idx] for idx in sorted(rows_by_index)]
    if changed:
        with open(stage_trace_jsonl, "w", encoding="utf-8") as handle:
            for row in ordered_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[resume] pruned stage trace to {len(ordered_rows)} rows "
            f"before checkpoint {start_index}: {stage_trace_jsonl}"
        )

    for idx, row in rows_by_index.items():
        timing = row.get("stage2_co_selection", {}).get("co_timing_seconds")
        if isinstance(timing, dict):
            prior_logs[idx] = {"co_timing_seconds": timing}
    return prior_logs


def _score_final_summary_for_trace(summary: str, reference: str) -> dict:
    scores = _TRACE_ROUGE_SCORER.score(reference, summary)
    return {
        metric_name: {
            "precision": float(score.precision),
            "recall": float(score.recall),
            "f1": float(score.fmeasure),
        }
        for metric_name, score in scores.items()
    }


def _selected_pair_scores(selected_indices, redundancy_matrix) -> list[dict]:
    selected_indices = list(selected_indices or [])
    if not redundancy_matrix:
        return []
    rows = []
    for left_pos, left_idx in enumerate(selected_indices):
        for right_idx in selected_indices[left_pos + 1:]:
            try:
                score = float(redundancy_matrix[left_idx][right_idx])
            except (IndexError, TypeError, ValueError):
                continue
            rows.append({"left": left_idx, "right": right_idx, "redundancy": score})
    return rows


def _candidate_pool_trace_rows(log: dict) -> list[dict]:
    pool = log.get("pool", [])
    utility_scores = log.get("utility_scores", [])
    rouge_scores = log.get("rouge_sentence_scores", [])
    minicheck_scores = log.get("minicheck_sentence_scores", [])
    calibrated_rouge_scores = log.get("calibrated_rouge_sentence_scores", [])
    calibrated_minicheck_scores = log.get("calibrated_minicheck_sentence_scores", [])
    pool_sources = log.get("pool_sources", [])
    selected = set(log.get("selected_indices", []))
    ordered_positions = {
        idx: pos for pos, idx in enumerate(log.get("ordered_selected_indices", []))
    }

    rows = []
    for idx, sentence in enumerate(pool):
        sources = pool_sources[idx] if idx < len(pool_sources) else []
        best_source_score = None
        if sources:
            best_source_score = max(float(source["beam_score"]) for source in sources)
        row = {
            "pool_index": idx,
            "sentence": sentence,
            "selected": idx in selected,
            "ordered_position": ordered_positions.get(idx),
            "beam_sources": sources,
            "best_beam_score": best_source_score,
        }
        if idx < len(utility_scores):
            row["utility_score"] = float(utility_scores[idx])
        if idx < len(rouge_scores):
            row["rouge_coverage_score"] = float(rouge_scores[idx])
        if idx < len(minicheck_scores):
            row["minicheck_score"] = float(minicheck_scores[idx])
        if idx < len(calibrated_rouge_scores):
            row["calibrated_rouge_coverage_score"] = float(calibrated_rouge_scores[idx])
        if idx < len(calibrated_minicheck_scores):
            row["calibrated_minicheck_score"] = float(calibrated_minicheck_scores[idx])
        rows.append(row)
    return rows


def _build_stage_trace_entry(
    sample_index: int,
    method: str,
    objective_desc: str,
    article: str,
    reference: str,
    final_summary: str,
    log: dict,
) -> dict:
    common = {
        "sample_index": sample_index,
        "method": method,
        "objective": objective_desc,
        "generator_backend": log.get("generator_backend"),
        "generator_model": log.get("generator_model"),
        "prompt_version": log.get("prompt_version"),
        "source_document": article,
        "reference_summary": reference,
    }
    if method == "baseline":
        return {
            **common,
            "stage1_llm_summary_generation": {
                "output_name": "final_summary",
                "generator_backend": log.get("generator_backend"),
                "generator_model": log.get("generator_model"),
                "prompt_version": log.get("prompt_version"),
                "generator_candidate_count": log.get("generator_candidate_count"),
                "beam_candidates": log.get("beam_candidates", []),
                "selected_beam_index": log.get("selected_beam_index"),
                "selected_beam_score": log.get("selected_beam_score"),
                "generated_summary": final_summary,
                "output_sentence_count": log.get("total_sentences"),
                "scores": {
                    "rouge_against_reference": _score_final_summary_for_trace(final_summary, reference),
                },
            },
        }

    selected_indices = list(log.get("selected_indices", []))
    selected_sentences = [
        log.get("pool", [])[idx]
        for idx in selected_indices
        if isinstance(idx, int) and idx < len(log.get("pool", []))
    ]
    ordered_indices = list(log.get("ordered_selected_indices", selected_indices))
    ordered_sentences = [
        log.get("pool", [])[idx]
        for idx in ordered_indices
        if isinstance(idx, int) and idx < len(log.get("pool", []))
    ]
    selected_pair_scores = _selected_pair_scores(selected_indices, log.get("redundancy_matrix"))
    selected_pair_raw_scores = _selected_pair_scores(selected_indices, log.get("raw_redundancy_matrix"))

    return {
        **common,
        "stage1_llm_candidate_generation": {
            "output_name": "candidate_pool",
            "generator_backend": log.get("generator_backend"),
            "generator_model": log.get("generator_model"),
            "prompt_version": log.get("prompt_version"),
            "generator_candidate_count": log.get("generator_candidate_count"),
            "beam_candidates": log.get("beam_candidates", []),
            "candidate_pool": _candidate_pool_trace_rows(log),
            "candidate_pool_size": len(log.get("pool", [])),
        },
        "stage2_co_selection": {
            "output_name": "selected_subset",
            "utility_mode": log.get("utility_mode"),
            "effective_weights": log.get("effective_weights"),
            "selector_kwargs": log.get("selector_kwargs", {}),
            "selected_indices": selected_indices,
            "selected_sentences_unordered": selected_sentences,
            "selected_utility_sum": log.get("utility_sum"),
            "selected_pair_redundancy": selected_pair_scores,
            "selected_pair_raw_redundancy": selected_pair_raw_scores,
            "redundancy_matrix": log.get("redundancy_matrix"),
            "raw_redundancy_matrix": log.get("raw_redundancy_matrix"),
            "co_timing_seconds": log.get("co_timing_seconds"),
        },
        "stage3_ordering_and_final_realization": {
            "output_name": "final_summary",
            "ordering_method": log.get("ordering_method"),
            "ordering_matches": log.get("ordering_matches", []),
            "ordered_selected_indices": ordered_indices,
            "ordered_selected_sentences": ordered_sentences,
            "pre_order_summary": log.get("pre_order_summary"),
            "final_summary": final_summary,
            "final_scores": {
                "rouge_against_reference": _score_final_summary_for_trace(final_summary, reference),
            },
        },
    }


def _append_stage_trace(
    trace_jsonl_path: str,
    sample_index: int,
    method: str,
    objective_desc: str,
    article: str,
    reference: str,
    final_summary: str,
    log: dict,
) -> None:
    os.makedirs(os.path.dirname(trace_jsonl_path), exist_ok=True)
    entry = _build_stage_trace_entry(
        sample_index,
        method,
        objective_desc,
        article,
        reference,
        final_summary,
        log,
    )
    with open(trace_jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _materialize_reused_candidates(beam_candidates) -> list[tuple[float, dict]]:
    candidates: list[tuple[float, dict]] = []
    for beam_index, candidate in enumerate(beam_candidates or []):
        if isinstance(candidate, dict):
            payload = dict(candidate)
            score = float(payload.get("beam_score", payload.get("score", 0.0)))
            payload.setdefault("beam_index", beam_index)
            payload.setdefault("summary", payload.get("postprocessed_summary") or payload.get("raw_model_output") or "")
        else:
            score = 0.0
            payload = {"summary": str(candidate)}
        candidates.append((score, payload))
    return candidates


def _load_reuse_stage1_rows(
    reuse_stage1_from: str,
    articles,
    references,
    num_samples: int,
) -> list[dict]:
    resolved_path = os.path.abspath(os.path.expanduser(reuse_stage1_from))
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"reuse stage1 file not found: {resolved_path}")

    rows_by_index: dict[int, dict] = {}
    with open(resolved_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_index = int(row.get("sample_index", -1))
            if 0 <= sample_index < num_samples:
                rows_by_index[sample_index] = row

    missing = [idx for idx in range(num_samples) if idx not in rows_by_index]
    if missing:
        raise ValueError(
            f"reuse stage1 file {resolved_path} is missing {len(missing)} required rows; "
            f"first missing sample_index={missing[0]}"
        )

    reuse_rows = []
    for idx in range(num_samples):
        row = rows_by_index[idx]
        stage1 = row.get("stage1_llm_candidate_generation")
        if not isinstance(stage1, dict):
            raise ValueError(
                f"reuse stage1 file {resolved_path} row {idx} has no stage1_llm_candidate_generation block"
            )

        beam_candidates = stage1.get("beam_candidates") or []
        candidates = _materialize_reused_candidates(beam_candidates)
        if not candidates:
            raise ValueError(
                f"reuse stage1 file {resolved_path} row {idx} has no beam candidates to reuse"
            )

        source_article = row.get("source_document")
        if source_article is not None and source_article != articles[idx]:
            raise ValueError(
                f"reuse stage1 article mismatch at sample_index={idx}: source file does not match current dataset slice"
            )
        source_reference = row.get("reference_summary")
        if source_reference is not None and source_reference != references[idx]:
            raise ValueError(
                f"reuse stage1 reference mismatch at sample_index={idx}: source file does not match current dataset slice"
            )

        reuse_rows.append(
            {
                "candidates": candidates,
                "generator_backend": stage1.get("generator_backend") or row.get("generator_backend"),
                "generator_model": stage1.get("generator_model") or row.get("generator_model"),
                "prompt_version": stage1.get("prompt_version") or row.get("prompt_version"),
                "generator_candidate_count": stage1.get("generator_candidate_count") or len(candidates),
            }
        )

    return reuse_rows


def _try_resume_from_checkpoint(
    checkpoint_json_path: str,
    checkpoint_jsonl_path: str,
    expected_config: dict,
    num_samples: int,
    live_articles=None,
):
    """Return (start_index, prior_summaries, prior_gen_seconds) or None.

    None means caller should treat this as a fresh run (and wipe any stale files).
    Any config mismatch, jsonl corruption, or article-order mismatch falls back to fresh.

    When ``live_articles`` is provided, the helper spot-checks that the jsonl-stored
    articles at idx 0 and idx start_index-1 match the live dataset — this catches cases
    where shuffle/select flags have changed between the original run and the resume
    attempt, which would otherwise silently corrupt the output.
    """
    if not os.path.exists(checkpoint_json_path) or not os.path.exists(checkpoint_jsonl_path):
        return None
    try:
        with open(checkpoint_json_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[resume] failed to read {checkpoint_json_path}: {exc!r}. Starting fresh.")
        return None

    for key, expected in expected_config.items():
        got = meta.get(key)
        if key == "dataset" and got is None:
            got = "cnn_dailymail"
        if got != expected:
            print(
                f"[resume] checkpoint config mismatch on {key!r}: "
                f"expected {expected!r}, got {got!r}. Starting fresh."
            )
            return None
    if int(meta.get("samples_total", -1)) != int(num_samples):
        print(
            f"[resume] samples_total mismatch: expected {num_samples}, "
            f"got {meta.get('samples_total')!r}. Starting fresh."
        )
        return None

    start_index = int(meta.get("samples_completed", 0))
    if start_index <= 0:
        return None
    if start_index > num_samples:
        print(f"[resume] samples_completed={start_index} > total={num_samples}. Starting fresh.")
        return None

    prior_summaries: list = [None] * start_index
    prior_articles_spot = {}
    try:
        with open(checkpoint_jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                idx = int(entry["sample_index"])
                if 0 <= idx < start_index:
                    prior_summaries[idx] = entry["summary"]
                    if idx == 0 or idx == start_index - 1:
                        prior_articles_spot[idx] = entry["article"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"[resume] failed to read {checkpoint_jsonl_path}: {exc!r}. Starting fresh.")
        return None

    missing = [i for i, s in enumerate(prior_summaries) if s is None]
    if missing:
        print(
            f"[resume] jsonl missing {len(missing)} of {start_index} summaries "
            f"(first missing idx={missing[0]}). Starting fresh."
        )
        return None

    if live_articles is not None:
        for spot_idx, ckpt_article in prior_articles_spot.items():
            if spot_idx >= len(live_articles):
                print(
                    f"[resume] live dataset too short for spot-check at idx={spot_idx}. Starting fresh."
                )
                return None
            live_article = live_articles[spot_idx]
            if ckpt_article != live_article:
                print(
                    f"[resume] ARTICLE ORDER MISMATCH at idx={spot_idx}. "
                    f"Checkpoint was generated with different sample_mode/num_samples/split. "
                    f"Refusing to resume to avoid cross-contaminating summaries.\n"
                    f"  ckpt[{spot_idx}][:80]={ckpt_article[:80]!r}\n"
                    f"  live[{spot_idx}][:80]={live_article[:80]!r}"
                )
                return None

    prior_gen_seconds = float(meta.get("gen_seconds_so_far", 0.0))
    return start_index, prior_summaries, prior_gen_seconds


def _resolve_tri_metric_weights(args):
    if not getattr(args, "tri_metric", False):
        return None

    from core.config import (
        TRI_ROUGE_WEIGHT as _DEF_R,
        TRI_MINICHECK_WEIGHT as _DEF_M,
        TRI_REDUNDANCY_WEIGHT as _DEF_D,
    )
    raw_r = float(getattr(args, "w_rouge", _DEF_R))
    raw_m = float(getattr(args, "w_minicheck", _DEF_M))
    raw_d = float(getattr(args, "w_redundancy", _DEF_D))

    cli_at_defaults = (raw_r == _DEF_R and raw_m == _DEF_M and raw_d == _DEF_D)
    method = getattr(args, "method", None)
    if cli_at_defaults and method in TRI_METRIC_WEIGHTS_BY_METHOD:
        per_method = TRI_METRIC_WEIGHTS_BY_METHOD[method]
        raw_r = float(per_method["rouge"])
        raw_m = float(per_method["minicheck"])
        raw_d = float(per_method["redundancy"])
        print(
            f"[tri-metric] Using method-specific weights for '{method}': "
            f"rouge={raw_r:.4f} minicheck={raw_m:.4f} redundancy={raw_d:.4f} (from TRI_METRIC_WEIGHTS_BY_METHOD)."
        )

    raw_weights, _ = normalize_tri_metric_weights(raw_r, raw_m, raw_d)
    return raw_weights


def run_experiment(args) -> str:
    """Run a full BART experiment and return the result file path."""
    global BUDGET_SENTENCES, CHECKPOINT_INTERVAL
    if getattr(args, "budget_sentences", None) is not None:
        BUDGET_SENTENCES = max(1, int(args.budget_sentences))
    if getattr(args, "checkpoint_interval", None) is not None:
        CHECKPOINT_INTERVAL = max(1, int(args.checkpoint_interval))
    log_path, _restore_logger = _setup_run_logger(
        method=getattr(args, "method", "unknown"),
        output_dir=getattr(args, "output_dir", None),
    )
    print(f"[logger] tee'ing stdout/stderr to: {log_path}")
    try:
        return _run_experiment_impl(args)
    finally:
        _restore_logger()


def _run_experiment_impl(args) -> str:
    started_at = perf_counter()
    started_wallclock = datetime.now()
    method = args.method
    generator = getattr(args, "generator", GENERATOR_NAME)
    if generator not in GENERATOR_CHOICES:
        print(f"Error: --generator must be one of {sorted(GENERATOR_CHOICES)}")
        sys.exit(1)
    requested_num_samples = args.num_samples
    beam_size = args.beam_size
    data_split = args.split
    sample_mode = getattr(args, "sample_mode", "shuffle")
    sample_seed = int(getattr(args, "sample_seed", 42))
    ordering_method = getattr(args, "ordering", "source_similarity")
    if ordering_method not in ORDERING_CHOICES:
        print(f"Error: --ordering must be one of {sorted(ORDERING_CHOICES)}")
        sys.exit(1)
    rouge_impl = "local" if args.use_local_rouge else DEFAULT_ROUGE_IMPL
    rouge_impl_desc = "rouge_score" if args.use_local_rouge else "huggingface_evaluate"
    rouge_sentence_split = getattr(args, "rouge_sentence_split", "nltk")

    if beam_size < 1:
        print("Error: --beam-size must be >= 1")
        sys.exit(1)

    is_generation_baseline = method == "baseline"
    is_any_baseline = is_generation_baseline
    reuse_stage1_from = getattr(args, "reuse_stage1_from", None)
    if reuse_stage1_from and is_any_baseline:
        print("Error: --reuse-stage1-from is only supported for sentence-level optimization methods.")
        sys.exit(1)
    tri_metric_weights = _resolve_tri_metric_weights(args)
    objective_name, use_minicheck_utility, use_rouge_redundancy, objective_desc = resolve_objective(
        method,
        args.objective,
        tri_metric=tri_metric_weights is not None,
    )

    dataset_name = getattr(args, "dataset", "cnn_dailymail")
    print(f"Loading {dataset_name} dataset...")
    dataset = load_summarization_dataset(dataset_name)

    segmenter = _NLTKSegmenter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_profile = _resolve_runtime_profile(args, device, beam_size)
    compute_dtype, compute_dtype_desc = _resolve_compute_dtype(getattr(args, "compute_dtype", "auto"), device)

    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)

    model_path = getattr(args, "model_path", None)
    summary_generator = None
    tokenizer = None
    reuse_stage1_rows = None
    resolved_reuse_stage1_from = (
        os.path.abspath(os.path.expanduser(reuse_stage1_from)) if reuse_stage1_from else None
    )
    if is_generation_baseline:
        budget_desc = "direct full output"
    else:
        budget_desc = f"{BUDGET_SENTENCES} sentences"
    ordering_desc = "none (baseline direct generation)" if is_any_baseline else ordering_method
    result_file = _build_result_paths(args, method, objective_name, is_any_baseline)
    checkpoint_json = result_file.replace("_results.txt", "_progress.json")
    checkpoint_jsonl = result_file.replace("_results.txt", "_progress_summaries.jsonl")
    stage_trace_jsonl = _stage_trace_path(result_file)

    prompt_version_desc = (
        summary_generator.prompt_version
        if summary_generator is not None
        else (reuse_stage1_rows[0].get("prompt_version") if reuse_stage1_rows else get_summary_prompt_version(dataset_name))
    )
    model_class_desc = (
        summary_generator.model_class
        if summary_generator is not None
        else ("reused_stage1_candidates" if resolved_reuse_stage1_from else "pending_model_load")
    )
    generator_model_desc = (
        model_path
        if summary_generator is not None
        else (reuse_stage1_rows[0].get("generator_model") if reuse_stage1_rows else model_path)
    )

    print(f"Device: {device}")
    print(f"Generator: {generator}")
    print(f"Generator model: {generator_model_desc}")
    print(f"Method: {method}, Objective: {objective_desc}")
    print(f"Split={data_split}, ROUGE impl={rouge_impl_desc}, ROUGE sentence split={rouge_sentence_split}")
    print(f"Sample mode={sample_mode}, seed={sample_seed}")
    print(f"Ordering: {ordering_desc}")
    print(f"Stage output trace: {stage_trace_jsonl}")
    if resolved_reuse_stage1_from:
        print(f"Stage1 reuse source: {resolved_reuse_stage1_from}")
    if tri_metric_weights is not None:
        print(f"Tri-metric weights: {_format_tri_metric_weights(method, tri_metric_weights, precision=4)}")
    print(
        "Runtime profile: "
        f"big_gpu={runtime_profile['big_gpu']} "
        f"gen_batch={runtime_profile['generation_batch_size']} "
        f"utility_batch={runtime_profile['utility_batch_size']} "
        f"eval_batch={runtime_profile['eval_batch_size']} "
        f"dtype={compute_dtype_desc}"
    )
    if is_generation_baseline:
        print(
            f"LLM direct generation, Budget={budget_desc}, "
            f"MaxInputTokens={getattr(args, 'max_input_tokens', None)}, "
            f"MaxNewTokens={getattr(args, 'max_new_tokens', MAX_GENERATION_NEW_TOKENS)}"
        )
    elif resolved_reuse_stage1_from:
        print(
            f"LLM candidates reused from file, Budget={budget_desc}, "
            f"selector rerun only"
        )
    else:
        print(
            f"LLM candidates={beam_size}, Budget={budget_desc}, "
            f"MaxInputTokens={getattr(args, 'max_input_tokens', None)}, "
            f"MaxNewTokens={getattr(args, 'max_new_tokens', MAX_GENERATION_NEW_TOKENS)}"
        )
    print(
        f"LLM decoding: do_sample={getattr(args, 'do_sample', GENERATION_DO_SAMPLE)}, "
        f"temperature={getattr(args, 'temperature', None)}, "
        f"top_p={getattr(args, 'top_p', None)}, "
        f"top_k={getattr(args, 'top_k', None)}, "
        f"min_p={getattr(args, 'min_p', GENERATION_MIN_P)}, "
        f"presence_penalty={getattr(args, 'presence_penalty', GENERATION_PRESENCE_PENALTY)}, "
        f"thinking={bool(getattr(args, 'enable_thinking', False))}, "
        f"prompt_version={prompt_version_desc}, "
        f"model_class={model_class_desc}\n"
    )

    needs_sentence_level_utility_scorer = (not is_any_baseline) and (
        use_minicheck_utility or tri_metric_weights is not None
    )
    minicheck_utility_scorer = load_utility_model(
        device,
        needs_sentence_level_utility_scorer,
        batch_size=runtime_profile["utility_batch_size"],
    )

    selector_fn = None if is_any_baseline else get_selector(method)

    split_dataset = dataset[data_split]
    if requested_num_samples > 0:
        limit = min(requested_num_samples, len(split_dataset))
        if sample_mode == "shuffle":
            split_dataset = split_dataset.shuffle(seed=sample_seed).select(range(limit))
        else:
            split_dataset = split_dataset.select(range(limit))
    articles = split_dataset["article"]
    references = split_dataset["highlights"]
    num_samples = len(split_dataset)
    if reuse_stage1_from:
        print(f"Loading reused stage1 candidates from {resolved_reuse_stage1_from} ...")
        reuse_stage1_rows = _load_reuse_stage1_rows(
            resolved_reuse_stage1_from,
            articles,
            references,
            num_samples,
        )
        print(
            f"Reusing stage1 candidates for {num_samples} samples "
            f"from {resolved_reuse_stage1_from}"
        )
    else:
        print(f"Loading generation model ({model_path})...")
        summary_generator = SummaryGenerator(
            model_path=model_path,
            dataset_name=dataset_name,
            device=device,
            compute_dtype=compute_dtype,
            max_input_tokens=getattr(args, "max_input_tokens", 8192),
            max_new_tokens=getattr(args, "max_new_tokens", MAX_GENERATION_NEW_TOKENS),
            temperature=getattr(args, "temperature", 0.7),
            top_p=getattr(args, "top_p", 0.8),
            top_k=getattr(args, "top_k", 20),
            min_p=getattr(args, "min_p", GENERATION_MIN_P),
            presence_penalty=getattr(args, "presence_penalty", GENERATION_PRESENCE_PENALTY),
            repetition_penalty=getattr(args, "repetition_penalty", 1.0),
            enable_thinking=bool(getattr(args, "enable_thinking", False)),
            do_sample=getattr(args, "do_sample", GENERATION_DO_SAMPLE),
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        )
        tokenizer = summary_generator.tokenizer

    no_resume = bool(getattr(args, "no_resume", False))
    resume_state = None
    expected_checkpoint_config = {
        "method": method,
        "objective": objective_desc,
        "generator": generator,
        "model_path": model_path,
        "dataset": dataset_name,
        "split": data_split,
        "sample_mode": sample_mode,
        "sample_seed": sample_seed,
        "beam_size": beam_size,
        "sentence_split_for_rouge": rouge_sentence_split,
        "ordering": ordering_desc,
        "reuse_stage1_from": resolved_reuse_stage1_from,
    }
    if generator == GENERATOR_NAME:
        expected_checkpoint_config.update(
            {
                "prompt_version": get_summary_prompt_version(dataset_name),
                "max_input_tokens": int(getattr(args, "max_input_tokens", 8192)),
                "max_new_tokens": int(getattr(args, "max_new_tokens", MAX_GENERATION_NEW_TOKENS)),
                "temperature": float(getattr(args, "temperature", 0.7)),
                "top_p": float(getattr(args, "top_p", 0.8)),
                "top_k": int(getattr(args, "top_k", 20)),
                "min_p": float(getattr(args, "min_p", GENERATION_MIN_P)),
                "presence_penalty": float(getattr(args, "presence_penalty", GENERATION_PRESENCE_PENALTY)),
                "repetition_penalty": float(getattr(args, "repetition_penalty", 1.0)),
                "enable_thinking": bool(getattr(args, "enable_thinking", False)),
                "do_sample": getattr(args, "do_sample", GENERATION_DO_SAMPLE),
            }
        )
    if not no_resume:
        resume_state = _try_resume_from_checkpoint(
            checkpoint_json,
            checkpoint_jsonl,
            expected_checkpoint_config,
            num_samples,
            live_articles=articles,
        )

    if resume_state is not None:
        start_index, prior_summaries, prior_gen_seconds = resume_state
        generated_summaries = list(prior_summaries)
        all_sample_logs = _prepare_stage_trace_for_resume(stage_trace_jsonl, start_index)
        last_checkpoint_written = start_index
        next_checkpoint = start_index + CHECKPOINT_INTERVAL
        resume_info = {
            "resumed": True,
            "resumed_from": start_index,
            "prior_gen_seconds": prior_gen_seconds,
        }
        print(
            f"[resume] picked up checkpoint: {start_index}/{num_samples} samples done, "
            f"prior gen_seconds={prior_gen_seconds:.2f}s. Continuing..."
        )
    else:
        if os.path.exists(checkpoint_json):
            os.remove(checkpoint_json)
        if os.path.exists(checkpoint_jsonl):
            os.remove(checkpoint_jsonl)
        if os.path.exists(stage_trace_jsonl):
            os.remove(stage_trace_jsonl)
        generated_summaries = []
        all_sample_logs = []
        start_index = 0
        prior_gen_seconds = 0.0
        last_checkpoint_written = 0
        next_checkpoint = CHECKPOINT_INTERVAL
        resume_info = {"resumed": False, "resumed_from": 0, "prior_gen_seconds": 0.0}

    print(f"Start generating summaries ({num_samples} samples)...\n")

    def _process_sample(idx: int, article: str, candidates, reuse_row: dict | None = None) -> None:
        nonlocal last_checkpoint_written, next_checkpoint

        if is_any_baseline:
            final_summary, selected_indices, sample_log = generate_baseline_summary(
                candidates,
                tokenizer,
                segmenter,
                True,
                label="Baseline",
            )
        else:
            final_summary, selected_indices, sample_log = generate_sentence_level_summary(
                candidates,
                tokenizer,
                segmenter,
                article,
                method,
                objective_desc,
                use_minicheck_utility,
                use_rouge_redundancy,
                minicheck_utility_scorer,
                selector_fn,
                device,
                tri_metric_weights=tri_metric_weights,
                redundancy_threshold_override=getattr(args, "redundancy_threshold_override", None),
                ilp_penalty_scale=getattr(args, "ilp_penalty_scale", None),
                ordering_method=ordering_method,
            )

        sample_log["generator_backend"] = reuse_row.get("generator_backend") if reuse_row is not None else generator
        sample_log["generator_model"] = reuse_row.get("generator_model") if reuse_row is not None else model_path
        sample_log["generator_candidate_count"] = (
            int(reuse_row.get("generator_candidate_count"))
            if reuse_row is not None
            else (1 if is_generation_baseline else beam_size)
        )
        sample_log["prompt_version"] = (
            reuse_row.get("prompt_version") if reuse_row is not None else summary_generator.prompt_version
        )

        generated_summaries.append(final_summary)
        all_sample_logs.append(sample_log)
        _append_stage_trace(
            stage_trace_jsonl,
            idx,
            method,
            objective_desc,
            article,
            references[idx],
            final_summary,
            sample_log,
        )

        if idx == 0:
            print("\n[Diagnostic] Sample 1:")
            if is_any_baseline:
                print(f"  Output sentences: {len(selected_indices)}")
            else:
                print(f"  Selected indices: {selected_indices}")
            print(f"  Summary: {final_summary[:150]}...")

        if (idx + 1) >= next_checkpoint or (idx + 1) >= num_samples:
            current_cumulative_gen_seconds = prior_gen_seconds + (perf_counter() - gen_started_at)
            _write_progress_checkpoint(
                checkpoint_json,
                checkpoint_jsonl,
                method,
                objective_desc,
                data_split,
                sample_mode,
                sample_seed,
                beam_size,
                rouge_sentence_split,
                ordering_method,
                articles,
                references,
                generated_summaries,
                last_checkpoint_written,
                idx + 1,
                num_samples,
                gen_seconds_so_far=current_cumulative_gen_seconds,
                extra_config=expected_checkpoint_config,
            )
            last_checkpoint_written = idx + 1
            next_checkpoint = (idx + 1) + CHECKPOINT_INTERVAL
            print(
                f"  [checkpoint] saved {idx + 1}/{num_samples} "
                f"(gen_seconds_so_far={current_cumulative_gen_seconds:.2f})"
            )

        if (idx + 1) % 50 == 0 or idx == 0:
            if is_any_baseline:
                print(f"\n[{idx + 1}/{num_samples}] output_sentences={len(selected_indices)}")
            else:
                print(f"\n[{idx + 1}/{num_samples}] selected={len(selected_indices)}")
            print(f"  Summary: {final_summary[:120]}...")

    gen_started_at = perf_counter()
    try:
        if reuse_stage1_rows is not None:
            for idx in range(start_index, num_samples):
                _process_sample(
                    idx,
                    articles[idx],
                    reuse_stage1_rows[idx]["candidates"],
                    reuse_row=reuse_stage1_rows[idx],
                )
        else:
            generation_batch_size = runtime_profile["generation_batch_size"]
            first_batch_start = (start_index // generation_batch_size) * generation_batch_size
            for batch_start in range(first_batch_start, num_samples, generation_batch_size):
                batch_end = min(batch_start + generation_batch_size, num_samples)
                batch_articles = articles[batch_start:batch_end]
                generation_candidate_count = 1 if is_any_baseline else beam_size
                batch_candidates = summary_generator.generate_batch(
                    list(batch_articles),
                    num_candidates=generation_candidate_count,
                    baseline_mode=is_generation_baseline,
                )

                for local_offset, (article, candidates) in enumerate(zip(batch_articles, batch_candidates)):
                    idx = batch_start + local_offset
                    if idx < start_index:
                        continue
                    _process_sample(idx, article, candidates)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Progress saved to {checkpoint_json} and {checkpoint_jsonl}")
        raise

    gen_seconds_current = perf_counter() - gen_started_at
    gen_seconds = prior_gen_seconds + gen_seconds_current
    if resume_info["resumed"]:
        print(
            f"\n[stage] generation finished: cumulative={gen_seconds:.2f}s "
            f"(prior={prior_gen_seconds:.2f}s + this run={gen_seconds_current:.2f}s), "
            f"{num_samples} samples\n"
        )
    else:
        print(f"\n[stage] generation finished in {gen_seconds:.2f}s ({num_samples} samples)\n")
    co_timing_summary = _summarize_co_timing(all_sample_logs)
    if co_timing_summary:
        print(
            "[stage] CO timing: "
            f"samples={co_timing_summary['samples']} "
            f"total={co_timing_summary['total']:.2f}s "
            f"mean={co_timing_summary['mean']:.6f}s"
        )
    if minicheck_utility_scorer is not None:
        del minicheck_utility_scorer
        clear_cuda_cache()

    if minicheck_scorer is not None:
        del minicheck_scorer
        clear_cuda_cache()

    if summary_generator is not None:
        summary_generator.close()
        clear_cuda_cache()

    eval_suite = "rouge_only" if getattr(args, "rouge_only_eval", False) else "full"
    table_metric_names = getattr(args, "table_metric_names", None)
    extra_metric_names = getattr(args, "extra_metric_names", None)

    eval_checkpoint = result_file.replace("_results.txt", "_eval_partial.json")
    eval_started_at = perf_counter()
    try:
        metrics = run_all_evaluations(
            generated_summaries,
            articles,
            references,
            device,
            segmenter,
            num_samples,
            rouge_impl=rouge_impl,
            eval_suite=eval_suite,
            table_metric_names=table_metric_names,
            extra_metric_names=extra_metric_names,
            sentence_split_for_rouge=rouge_sentence_split,
            eval_batch_size=runtime_profile["eval_batch_size"],
            checkpoint_path=eval_checkpoint,
        )
    except (KeyboardInterrupt, Exception) as exc:
        eval_seconds = perf_counter() - eval_started_at
        runtime_seconds = perf_counter() - started_at
        partial_metrics = {}
        if os.path.exists(eval_checkpoint):
            try:
                with open(eval_checkpoint, "r", encoding="utf-8") as fh:
                    partial_metrics = json.load(fh)
            except Exception as load_exc:
                print(f"[eval-recovery] failed to load partial checkpoint: {load_exc!r}")
        partial_metrics.setdefault("metric_errors", {})["__interrupted__"] = repr(exc)
        partial_metrics["__partial__"] = True
        partial_metrics["__eval_seconds_at_interrupt__"] = round(eval_seconds, 2)
        partial_header = (
            f"!!! PARTIAL RESULT — eval interrupted/failed: {type(exc).__name__}: {exc} !!!\n"
            f"Eval elapsed before interrupt: {eval_seconds:.2f}s\n"
            f"Total runtime so far: {runtime_seconds:.2f}s\n"
            f"Recovered metrics from: {eval_checkpoint}\n"
            "============================================================\n"
        )
        partial_path = result_file.replace("_results.txt", "_results.partial.txt")
        try:
            save_results(partial_path, partial_metrics, generated_summaries, references, partial_header, all_sample_logs)
            print(f"[eval-recovery] partial result saved to: {partial_path}")
        except Exception as save_exc:
            print(f"[eval-recovery] partial save failed: {save_exc!r}")
        raise
    eval_seconds = perf_counter() - eval_started_at
    runtime_seconds = perf_counter() - started_at
    finished_wallclock = datetime.now()
    metrics["__eval_seconds__"] = round(eval_seconds, 2)
    if co_timing_summary:
        metrics["co_timing_seconds"] = co_timing_summary
    metrics["runtime"] = {
        "started_at": started_wallclock.isoformat(),
        "finished_at": finished_wallclock.isoformat(),
        "total_seconds": round(runtime_seconds, 2),
        "generation_seconds": round(gen_seconds, 2),
        "evaluation_seconds": round(eval_seconds, 2),
        "resumed": bool(resume_info["resumed"]),
        "resumed_from": int(resume_info["resumed_from"]),
        "prior_generation_seconds": round(resume_info["prior_gen_seconds"], 2),
    }
    if os.path.exists(eval_checkpoint):
        try:
            os.remove(eval_checkpoint)
        except OSError:
            pass

    generation_title = "gemma-4-E4B Baseline" if is_generation_baseline else f"gemma-4-E4B Candidates-{beam_size}"
    generation_count_detail = (
        "Generation mode: direct baseline\n"
        if is_generation_baseline
        else f"Candidate count: {beam_size}\n"
    )
    generation_detail = (
        f"MaxInputTokens: {int(getattr(args, 'max_input_tokens', 8192))}\n"
        f"MaxNewTokens: {int(getattr(args, 'max_new_tokens', MAX_GENERATION_NEW_TOKENS))}\n"
        f"Temperature: {float(getattr(args, 'temperature', 0.7))}\n"
        f"TopP: {float(getattr(args, 'top_p', 0.8))}\n"
        f"TopK: {int(getattr(args, 'top_k', 20))}\n"
        f"MinP: {float(getattr(args, 'min_p', GENERATION_MIN_P))}\n"
        f"PresencePenalty: {float(getattr(args, 'presence_penalty', GENERATION_PRESENCE_PENALTY))}\n"
        f"RepetitionPenalty: {float(getattr(args, 'repetition_penalty', 1.0))}\n"
        f"Thinking: {bool(getattr(args, 'enable_thinking', False))}\n"
        f"DoSample: {getattr(args, 'do_sample', GENERATION_DO_SAMPLE)}\n"
        f"PromptVersion: {prompt_version_desc}\n"
    )

    config_header = (
        f"{generation_title} + {method.upper()} Summarization Results (Generate-then-Optimize)\n"
        f"Generator: {generator}\n"
        f"Method: {method}\n"
        f"Objective variant: {objective_desc}\n"
        f"Model: {generator_model_desc}\n"
        f"Dataset: {dataset_name}\n"
        f"Split: {data_split}\n"
        f"Samples: {num_samples}\n"
        f"Sample mode: {sample_mode}\n"
        f"Sample seed: {sample_seed}\n"
        f"{generation_count_detail}"
        f"Budget: {budget_desc}\n"
        f"Runtime generation batch size: {runtime_profile['generation_batch_size']}\n"
        f"Runtime utility batch size: {runtime_profile['utility_batch_size']}\n"
        f"Runtime eval batch size: {runtime_profile['eval_batch_size']}\n"
        f"Runtime dtype: {compute_dtype_desc}\n"
        f"ROUGE sentence split: {rouge_sentence_split}\n"
        f"Ordering: {ordering_desc}\n"
        f"Stage output trace: {stage_trace_jsonl}\n"
        f"Stage1 reuse source: {resolved_reuse_stage1_from}\n"
        f"Started at: {started_wallclock.isoformat()}\n"
        f"Finished at: {finished_wallclock.isoformat()}\n"
        f"Runtime total seconds (this process): {runtime_seconds:.2f}\n"
        f"Stage seconds: generation={gen_seconds:.2f} (cumulative across resumes), "
        f"evaluation={eval_seconds:.2f}\n"
        f"Resume: resumed={resume_info['resumed']}, resumed_from={resume_info['resumed_from']}, "
        f"prior_gen_seconds={resume_info['prior_gen_seconds']:.2f}\n"
        f"Eval suite: {eval_suite}\n"
        f"ROUGE implementation: {rouge_impl_desc}\n"
        f"{generation_detail}"
    )
    if tri_metric_weights is not None:
        config_header += f"Tri-metric weights: {_format_tri_metric_weights(method, tri_metric_weights, precision=6)}\n"
    if table_metric_names is not None or extra_metric_names is not None:
        config_header += f"Table Metrics: {table_metric_names or ['rouge']}\n"
        config_header += f"Extra metrics: {extra_metric_names or []}\n"

    save_results(result_file, metrics, generated_summaries, references, config_header, all_sample_logs)
    if os.path.exists(checkpoint_json):
        os.remove(checkpoint_json)
    if os.path.exists(checkpoint_jsonl):
        os.remove(checkpoint_jsonl)
    print(f"Saved results to {result_file}")
    print("Done.")
    if getattr(args, "return_payload", False):
        return {
            "result_file": result_file,
            "metrics": metrics,
            "runtime_seconds": runtime_seconds,
            "num_samples": num_samples,
            "objective_name": objective_name,
            "objective_desc": objective_desc,
            "tri_metric_weights": tri_metric_weights,
        }
    return result_file
