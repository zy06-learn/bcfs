#!/usr/bin/env python3
"""Build baseline-vs-CO pairs for machine judging.

This reads existing stage output traces only. It does not run summarizers.
Run it on dgxspark where the full stage output files are local.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


TASKS = {
    "bart_cnn": {
        "baseline": "1_bart/results/bart/cnn_dailymail/baseline_baseline/beam12_baseline_hfrouge_shuffle_seed42_stage_outputs.jsonl",
        "co": "outputs/cnn_dailymail/bart/dpp/full_cnn_dailymail_co_tri_metric_beam12_test_optimal_20260612/beam12_dpp_tri_metric_hfrouge_shuffle_seed42_stage_outputs.jsonl",
        "co_selector": "dpp",
        "co_beam": 12,
    },
    "llama_cnn": {
        "baseline": "outputs/cnn_dailymail/llama3_8b/stage1/full_cnn_dailymail_beam12_stage1_beam12_stage1_test_faith_best_20260614/beam12_baseline_hfrouge_shuffle_seed42_stage_outputs.jsonl",
        "co": "outputs/cnn_dailymail/llama3_8b/ilp/full_cnn_dailymail_co_tri_metric_beam12_test_faith_best_20260618/beam12_ilp_tri_metric_hfrouge_shuffle_seed42_stage_outputs.jsonl",
        "co_selector": "ilp",
        "co_beam": 12,
    },
}


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def source_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compact_row(row: dict[str, Any], summary_kind: str) -> dict[str, Any] | None:
    stage1 = row.get("stage1_llm_summary_generation") or {}
    stage3 = row.get("stage3_ordering_and_final_realization") or {}
    source = first_present(row.get("source_document"), row.get("article"), row.get("document"), row.get("source_text"))
    if summary_kind == "baseline":
        summary = first_present(stage1.get("generated_summary"), stage3.get("final_summary"), row.get("generated_summary"), row.get("summary"))
    else:
        summary = first_present(stage3.get("final_summary"), row.get("final_summary"), row.get("summary"))
    if not source or not summary:
        return None
    sample_index = row.get("sample_index", row.get("index", row.get("id")))
    return {
        "raw_sample_id": str(sample_index) if sample_index is not None else None,
        "source_text": source,
        "reference_summary": first_present(row.get("reference_summary"), row.get("gold_summary"), row.get("reference"), row.get("highlights")),
        "summary": summary,
        "source_hash": source_hash(source),
    }


def load_rows(path: Path, summary_kind: str) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        compact = compact_row(row, summary_kind)
        if compact:
            rows.append(compact)
    return rows


def index_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id = {}
    by_hash = {}
    for row in rows:
        if row.get("raw_sample_id") is not None:
            by_id[str(row["raw_sample_id"])] = row
        by_hash[row["source_hash"]] = row
    return by_id, by_hash


def build_pairs(root: Path, task: str, cfg: dict[str, Any], n: int, rng: random.Random) -> tuple[list[dict[str, Any]], int]:
    baseline_path = root / cfg["baseline"]
    co_path = root / cfg["co"]
    baseline_rows = load_rows(baseline_path, "baseline")
    co_rows = load_rows(co_path, "co")
    baseline_by_id, baseline_by_hash = index_rows(baseline_rows)
    pairs = []
    for co_row in co_rows:
        baseline_row = baseline_by_id.get(str(co_row.get("raw_sample_id")))
        if baseline_row is None:
            baseline_row = baseline_by_hash.get(co_row["source_hash"])
        if baseline_row is None:
            continue
        sample_token = str(co_row.get("raw_sample_id") or co_row["source_hash"])
        pairs.append({
            "task": task,
            "raw_sample_id": sample_token,
            "sample_id": f"{task}_{sample_token}",
            "source_text": co_row["source_text"],
            "reference_summary": co_row.get("reference_summary") or baseline_row.get("reference_summary"),
            "baseline_summary": baseline_row["summary"],
            "co_summary": co_row["summary"],
            "baseline_file": str(baseline_path),
            "co_file": str(co_path),
            "co_selector": cfg["co_selector"],
            "co_beam": cfg["co_beam"],
        })
    total = len(pairs)
    rng.shuffle(pairs)
    return pairs[:n], total


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "eval_id", "task", "sample_id", "source_text", "reference_summary",
        "baseline_summary", "co_summary", "baseline_file", "co_file", "co_selector", "co_beam",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Experiment repository root (default: current working directory).",
    )
    parser.add_argument("--out_dir", default="outputs/machine_judge_500_pairs")
    parser.add_argument("--n_per_task", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    matched_counts = {}
    for task, cfg in TASKS.items():
        task_rows, total = build_pairs(root, task, cfg, args.n_per_task, rng)
        matched_counts[task] = total
        for idx, row in enumerate(task_rows, 1):
            row["eval_id"] = f"{task}_{idx:04d}"
            rows.append(row)
    rows.sort(key=lambda row: (row["task"], row["eval_id"]))
    write_jsonl(out_dir / "machine_judge_500_pairs.jsonl", rows)
    write_csv(out_dir / "machine_judge_500_pairs.csv", rows)

    report = {
        "total_pairs": len(rows),
        "expected_pairs": len(TASKS) * args.n_per_task,
        "n_per_task": args.n_per_task,
        "matched_counts": matched_counts,
        "task_counts": {task: sum(row["task"] == task for row in rows) for task in TASKS},
        "identical_summary_count": sum(row["baseline_summary"] == row["co_summary"] for row in rows),
        "source_empty_count": sum(not row["source_text"] for row in rows),
        "summary_empty_count": sum((not row["baseline_summary"]) or (not row["co_summary"]) for row in rows),
    }
    (out_dir / "PAIR_QUALITY_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
