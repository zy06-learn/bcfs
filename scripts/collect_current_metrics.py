#!/usr/bin/env python3
"""Collect compact metrics for the currently selected result rows."""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "raw"
DEFAULT_SELECTED_ROWS = ROOT / "results" / "tables" / "selected_rows.csv"
DEFAULT_OUTPUT_CSV = ROOT / "results" / "tables" / "current_metrics.csv"

HEADER_KEYS = [
    "Generator",
    "Method",
    "Objective variant",
    "Model",
    "Dataset",
    "Split",
    "Samples",
    "Sample mode",
    "Sample seed",
    "Beam size",
    "Candidate count",
    "Budget",
    "Runtime dtype",
    "ROUGE implementation",
    "ROUGE sentence split",
    "Ordering",
    "Tri-metric weights",
    "Stage1 reuse source",
]

METRIC_PATTERNS = {
    "rouge1": re.compile(r"^\s*rouge1\s+F1=([-\d.]+)%"),
    "rouge2": re.compile(r"^\s*rouge2\s+F1=([-\d.]+)%"),
    "rougeL": re.compile(r"^\s*rougeL\s+F1=([-\d.]+)%"),
    "rougeLsum": re.compile(r"^\s*rougeLsum\s+F1=([-\d.]+)%"),
    "bertscore_f1": re.compile(r"^\s*Precision=[-\d.]+%\s+Recall=[-\d.]+%\s+F1=([-\d.]+)%"),
    "factcc": re.compile(r"^\s*SentenceAvgCorrect:\s*([-\d.]+)%"),
    "minicheck": re.compile(r"^\s*SummaryAvgConsistent:\s*([-\d.]+)%"),
    "alignscore": re.compile(r"^\s*SummaryAvg:\s*([-\d.]+)%"),
    "factkb": re.compile(r"^\s*FactualProb:\s*([-\d.]+)%"),
}

UNAVAILABLE_PATTERNS = {
    "bertscore_status": re.compile(r"^BERTScore:\s*unavailable\s*(.*)$"),
    "factcc_status": re.compile(r"^FactCC:\s*unavailable\s*(.*)$"),
    "minicheck_status": re.compile(r"^MiniCheck:\s*unavailable\s*(.*)$"),
    "alignscore_status": re.compile(r"^AlignScore:\s*unavailable\s*(.*)$"),
    "factkb_status": re.compile(r"^FactKB:\s*unavailable\s*(.*)$"),
}

FIELDNAMES = [
    "dataset",
    "generator",
    "method",
    "model",
    "samples",
    "sample_mode",
    "sample_seed",
    "candidate_count",
    "beam_size",
    "budget",
    "tri_metric_weights",
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "bertscore_f1",
    "bertscore_status",
    "factcc",
    "factcc_status",
    "minicheck",
    "minicheck_status",
    "alignscore",
    "alignscore_status",
    "factkb",
    "factkb_status",
    "ordering",
    "stage1_reuse_source",
    "selection_group",
    "source_result",
]


def parse_result(path: Path, root: Path) -> dict[str, str]:
    row: dict[str, str] = {field: "" for field in FIELDNAMES}
    row["source_result"] = path.relative_to(root).as_posix()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for key in HEADER_KEYS:
            prefix = f"{key}: "
            if line.startswith(prefix):
                normalized = key.lower().replace(" ", "_").replace("-", "_")
                if normalized == "objective_variant":
                    normalized = "objective"
                if normalized in row:
                    row[normalized] = line[len(prefix):].strip()

        for metric_name, pattern in METRIC_PATTERNS.items():
            match = pattern.search(line)
            if match and not row[metric_name]:
                row[metric_name] = match.group(1)

        for status_name, pattern in UNAVAILABLE_PATTERNS.items():
            match = pattern.search(line)
            if match and not row[status_name]:
                detail = match.group(1).strip()
                row[status_name] = f"unavailable {detail}".strip()

    return row


def infer_generator(row: dict[str, str]) -> str:
    generator = row.get("generator", "").strip()
    if generator:
        return generator

    model = row.get("model", "")
    source = row.get("source_result", "")
    if "bart-large-cnn" in model or "/bart/" in source:
        return "bart"
    if "Qwen3.5-9B" in model:
        return "qwen3.5_9B"
    if "Llama-3.1-8B" in model:
        return "llama3_8b"
    if "gemma-4-E4B" in model:
        return "gemma_4_e4b"
    return generator


def normalize_method(method: str) -> str:
    if method == "baseline_raw":
        return "baseline"
    return method


def normalize_stage1_source(path: str) -> str:
    path = (path or "").strip()
    if path in {"", "None", "none", "null"}:
        return ""

    outputs_marker = "outputs/"
    if outputs_marker in path:
        return path[path.index(outputs_marker):]
    return path


def normalize_row(row: dict[str, str]) -> None:
    row["dataset"] = row.get("dataset", "").split(" ", 1)[0].strip()
    row["generator"] = infer_generator(row)
    row["method"] = normalize_method(row.get("method", ""))
    row["stage1_reuse_source"] = normalize_stage1_source(row.get("stage1_reuse_source", ""))
    if not row.get("candidate_count") and row.get("beam_size"):
        row["candidate_count"] = row["beam_size"]


def load_selected(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_csv(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--selected-rows", type=Path, default=DEFAULT_SELECTED_ROWS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that --output-csv is current without modifying it.",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    selected_rows = load_selected(args.selected_rows)
    rows: list[dict[str, str]] = []
    missing: list[str] = []

    for selected in selected_rows:
        rel = selected["source_result"]
        path = results_root / rel
        if not path.exists():
            missing.append(rel)
            continue
        row = parse_result(path, results_root)
        normalize_row(row)
        for key in ("dataset", "generator", "method"):
            if selected.get(key):
                row[key] = selected[key]
        row["selection_group"] = selected.get("selection_group", "")
        rows.append(row)

    rows.sort(key=lambda row: (row["dataset"], row["generator"], row["method"], row["source_result"]))
    if missing:
        print("[collect] missing selected result files:")
        for rel in missing:
            print(f"  {rel}")
        raise SystemExit(1)

    rendered = render_csv(rows)
    if args.check:
        if not args.output_csv.exists():
            print(f"[collect] missing generated table: {args.output_csv}", file=sys.stderr)
            raise SystemExit(1)
        current = args.output_csv.read_text(encoding="utf-8")
        if current != rendered:
            print(
                f"[collect] stale generated table: {args.output_csv}; "
                "run without --check to update it",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"[collect] verified {len(rows)} rows in {args.output_csv}")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.write_text(rendered, encoding="utf-8", newline="")
    print(f"[collect] wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
