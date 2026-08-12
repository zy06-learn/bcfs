#!/usr/bin/env python3
"""Validate the machine-readable artifacts transcribed from arXiv v1."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "results" / "paper"
MANIFEST = PAPER_DIR / "artifact_manifest.sha256"

EXPECTED_HEADERS = {
    "arxiv_v1_main_results.csv": [
        "dataset", "method", "category", "rouge1", "rouge2", "rougeL",
        "rougeLsum", "bertscore_f1", "factcc", "minicheck", "alignscore",
        "factkb", "faithlens", "paired_against",
    ],
    "arxiv_v1_faithfulness_benchmarks.csv": [
        "dataset", "method", "category", "factcc", "minicheck", "alignscore",
        "factkb", "faithlens",
    ],
    "arxiv_v1_selector_ablation.csv": [
        "dataset", "method", "rouge1", "rouge2", "rougeL", "rougeLsum",
        "bertscore_f1", "factcc", "minicheck", "alignscore", "factkb",
        "faithlens",
    ],
    "arxiv_v1_human_evaluation.csv": [
        "section", "group", "dpp", "direct", "same", "unsure", "delta",
        "scale", "n",
    ],
    "arxiv_v1_weight_sensitivity.csv": [
        "dataset", "model", "profile", "w_coverage", "w_factuality",
        "w_redundancy", "rougeLsum", "factcc", "minicheck", "alignscore",
        "factkb", "faithlens", "beam",
    ],
    "arxiv_v1_budget_analysis.csv": [
        "dataset", "backbone", "budget", "method", "avg_words",
        "exact_budget_hit_pct", "rougeLsum", "bertscore_f1", "factcc",
        "minicheck", "alignscore", "factkb", "faithlens_fully_faithful",
    ],
    "arxiv_v1_candidate_pool.csv": [
        "dataset", "backbone", "beam", "rougeLsum", "factcc", "minicheck",
        "alignscore", "factkb", "faithlens",
    ],
    "arxiv_v1_significance.csv": [
        "dataset", "n", "backbone", "metric", "delta", "ci95_low",
        "ci95_high", "p_holm", "significance",
    ],
}

EXPECTED_ROWS = {
    "arxiv_v1_main_results.csv": 21,
    "arxiv_v1_faithfulness_benchmarks.csv": 20,
    "arxiv_v1_selector_ablation.csv": 12,
    "arxiv_v1_human_evaluation.csv": 8,
    "arxiv_v1_weight_sensitivity.csv": 16,
    "arxiv_v1_budget_analysis.csv": 16,
    "arxiv_v1_candidate_pool.csv": 12,
    "arxiv_v1_significance.csv": 40,
}

PERCENT_COLUMNS = {
    "rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1", "factcc",
    "minicheck", "alignscore", "factkb", "faithlens",
    "faithlens_fully_faithful", "exact_budget_hit_pct",
}


def fail(message: str) -> None:
    print(f"[paper-artifacts] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_csv(name: str) -> list[dict[str, str]]:
    path = PAPER_DIR / name
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_HEADERS[name]:
            fail(f"unexpected header in {name}: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != EXPECTED_ROWS[name]:
        fail(f"unexpected row count in {name}: {len(rows)}")
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            fail(f"extra column in {name}:{line_number}")
        for field in PERCENT_COLUMNS & row.keys():
            if row[field] == "":
                continue
            try:
                value = float(row[field])
            except ValueError:
                fail(f"non-numeric {field} in {name}:{line_number}")
            if not 0 <= value <= 100:
                fail(f"out-of-range {field} in {name}:{line_number}: {value}")
    return rows


def find(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    found = [row for row in rows if all(row.get(key) == value for key, value in match.items())]
    if len(found) != 1:
        fail(f"expected one row matching {match}, found {len(found)}")
    return found[0]


def validate_manifest() -> None:
    entries: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
        match = pattern.fullmatch(line)
        if not match:
            fail(f"invalid manifest line {line_number}")
        expected, relative = match.groups()
        if relative in entries:
            fail(f"duplicate manifest entry: {relative}")
        path = ROOT / relative
        if not path.is_file():
            fail(f"manifest target missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"checksum mismatch for {relative}: expected {expected}, got {actual}")
        entries[relative] = actual

    expected_paths = {f"results/paper/{name}" for name in EXPECTED_HEADERS}
    expected_paths.add("assets/pipeline.png")
    if set(entries) != expected_paths:
        fail("manifest targets do not exactly match the release artifact set")


def main() -> None:
    loaded = {name: read_csv(name) for name in EXPECTED_HEADERS}

    main_rows = loaded["arxiv_v1_main_results.csv"]
    if find(main_rows, dataset="CNN/DailyMail", method="BART+DPP")["minicheck"] != "97.73":
        fail("CNN/DailyMail BART+DPP MiniCheck anchor does not match arXiv v1")
    if find(main_rows, dataset="Multi-News", method="PRIMERA+DPP")["alignscore"] != "71.27":
        fail("Multi-News PRIMERA+DPP AlignScore anchor does not match arXiv v1")

    human = loaded["arxiv_v1_human_evaluation.csv"]
    overall = find(human, section="preference", group="Overall")
    preference_total = sum(int(overall[field]) for field in ("dpp", "direct", "same", "unsure"))
    if preference_total != int(overall["n"]):
        fail("overall human preference counts do not sum to n")
    if any(row["n"] for row in human if row["section"] == "rating"):
        fail("rating rows must not infer per-dimension n absent from arXiv v1")

    weights = loaded["arxiv_v1_weight_sensitivity.csv"]
    for row in weights:
        total = sum(float(row[field]) for field in ("w_coverage", "w_factuality", "w_redundancy"))
        if abs(total - 1.0) > 1e-9 or row["beam"] != "8":
            fail(f"invalid weight profile: {row}")

    budgets = loaded["arxiv_v1_budget_analysis.csv"]
    dpp_rows = [row for row in budgets if row["method"] == "+DPP"]
    if any(row["exact_budget_hit_pct"] != "100.00" for row in dpp_rows):
        fail("DPP exact-budget invariant does not match arXiv v1")

    significance = loaded["arxiv_v1_significance.csv"]
    if sum(row["significance"] == "n.s." for row in significance) != 2:
        fail("expected exactly two non-significant paired comparisons")

    validate_manifest()
    print(
        f"[paper-artifacts] verified {sum(len(rows) for rows in loaded.values())} "
        f"rows across {len(loaded)} CSV files"
    )


if __name__ == "__main__":
    main()
