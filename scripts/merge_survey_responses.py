#!/usr/bin/env python3
"""Merge exported survey CSV files and decode blind A/B labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_key(path: Path) -> dict[str, dict[str, str]]:
    key: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key[row["eval_id"]] = row
    return key


def decode_preference(preference: str, key_row: dict[str, str]) -> str:
    if preference == "A_better":
        return key_row["summary_A_source"]
    if preference == "B_better":
        return key_row["summary_B_source"]
    if preference == "same":
        return "same"
    if preference == "not_sure":
        return "not_sure"
    return ""


def swap_source(source: str) -> str:
    if source == "baseline":
        return "co"
    if source == "co":
        return "baseline"
    return source


def source_for_side(key_row: dict[str, str], side: str, swap_sources: bool) -> str:
    source = key_row[f"summary_{side}_source"]
    return swap_source(source) if swap_sources else source


def preferred_summary_label(preference: str) -> str:
    if preference == "A_better":
        return "summary_A"
    if preference == "B_better":
        return "summary_B"
    if preference == "same":
        return "same"
    if preference == "not_sure":
        return "not_sure"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_dir")
    parser.add_argument("--responses_csv")
    parser.add_argument("--blind_key", default="outputs/human_eval_pairs/human_eval_100_pairs_blind_key.jsonl")
    parser.add_argument("--out_csv", default="outputs/human_eval_pairs/merged_survey_responses.csv")
    parser.add_argument(
        "--swap_sources",
        action="store_true",
        help="Decode A/B with baseline and CO labels swapped, preserving raw A/B responses.",
    )
    args = parser.parse_args()
    if not args.responses_dir and not args.responses_csv:
        parser.error("one of --responses_dir or --responses_csv is required")

    key = load_key(Path(args.blind_key))
    response_files: list[Path] = []
    if args.responses_dir:
        response_files.extend(sorted(Path(args.responses_dir).glob("*.csv")))
    if args.responses_csv:
        response_files.append(Path(args.responses_csv))
    rows: list[dict[str, str]] = []
    for path in response_files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                eval_id = row.get("eval_id", "")
                key_row = key.get(eval_id)
                if not key_row:
                    row["decoded_preference"] = ""
                    row["decode_error"] = "missing_eval_id_in_key"
                else:
                    decoded_key = {
                        **key_row,
                        "summary_A_source": source_for_side(key_row, "A", args.swap_sources),
                        "summary_B_source": source_for_side(key_row, "B", args.swap_sources),
                    }
                    row["summary_A_source"] = decoded_key["summary_A_source"]
                    row["summary_B_source"] = decoded_key["summary_B_source"]
                    row["summary_A_owner"] = decoded_key["summary_A_source"]
                    row["summary_B_owner"] = decoded_key["summary_B_source"]
                    row["preferred_summary_label"] = preferred_summary_label(row.get("preference", ""))
                    row["preferred_summary_owner"] = decode_preference(row.get("preference", ""), decoded_key)
                    row["decoded_preference"] = decode_preference(row.get("preference", ""), decoded_key)
                    row["label_correction"] = "baseline_co_swapped" if args.swap_sources else ""
                    row["decode_error"] = ""
                row["response_file"] = str(path)
                rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "response_file",
        "annotator_id",
        "eval_id",
        "task",
        "sample_id",
        "preference",
        "summary_A_source",
        "summary_B_source",
        "summary_A_owner",
        "summary_B_owner",
        "preferred_summary_label",
        "preferred_summary_owner",
        "decoded_preference",
        "label_correction",
        "A_consistency",
        "A_currency",
        "A_relevance",
        "A_clarity",
        "A_conciseness",
        "B_consistency",
        "B_currency",
        "B_relevance",
        "B_clarity",
        "B_conciseness",
        "comment",
        "updated_at",
        "submitted_at",
        "decode_error",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"merged_files={len(response_files)} rows={len(rows)} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
