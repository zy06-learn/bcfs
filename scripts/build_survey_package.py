#!/usr/bin/env python3
"""Build a human-evaluation survey package."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_blind_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_assignments(
    items: list[dict[str, str]],
    annotator_count: int,
    annotations_per_item: int,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    annotators = [f"annotator_{index:02d}" for index in range(1, annotator_count + 1)]
    assignments: dict[str, list[str]] = {annotator: [] for annotator in annotators}
    task_loads: dict[str, Counter[str]] = defaultdict(Counter)

    items_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        items_by_task[item["task"]].append(item)

    for task, task_items in sorted(items_by_task.items()):
        shuffled = list(task_items)
        rng.shuffle(shuffled)
        for item in shuffled:
            selected: list[str] = []
            for _ in range(annotations_per_item):
                candidates = [annotator for annotator in annotators if annotator not in selected]
                candidates.sort(
                    key=lambda annotator: (
                        task_loads[task][annotator],
                        len(assignments[annotator]),
                        annotator,
                    )
                )
                chosen = candidates[0]
                assignments[chosen].append(item["eval_id"])
                task_loads[task][chosen] += 1
                selected.append(chosen)

    for annotator in annotators:
        assignments[annotator].sort()
    return assignments


def validate_assignments(
    items: list[dict[str, str]],
    assignments: dict[str, list[str]],
    annotations_per_item: int,
) -> None:
    item_ids = {item["eval_id"] for item in items}
    assigned_ids = [item_id for item_ids in assignments.values() for item_id in item_ids]
    unknown = sorted(set(assigned_ids) - item_ids)
    missing = sorted(item_ids - set(assigned_ids))
    if unknown:
        raise ValueError(f"assignments contain unknown item ids: {unknown[:5]}")
    if missing:
        raise ValueError(f"assignments are missing item ids: {missing[:5]}")
    item_counts = Counter(assigned_ids)
    bad_counts = {item_id: count for item_id, count in item_counts.items() if count != annotations_per_item}
    if bad_counts:
        sample = dict(list(sorted(bad_counts.items()))[:5])
        raise ValueError(f"items do not have {annotations_per_item} annotations: {sample}")


def summarize_assignments(
    items: list[dict[str, str]],
    assignments: dict[str, list[str]],
    annotations_per_item: int,
) -> dict[str, Any]:
    task_by_id = {item["eval_id"]: item["task"] for item in items}
    item_counts = Counter(item_id for item_ids in assignments.values() for item_id in item_ids)
    annotator_rows = []
    for annotator, item_ids in sorted(assignments.items()):
        task_counts = Counter(task_by_id[item_id] for item_id in item_ids)
        annotator_rows.append(
            {
                "annotator": annotator,
                "total": len(item_ids),
                "task_counts": dict(sorted(task_counts.items())),
            }
        )
    return {
        "item_count": len(items),
        "annotator_count": len(assignments),
        "annotations_per_item": annotations_per_item,
        "total_assignments": sum(len(item_ids) for item_ids in assignments.values()),
        "min_item_annotations": min(item_counts.values()) if item_counts else 0,
        "max_item_annotations": max(item_counts.values()) if item_counts else 0,
        "annotators": annotator_rows,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    base_url = "https://zy06-learn.github.io/bcfs/"
    lines = [
        "# Human Evaluation Survey",
        "",
        "This folder contains a static browser survey for pairwise summary evaluation.",
        "",
        "## Local Test",
        "",
        "```bash",
        "cd survey",
        "python3 -m http.server 8000",
        "```",
        "",
        "Open `http://localhost:8000/?annotator=annotator_01`.",
        "",
        "## Annotator Links",
        "",
        "Send one link to each annotator. Each annotator exports a CSV after finishing.",
        "",
    ]
    for annotator in [row["annotator"] for row in summary["annotators"]]:
        lines.append(f"- `{annotator}`: `{base_url}?annotator={annotator}`")
    lines.extend(
        [
            "",
            "## Assignment Summary",
            "",
            f"- Items: {summary['item_count']}",
            f"- Annotators: {summary['annotator_count']}",
            f"- Annotations per item: {summary['annotations_per_item']}",
            f"- Total item assignments: {summary['total_assignments']}",
            f"- Min annotations per item: {summary['min_item_annotations']}",
            f"- Max annotations per item: {summary['max_item_annotations']}",
            "",
            "## Response Collection",
            "",
            "Ask every annotator to download and send back their response CSV. The public",
            "site includes only blind survey items and assignments. Keep the original blind",
            "key file private to the project owner.",
            "",
            "Put returned CSV files into a local folder, for example `survey_responses/`, then run:",
            "",
            "```bash",
            "python3 scripts/merge_survey_responses.py \\",
            "  --responses_dir survey_responses \\",
            "  --blind_key outputs/human_eval_pairs/human_eval_100_pairs_blind_key.jsonl \\",
            "  --out_csv outputs/human_eval_pairs/merged_survey_responses.csv",
            "```",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_annotator_links(out_dir: Path, summary: dict[str, Any]) -> None:
    base_url = "https://zy06-learn.github.io/bcfs/"
    fieldnames = ["annotator_id", "url", "item_count", "bart_cnn_count", "llama_cnn_count"]
    with (out_dir / "annotator_links.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["annotators"]:
            task_counts = row["task_counts"]
            writer.writerow(
                {
                    "annotator_id": row["annotator"],
                    "url": f"{base_url}?annotator={row['annotator']}",
                    "item_count": row["total"],
                    "bart_cnn_count": task_counts.get("bart_cnn", 0),
                    "llama_cnn_count": task_counts.get("llama_cnn", 0),
                }
            )


def write_worker_data(path: Path, items: list[dict[str, str]], assignments: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    item_index = {
        item["eval_id"]: {
            "task": item["task"],
            "sample_id": item["sample_id"],
        }
        for item in items
    }
    content = (
        "// Generated by scripts/build_survey_package.py. Do not edit by hand.\n"
        f"export const SURVEY_ITEMS = {json.dumps(item_index, ensure_ascii=False, indent=2)};\n\n"
        f"export const ASSIGNMENTS = {json.dumps(assignments, ensure_ascii=False, indent=2)};\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind_csv", default="outputs/human_eval_pairs/human_eval_100_pairs_blind.csv")
    parser.add_argument("--out_dir", default="survey")
    parser.add_argument("--annotators", type=int, default=20)
    parser.add_argument("--annotations_per_item", type=int, default=1)
    parser.add_argument("--worker_data", default="worker/survey_data.js")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_blind_csv(Path(args.blind_csv))
    items = [
        {
            "eval_id": row["eval_id"],
            "task": row["task"],
            "sample_id": row["sample_id"],
            "source_text": row["source_text"],
            "reference_summary": row.get("reference_summary") or "",
            "summary_A": row["summary_A"],
            "summary_B": row["summary_B"],
        }
        for row in rows
    ]
    assignments = build_assignments(items, args.annotators, args.annotations_per_item, args.seed)
    validate_assignments(items, assignments, args.annotations_per_item)
    summary = summarize_assignments(items, assignments, args.annotations_per_item)

    write_json(out_dir / "survey_items.json", items)
    write_json(out_dir / "assignments.json", assignments)
    write_json(out_dir / "assignment_summary.json", summary)
    write_readme(out_dir, summary)
    write_annotator_links(out_dir, summary)
    if args.worker_data:
        write_worker_data(Path(args.worker_data), items, assignments)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
