#!/usr/bin/env python3
"""Run blind pairwise LLM-as-judge evaluation for baseline-vs-CO summaries."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict summarization evaluator.
Judge which summary is better for the source article.
Evaluate factual consistency with the source, coverage/relevance, clarity, conciseness, and non-redundancy.
Return only valid JSON. Do not include markdown."""


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def build_prompt(row: dict[str, Any], a_text: str, b_text: str, max_source_words: int) -> str:
    source = truncate_words(row["source_text"], max_source_words)
    reference = row.get("reference_summary") or ""
    return f"""Source article:
{source}

Reference summary:
{reference}

Summary A:
{a_text}

Summary B:
{b_text}

Choose the better summary. Use this JSON schema exactly:
{{
  "preference": "A_better" | "B_better" | "same" | "not_sure",
  "A_consistency": 1-5,
  "A_relevance": 1-5,
  "A_clarity": 1-5,
  "A_conciseness": 1-5,
  "B_consistency": 1-5,
  "B_relevance": 1-5,
  "B_clarity": 1-5,
  "B_conciseness": 1-5,
  "reason": "one short sentence"
}}"""


def parse_json(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned), ""
    except Exception as exc:
        return None, str(exc)


def normalize_score(value: Any) -> int | None:
    try:
        score = int(value)
    except Exception:
        return None
    if 1 <= score <= 5:
        return score
    return None


def normalize_result(parsed: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if not parsed:
        return {"parse_error": error, "preference": "not_sure"}
    preference = parsed.get("preference")
    if preference not in {"A_better", "B_better", "same", "not_sure"}:
        preference = "not_sure"
    out = {"parse_error": "", "preference": preference, "reason": str(parsed.get("reason", ""))[:500]}
    for side in ["A", "B"]:
        for dim in ["consistency", "relevance", "clarity", "conciseness"]:
            out[f"{side}_{dim}"] = normalize_score(parsed.get(f"{side}_{dim}"))
    return out


def decode_preference(preference: str, a_source: str, b_source: str) -> str:
    if preference == "A_better":
        return a_source
    if preference == "B_better":
        return b_source
    return preference


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "eval_id", "task", "sample_id", "summary_A_source", "summary_B_source",
        "preference", "decoded_preference", "A_consistency", "A_relevance", "A_clarity", "A_conciseness",
        "B_consistency", "B_relevance", "B_clarity", "B_conciseness", "parse_error", "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary(path: Path, rows: list[dict[str, Any]], model: str) -> None:
    overall = Counter(row["decoded_preference"] for row in rows)
    by_task: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_task[row["task"]][row["decoded_preference"]] += 1
    summary = {
        "judge_model": model,
        "total_rows": len(rows),
        "overall_decoded_preference": dict(overall),
        "by_task_decoded_preference": {task: dict(counter) for task, counter in by_task.items()},
        "parse_error_count": sum(bool(row.get("parse_error")) for row in rows),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def run_with_vllm(prompts: list[list[dict[str, str]]], args: argparse.Namespace) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = llm.chat(prompts, sampling_params=sampling)
    return [output.outputs[0].text for output in outputs]


def run_with_transformers(prompts: list[list[dict[str, str]]], args: argparse.Namespace) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    generated = []
    for idx, messages in enumerate(prompts, start=1):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_tokens = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated.append(tokenizer.decode(output_ids[0][input_tokens:], skip_special_tokens=True))
        if idx % args.progress_every == 0 or idx == len(prompts):
            print(f"judged {idx}/{len(prompts)}", flush=True)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_source_words", type=int, default=900)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.pairs_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_rows:
        rows = rows[: args.max_rows]

    rng = random.Random(args.seed)
    prompts = []
    payloads = []
    for row in rows:
        baseline_first = bool(rng.getrandbits(1))
        if baseline_first:
            a_text, b_text = row["baseline_summary"], row["co_summary"]
            a_source, b_source = "baseline", "co"
        else:
            a_text, b_text = row["co_summary"], row["baseline_summary"]
            a_source, b_source = "co", "baseline"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(row, a_text, b_text, args.max_source_words)},
        ]
        prompts.append(messages)
        payloads.append((row, a_source, b_source))

    if args.backend == "vllm":
        outputs = run_with_vllm(prompts, args)
    else:
        outputs = run_with_transformers(prompts, args)

    judged = []
    for raw, (row, a_source, b_source) in zip(outputs, payloads):
        parsed, error = parse_json(raw)
        result = normalize_result(parsed, error)
        judged.append({
            "eval_id": row["eval_id"],
            "task": row["task"],
            "sample_id": row["sample_id"],
            "summary_A_source": a_source,
            "summary_B_source": b_source,
            "raw_judge_output": raw,
            **result,
            "decoded_preference": decode_preference(result["preference"], a_source, b_source),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "machine_judge_results.jsonl", judged)
    write_csv(out_dir / "machine_judge_results.csv", judged)
    write_summary(out_dir / "machine_judge_summary.json", judged, args.model)
    print((out_dir / "machine_judge_summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
