"""CLI argument parsing for model optimization experiments."""

import argparse

from core.config import (
    BEAM_SIZE,
    BUDGET_SENTENCES,
    DEFAULT_ROUGE_IMPL,
    EVAL_BATCH_SIZE,
    GENERATION_BATCH_SIZE,
    NUM_SAMPLES,
    GENERATOR_NAME,
    MAX_GENERATION_INPUT_TOKENS,
    MAX_GENERATION_NEW_TOKENS,
    GENERATION_MODEL_PATH,
    GENERATION_DO_SAMPLE,
    GENERATION_MIN_P,
    GENERATION_PRESENCE_PENALTY,
    GENERATION_REPETITION_PENALTY,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_K,
    GENERATION_TOP_P,
    TRI_MINICHECK_WEIGHT,
    TRI_REDUNDANCY_WEIGHT,
    TRI_ROUGE_WEIGHT,
    UTILITY_BATCH_SIZE,
)


ALL_METHODS = ["ilp", "mmr", "dpp", "baseline"]
OBJECTIVE_CHOICES = ["rouge_only", "rouge_redundancy", "minicheck_only", "minicheck_redundancy"]
GENERATOR_CHOICES = [GENERATOR_NAME]
FIXED_table_metric_names = ["rouge", "bertscore"]
FIXED_EXTRA_METRIC_NAMES = ["factcc", "minicheck", "alignscore", "factkb"]


def parse_args():
    """Parse command-line arguments for the main experiment runner."""
    parser = argparse.ArgumentParser(description="gemma-4-E4B generate-then-optimize summarization experiments")
    parser.add_argument(
        "--method",
        required=True,
        choices=ALL_METHODS,
        help="Optimization method to run for this gemma-4-E4B-it experiment directory.",
    )
    parser.add_argument(
        "--objective",
        default=None,
        choices=OBJECTIVE_CHOICES,
        help="Objective variant for sentence-level methods",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Number of samples to evaluate (default: {NUM_SAMPLES})",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["head", "shuffle"],
        default="shuffle",
        help="How to choose the subset when --num-samples > 0 (default: shuffle for reproducible random sampling).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when --sample-mode=shuffle (default: 42).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=BEAM_SIZE,
        help=(
            f"Candidate count for CO runs (default: {BEAM_SIZE}). "
            "Ignored by baseline, which uses direct generation."
        ),
    )
    parser.add_argument(
        "--generator",
        choices=GENERATOR_CHOICES,
        default=GENERATOR_NAME,
        help="Generation backend for this model directory.",
    )
    parser.add_argument(
        "--model-path",
        default=GENERATION_MODEL_PATH,
        help=f"Model id/path used when --generator={GENERATOR_NAME}.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=MAX_GENERATION_INPUT_TOKENS,
        help=f"Tokenizer truncation length (default: {MAX_GENERATION_INPUT_TOKENS}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_GENERATION_NEW_TOKENS,
        help=f"Max generated tokens per summary candidate (default: {MAX_GENERATION_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=GENERATION_TEMPERATURE,
        help=f"Sampling temperature (default: {GENERATION_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=GENERATION_TOP_P,
        help=f"Nucleus sampling top-p (default: {GENERATION_TOP_P}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=GENERATION_TOP_K,
        help=f"Sampling top-k (default: {GENERATION_TOP_K}).",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=GENERATION_MIN_P,
        help=f"Sampling min-p (default: {GENERATION_MIN_P}).",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=GENERATION_PRESENCE_PENALTY,
        help=f"Presence penalty (default: {GENERATION_PRESENCE_PENALTY}).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=GENERATION_REPETITION_PENALTY,
        help=f"Repetition penalty (default: {GENERATION_REPETITION_PENALTY}).",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode when the tokenizer chat template supports it.",
    )
    parser.add_argument(
        "--do-sample",
        choices=["auto", "true", "false"],
        default=GENERATION_DO_SAMPLE,
        help=(
            "LLM decoding mode. true uses Sampling parameters; false uses internal beam search; "
            "auto uses model-card sampling for baseline and beam search for CO runs."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--dataset",
        default="cnn_dailymail",
        choices=["cnn_dailymail", "multi_news"],
        help="Summarization dataset to use (default: cnn_dailymail).",
    )
    parser.add_argument(
        "--use-local-rouge",
        action="store_true",
        help="Use local rouge_score averaging instead of HuggingFace evaluate ROUGE (default: HF ROUGE)",
    )
    parser.add_argument(
        "--rouge-sentence-split",
        choices=["nltk", "pysbd"],
        default="nltk",
        help=(
            "Sentence splitter used to insert newlines before ROUGE-Lsum. "
            "Default nltk matches the HuggingFace summarization example; pysbd keeps the older local protocol."
        ),
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=None,
        help=(
            "Generation batch size. If omitted, the runtime will auto-pick a value from the GPU profile "
            f"(fallback default: {GENERATION_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--utility-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for MiniCheck utility/selection scoring. If omitted, runtime auto-profile is used "
            f"(fallback default: {UTILITY_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help=(
            "Shared batch size hint for heavyweight evaluation metrics. If omitted, runtime auto-profile is used "
            f"(fallback default: {EVAL_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--budget-sentences",
        type=int,
        default=None,
        help=(
            "Sentence budget for CO selection. If omitted, the runtime uses the model config default "
            f"({BUDGET_SENTENCES})."
        ),
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="bf16",
        help="Inference dtype for the generation stage (default: bf16 for gemma-4-E4B).",
    )
    parser.add_argument(
        "--ordering",
        choices=["source_similarity", "none"],
        default="source_similarity",
        help=(
            "How to order CO-selected sentences before realization. "
            "source_similarity sorts each selected sentence by the source-document sentence "
            "with the highest ROUGE-L similarity; none keeps selector output order."
        ),
    )
    parser.add_argument(
        "--tri-metric",
        action="store_true",
        help="Enable the unified tri-metric objective (ROUGE + MiniCheck + redundancy).",
    )
    parser.add_argument(
        "--w-rouge",
        type=float,
        default=TRI_ROUGE_WEIGHT,
        help=(
            f"Tri-metric ROUGE utility weight (fallback default: {TRI_ROUGE_WEIGHT}); "
            "when all three weights remain at fallback defaults, method-specific defaults from config are used."
        ),
    )
    parser.add_argument(
        "--w-minicheck",
        type=float,
        default=TRI_MINICHECK_WEIGHT,
        help=(
            f"Tri-metric MiniCheck utility weight (fallback default: {TRI_MINICHECK_WEIGHT}); "
            "when all three weights remain at fallback defaults, method-specific defaults from config are used."
        ),
    )
    parser.add_argument(
        "--w-redundancy",
        type=float,
        default=TRI_REDUNDANCY_WEIGHT,
        help=(
            f"Tri-metric redundancy weight (fallback default: {TRI_REDUNDANCY_WEIGHT}); "
            "when all three weights remain at fallback defaults, method-specific defaults from config are used."
        ),
    )
    parser.add_argument(
        "--redundancy-threshold-override",
        type=float,
        default=None,
        help="Optional selector-level redundancy threshold override for ILP.",
    )
    parser.add_argument(
        "--ilp-penalty-scale",
        choices=["per_edge", "per_sentence", "per_pair"],
        default="per_edge",
        help=(
            "Scaling of the soft-ILP pairwise penalty alpha under tri-metric mode. "
            "per_edge: alpha = w_redundancy / (budget - 1)  [default, validated]. "
            "per_sentence: alpha = w_redundancy / budget. "
            "per_pair: alpha = w_redundancy / C(budget, 2). "
            "Only affects method=ilp under --tri-metric; the default hard-ILP path is unchanged."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable auto-resume. By default the generation loop resumes from an existing "
            "*_progress.json / *_progress_summaries.jsonl pair when config matches. "
            "Pass this flag to force a fresh run that wipes prior checkpoints."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help=(
            "How many samples to process between generation checkpoint writes. "
            "If omitted, the runtime uses the built-in default."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism in vLLM (default: 1).",
    )
    parser.add_argument(
        "--reuse-stage1-from",
        default=None,
        help=(
            "Path to an existing *_stage_outputs.jsonl file whose stage1 beam candidates should be "
            "reused for this run. In reuse mode the selector reruns on the cached candidates and "
            "skips model loading/generation."
        ),
    )
    args = parser.parse_args()
    args.rouge_only_eval = False
    args.table_metric_names = list(FIXED_table_metric_names)
    args.extra_metric_names = list(FIXED_EXTRA_METRIC_NAMES)
    return args
