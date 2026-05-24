"""CLI argument parsing for BART-large-CNN optimization experiments."""

import argparse

from core.config import (
    BEAM_SIZE,
    BUDGET_SENTENCES,
    DEFAULT_GENERATOR,
    DEFAULT_ROUGE_IMPL,
    EVAL_BATCH_SIZE,
    GENERATION_BATCH_SIZE,
    GENERATOR_NAME,
    NUM_SAMPLES,
    MAX_INPUT_TOKENS,
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
    parser = argparse.ArgumentParser(description="BART-large-CNN generate-then-optimize summarization experiments")
    parser.add_argument(
        "--method",
        required=True,
        choices=ALL_METHODS,
        help="Optimization method to run for this BART-large-CNN experiment directory.",
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
        help=f"BART beam size for baseline/CO candidate generation (default: {BEAM_SIZE}).",
    )
    parser.add_argument(
        "--generator",
        choices=GENERATOR_CHOICES,
        default=DEFAULT_GENERATOR,
        help="Generation backend for this model directory.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=MAX_INPUT_TOKENS,
        help=f"Tokenizer truncation length for the BART encoder (default: {MAX_INPUT_TOKENS}).",
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
        choices=["cnn_dailymail"],
        help="Summarization dataset to use. This model directory only supports cnn_dailymail.",
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
        "--compute-dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="fp32",
        help="Inference dtype for the generation stage (default: fp32, matching Hugging Face defaults more closely).",
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
    args = parser.parse_args()
    args.rouge_only_eval = False
    args.table_metric_names = list(FIXED_table_metric_names)
    args.extra_metric_names = list(FIXED_EXTRA_METRIC_NAMES)
    return args
