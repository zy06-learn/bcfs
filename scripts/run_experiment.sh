#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_experiment.sh --model MODEL --method METHOD [options] [-- extra run.py args...]

Required:
  --model   bart | primera_multinews | llama3_8b | qwen3_5_9b | gemma4_e4b
  --method  baseline | ilp | dpp | mmr

Options:
  --dataset DATASET          cnn_dailymail or multi_news
  --split SPLIT              train, validation, or test (default: test)
  --num-samples N            0 means full selected split (default: 0)
  --sample-mode MODE         shuffle or head (default: shuffle)
  --sample-seed SEED         default: 42
  --beam-size N              default: method-specific value
  --budget-sentences N       supported for PRIMERA and instruction-tuned runners
  --output-tag TAG           default: run_TIMESTAMP
  --compute-dtype DTYPE      auto, fp32, fp16, or bf16
  --no-tri-metric            for non-baseline legacy objective evidence only

Environment:
  DRY_RUN=1 prints and records the command without running model code.
EOF
}

MODEL=""
METHOD=""
DATASET=""
SPLIT="test"
NUM_SAMPLES="0"
SAMPLE_MODE="shuffle"
SAMPLE_SEED="42"
BEAM_SIZE=""
BUDGET_SENTENCES=""
OUTPUT_TAG=""
COMPUTE_DTYPE=""
TRI_METRIC="1"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="${2:?missing value for --model}"; shift 2 ;;
    --method) METHOD="${2:?missing value for --method}"; shift 2 ;;
    --dataset) DATASET="${2:?missing value for --dataset}"; shift 2 ;;
    --split) SPLIT="${2:?missing value for --split}"; shift 2 ;;
    --num-samples) NUM_SAMPLES="${2:?missing value for --num-samples}"; shift 2 ;;
    --sample-mode) SAMPLE_MODE="${2:?missing value for --sample-mode}"; shift 2 ;;
    --sample-seed) SAMPLE_SEED="${2:?missing value for --sample-seed}"; shift 2 ;;
    --beam-size) BEAM_SIZE="${2:?missing value for --beam-size}"; shift 2 ;;
    --budget-sentences) BUDGET_SENTENCES="${2:?missing value for --budget-sentences}"; shift 2 ;;
    --output-tag) OUTPUT_TAG="${2:?missing value for --output-tag}"; shift 2 ;;
    --compute-dtype) COMPUTE_DTYPE="${2:?missing value for --compute-dtype}"; shift 2 ;;
    --no-tri-metric) TRI_METRIC="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -z "$MODEL" ] || [ -z "$METHOD" ]; then
  echo "Missing --model or --method." >&2
  usage >&2
  exit 1
fi

case "$METHOD" in
  baseline|ilp|dpp|mmr) ;;
  *) echo "Unsupported method: $METHOD" >&2; exit 2 ;;
esac

GENERATOR=""
RUNNER_DIR=""
SUPPORTS_BUDGET=0
case "$MODEL" in
  bart)
    RUNNER_DIR="$ROOT/src/bart"
    GENERATOR="bart"
    DATASET="${DATASET:-cnn_dailymail}"
    COMPUTE_DTYPE="${COMPUTE_DTYPE:-fp32}"
    SUPPORTS_BUDGET=0
    ;;
  primera_multinews)
    RUNNER_DIR="$ROOT/src/primera_multinews"
    GENERATOR="primera_multinews"
    DATASET="${DATASET:-multi_news}"
    COMPUTE_DTYPE="${COMPUTE_DTYPE:-bf16}"
    SUPPORTS_BUDGET=1
    ;;
  llama3_8b)
    RUNNER_DIR="$ROOT/src/llama3_8b"
    GENERATOR="llama3_8b"
    DATASET="${DATASET:-cnn_dailymail}"
    COMPUTE_DTYPE="${COMPUTE_DTYPE:-bf16}"
    SUPPORTS_BUDGET=1
    ;;
  qwen3_5_9b)
    RUNNER_DIR="$ROOT/src/qwen3_5_9b"
    GENERATOR="qwen3.5_9B"
    DATASET="${DATASET:-cnn_dailymail}"
    COMPUTE_DTYPE="${COMPUTE_DTYPE:-bf16}"
    SUPPORTS_BUDGET=1
    ;;
  gemma4_e4b)
    RUNNER_DIR="$ROOT/src/gemma4_e4b"
    GENERATOR="gemma_4_e4b"
    DATASET="${DATASET:-cnn_dailymail}"
    COMPUTE_DTYPE="${COMPUTE_DTYPE:-bf16}"
    SUPPORTS_BUDGET=1
    ;;
  *) echo "Unsupported model: $MODEL" >&2; exit 2 ;;
esac

if [ "$METHOD" = "baseline" ]; then
  case "$MODEL" in
    bart) BEAM_SIZE="${BEAM_SIZE:-4}" ;;
    primera_multinews) BEAM_SIZE="${BEAM_SIZE:-5}" ;;
    *) BEAM_SIZE="${BEAM_SIZE:-8}" ;;
  esac
else
  BEAM_SIZE="${BEAM_SIZE:-8}"
fi

if [ -z "$OUTPUT_TAG" ]; then
  OUTPUT_TAG="run_$(date +%Y%m%d_%H%M%S)"
fi

OUTPUT_DIR="$ROOT/results/runs/$DATASET/$MODEL/$METHOD/$OUTPUT_TAG"
mkdir -p "$OUTPUT_DIR"

CMD=(
  "$PYTHON" "$RUNNER_DIR/run.py"
  --method "$METHOD"
  --generator "$GENERATOR"
  --dataset "$DATASET"
  --split "$SPLIT"
  --num-samples "$NUM_SAMPLES"
  --sample-mode "$SAMPLE_MODE"
  --sample-seed "$SAMPLE_SEED"
  --beam-size "$BEAM_SIZE"
  --compute-dtype "$COMPUTE_DTYPE"
  --output-dir "$OUTPUT_DIR"
)

if [ "$METHOD" != "baseline" ] && [ "$TRI_METRIC" = "1" ]; then
  CMD+=(--tri-metric)
fi

if [ -n "$BUDGET_SENTENCES" ]; then
  if [ "$SUPPORTS_BUDGET" = "1" ]; then
    CMD+=(--budget-sentences "$BUDGET_SENTENCES")
  else
    echo "--budget-sentences is not supported by this runner." >&2
    exit 2
  fi
fi

CMD+=("${EXTRA_ARGS[@]}")

printf '[runner] root=%s\n' "$ROOT"
printf '[runner] output_dir=%s\n' "$OUTPUT_DIR"
printf '[runner] command: '
printf '%q ' "${CMD[@]}"
printf '\n'

if [ "${DRY_RUN:-0}" = "1" ]; then
  MARKER="$OUTPUT_DIR/dry_run_command.txt"
  {
    printf 'dry_run=1\n'
    printf 'command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
  } > "$MARKER"
  echo "[runner] dry run only; wrote $MARKER"
  exit 0
fi

exec "${CMD[@]}"
