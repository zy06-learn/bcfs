#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_live.sh [--name RUN_NAME] [--log-dir DIR] -- command [args...]

Examples:
  scripts/run_live.sh --name full_bart_cnn_baseline -- \
    bash scripts/run_experiment.sh --model bart --method baseline --dataset cnn_dailymail

Behavior:
  - Runs in the foreground.
  - Prints logs to the terminal in real time.
  - Writes the same stream to logs/RUN_NAME_TIMESTAMP.log.
EOF
}

NAME="run"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

while [ $# -gt 0 ]; do
  case "$1" in
    --name)
      NAME="${2:?missing value for --name}"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:?missing value for --log-dir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "Missing command to run." >&2
  usage >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${NAME}_${TIMESTAMP}.log"
HOLD_FILE="${QWEN_FULL_AFTER_BASELINE_HOLD_FILE:-}"

export PYTHONUNBUFFERED=1

echo "[start] $(date -Is)" | tee -a "$LOG_FILE"
printf '[command] ' | tee -a "$LOG_FILE"
printf '%q ' "$@" | tee -a "$LOG_FILE"
printf '\n' | tee -a "$LOG_FILE"

case "$NAME" in
  qwen_full_ilp|qwen_full_dpp|qwen_full_mmr)
    if [ -n "$HOLD_FILE" ]; then
      while [ -e "$HOLD_FILE" ]; do
        echo "[hold] $(date -Is) waiting for queue release: $HOLD_FILE" | tee -a "$LOG_FILE"
        sleep 60
      done
    fi
    ;;
esac

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "$@" 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}
else
  "$@" 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}
fi

echo "[exit] $(date -Is) status=$status" | tee -a "$LOG_FILE"
echo "[log] $LOG_FILE"
exit "$status"
