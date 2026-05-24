#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

run_logged() {
  local name="$1"
  shift
  PYTHON="$PYTHON" "$ROOT/scripts/run_live.sh" --name "$name" -- "$@"
}

run_logged full_bart_cnn_baseline \
  bash "$ROOT/scripts/run_experiment.sh" \
    --model bart \
    --method baseline \
    --dataset cnn_dailymail \
    --num-samples 0 \
    --beam-size 4 \
    --output-tag full_bart_cnn_baseline -- \
    --rouge-sentence-split pysbd

run_logged full_bart_cnn_mmr \
  bash "$ROOT/scripts/run_experiment.sh" \
    --model bart \
    --method mmr \
    --dataset cnn_dailymail \
    --num-samples 0 \
    --beam-size 5 \
    --output-tag mmr_tri_metric -- \
    --rouge-sentence-split pysbd \
    --w-rouge 0.0 \
    --w-minicheck 0.5 \
    --w-redundancy 0.5

run_logged full_bart_cnn_ilp \
  bash "$ROOT/scripts/run_experiment.sh" \
    --model bart \
    --method ilp \
    --dataset cnn_dailymail \
    --num-samples 0 \
    --beam-size 5 \
    --output-tag ilp_tri_metric_softilp_per_edge_wr010_wm020_wd070 -- \
    --rouge-sentence-split pysbd \
    --w-rouge 0.1 \
    --w-minicheck 0.2 \
    --w-redundancy 0.7

run_logged full_bart_cnn_dpp \
  bash "$ROOT/scripts/run_experiment.sh" \
    --model bart \
    --method dpp \
    --dataset cnn_dailymail \
    --num-samples 0 \
    --beam-size 5 \
    --no-tri-metric \
    --output-tag dpp_minicheck_redundancy -- \
    --objective minicheck_redundancy \
    --rouge-sentence-split pysbd

for model in qwen3_5_9b llama3_8b gemma4_e4b; do
  run_logged "full_${model}_cnn_baseline" \
    bash "$ROOT/scripts/run_experiment.sh" \
      --model "$model" \
      --method baseline \
      --dataset cnn_dailymail \
      --num-samples 0 \
      --output-tag "full_${model}_cnn_baseline"
done

for method in mmr ilp; do
  run_logged "full_llama_cnn_${method}_balanced" \
    bash "$ROOT/scripts/run_experiment.sh" \
      --model llama3_8b \
      --method "$method" \
      --dataset cnn_dailymail \
      --num-samples 0 \
      --beam-size 8 \
      --budget-sentences 4 \
      --output-tag "full_llama_cnn_${method}_balanced" -- \
      --w-rouge 0.20 \
      --w-minicheck 0.60 \
      --w-redundancy 0.20
done

run_logged full_llama_cnn_dpp_diverse \
  bash "$ROOT/scripts/run_experiment.sh" \
    --model llama3_8b \
    --method dpp \
    --dataset cnn_dailymail \
    --num-samples 0 \
    --beam-size 8 \
    --budget-sentences 4 \
    --output-tag full_llama_cnn_dpp_diverse -- \
    --w-rouge 0.01 \
    --w-minicheck 0.495 \
    --w-redundancy 0.495

run_logged full_primera_multinews_baseline \
  bash "$ROOT/scripts/run_experiment.sh" \
    --model primera_multinews \
    --method baseline \
    --dataset multi_news \
    --num-samples 0 \
    --beam-size 5 \
    --output-tag full_primera_multinews_baseline

for method in mmr ilp dpp; do
  run_logged "full_primera_multinews_${method}" \
    bash "$ROOT/scripts/run_experiment.sh" \
      --model primera_multinews \
      --method "$method" \
      --dataset multi_news \
      --num-samples 0 \
      --beam-size 8 \
      --budget-sentences 8 \
      --output-tag "full_primera_multinews_${method}"
done

for model in qwen3_5_9b llama3_8b gemma4_e4b; do
  run_logged "full_${model}_multinews_baseline" \
    bash "$ROOT/scripts/run_experiment.sh" \
      --model "$model" \
      --method baseline \
      --dataset multi_news \
      --num-samples 0 \
      --output-tag "full_${model}_multinews_baseline"
done
