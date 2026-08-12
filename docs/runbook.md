# Maintainer Runbook

## One Run

```bash
PYTHON=python3 \
  scripts/run_live.sh --name example_run -- \
  bash scripts/run_experiment.sh \
    --model bart \
    --method baseline \
    --dataset cnn_dailymail \
    --num-samples 0
```

Use `DRY_RUN=1` and a small `--num-samples` first when checking a new command.
The launcher writes under ignored `results/runs/`; `scripts/run_live.sh` writes
the terminal stream under ignored `logs/`.

Supported models:

- `bart`
- `primera_multinews`
- `llama3_8b`
- `qwen3_5_9b`
- `gemma4_e4b`

Supported methods:

- `baseline`
- `mmr`
- `ilp`
- `dpp`

## Historical Compact-Evidence Run Set

```bash
PYTHON=python3 \
  scripts/run_live.sh --name current_selected_runs -- \
  bash scripts/current_runs/run_current_results.sh
```

This script launches the configurations represented by
`results/tables/selected_rows.csv`. It is a historical evidence replay, not an
exact arXiv Table 1 reproduction. Do not start the full script without first
reviewing every command, expected runtime, output path, model access, and GPU
capacity.

## Metrics Table

```bash
PYTHON=python3 \
  scripts/run_live.sh --name collect_current_metrics -- \
  python3 scripts/collect_current_metrics.py --check
```

The collector reads `results/tables/selected_rows.csv`, parses compact result
files under `results/raw/`, and compares them with
`results/tables/current_metrics.csv`. Omit `--check` only when intentionally
regenerating the derived table.

## Release Check

```bash
bash scripts/validate_release.sh --lightweight
```

Use `bash scripts/validate_release.sh` after installing Python dependencies.
See [`reproducibility.md`](reproducibility.md) for protocol details.
