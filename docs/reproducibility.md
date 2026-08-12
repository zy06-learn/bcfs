# Reproducibility Guide

This release supports three distinct verification levels. Keeping them
separate avoids presenting a paper-table transcription as if it were generated
by the compact evidence bundle.

## 1. Verify the Published Artifacts

This is the fastest, CPU-only check. It verifies CSV schemas, table invariants,
known anchor values, and SHA-256 digests:

```bash
python scripts/validate_paper_artifacts.py
```

The files in `results/paper/` were transcribed from arXiv v1. They are not
generated from `results/raw/`.

## 2. Reparse the Committed Evidence

The repository contains a small set of result summaries whose configuration
headers and metrics can be parsed without model weights:

```bash
python scripts/collect_current_metrics.py --check
```

To update the derived CSV after deliberately changing
`results/tables/selected_rows.csv` or `results/raw/`, run:

```bash
python scripts/collect_current_metrics.py
```

This evidence layer covers only the rows enumerated in
`results/tables/selected_rows.csv`. It is not a complete archive of the paper's
per-example generations or metric predictions.

## 3. Inspect the Maintained Pipeline

Inspect a small smoke-test command without loading models or datasets:

```bash
DRY_RUN=1 bash scripts/run_experiment.sh \
  --model llama3_8b \
  --method dpp \
  --dataset cnn_dailymail \
  --num-samples 10 \
  --beam-size 4 \
  --budget-sentences 4 \
  --output-tag smoke
```

This confirms command construction only. It does not reproduce a published
row. Do not describe any launch as paper-exact unless every protocol detail,
model and dataset revision, cached candidate pool, per-example prediction, and
metric implementation has been pinned and independently verified. The BART
runner currently takes its sentence budget from `src/bart/core/config.py`
rather than a CLI flag.

## Paper Protocol vs. Maintained Defaults

arXiv v1 states the following default main-experiment configuration:

- optimization candidate width: 12;
- target sentence budget: 3;
- balanced weights: coverage 0.33, factuality 0.33, redundancy 0.34;
- standard test splits: 11,490 CNN/DailyMail examples and 5,622 Multi-News
  examples;
- selectors in the selector comparison share candidate pools and scores.

The checked-in code and compact evidence also retain historical research
configurations. Notable differences include:

| Surface | Current release state | Consequence |
| --- | --- | --- |
| `src/*/core/config.py` | BART/Llama defaults use budget 4; PRIMERA uses budget 8. | Omitting the budget does not reproduce the paper default. |
| Method defaults | Several configs substitute method-specific weights such as `(0.01, 0.495, 0.495)`. | Explicitly pass paper weights when testing that protocol. |
| `scripts/current_runs/run_current_results.sh` | Replays the compact historical evidence set with beam 5 or 8 and historical budgets/weights. | It reproduces the evidence inventory, not arXiv Table 1. |
| `results/raw/` | Contains compact result summaries, not full predictions or every paper run. | Exact per-example re-evaluation and significance recomputation are unavailable from this release alone. |

These mismatches are documented rather than silently rewritten because changing
defaults would alter the behavior of an existing research codebase and still
would not recover missing historical artifacts.

## Data, Models, and Metrics

- Datasets load through Hugging Face `datasets`: `cnn_dailymail` and
  `multi_news`.
- Generator checkpoints are specified in the model-specific configs and CLI
  modules.
- FactCC and FactKB load Hugging Face checkpoints.
- MiniCheck uses its checkpoint cache.
- AlignScore requires an external source directory and checkpoint configured
  through `NLM_ASSETS_DIR` or `src/.nlm_assets.json`.
- FaithLens values are published in the paper artifacts, but FaithLens is not
  implemented by the released evaluation runner.

Dataset caches, gated model access, evaluator checkpoints, and third-party
licenses are the responsibility of the reproducer. See
[`dependency_notes.md`](dependency_notes.md).

## Full-Run Record

Before a full run, record at minimum:

```bash
git rev-parse HEAD
python --version
python -m pip freeze > environment.freeze.txt
nvidia-smi > nvidia-smi.txt
```

Always use `scripts/run_live.sh` so the terminal output is preserved. Keep the
resolved command, model revisions, dataset revisions, random seed, output path,
and completion state with the result. Full runs can take hours and require
substantial GPU memory; no full experiment was launched while preparing this
release.

## Release-Level Validation

Run all dependency-free checks with:

```bash
bash scripts/validate_release.sh --lightweight
```

After installing the maintained environment, run:

```bash
bash scripts/validate_release.sh
```

The full variant adds model-runner import and `--help` checks. Neither variant
downloads models, downloads datasets, launches experiments, deploys GitHub
Pages, or deploys the Cloudflare Worker.
