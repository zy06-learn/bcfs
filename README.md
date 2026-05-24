# Decoupling Generation and Selection for Budget-Constrained Faithful Summarization

This repository is the reproducibility release for the paper:

**Decoupling Generation and Selection for Budget-Constrained Faithful Summarization**

The project studies a generate-then-select framework for abstractive summarization. A generation model first produces one or more candidate summaries. For combinatorial-optimization runs, those candidates are decomposed into sentence-level units, deduplicated, scored for coverage and factuality, penalized for redundancy, and selected under a sentence budget by MMR, ILP, or a DPP-inspired greedy selector.

This repository is intentionally a compact research release. It includes runnable code, launch scripts, compact result evidence, and table-generation utilities. It does not include full model weights, dataset caches, full generation traces, or the complete local experiment output tree.

## What Is Included

```text
.
├── src/
│   ├── bart/                 # BART CNN/DailyMail runner
│   ├── primera_multinews/    # PRIMERA Multi-News runner
│   ├── llama3_8b/            # Llama-3.1-8B-Instruct runner
│   ├── qwen3_5_9b/           # Qwen3.5-9B runner
│   └── gemma4_e4b/           # Gemma-4-E4B-it runner
├── scripts/
│   ├── run_live.sh           # real-time logging wrapper
│   ├── run_experiment.sh     # common experiment launcher
│   ├── collect_current_metrics.py
│   ├── validate_static.sh
│   └── current_runs/
├── results/
│   ├── raw/                  # compact result files selected for the paper table
│   └── tables/
│       ├── selected_rows.csv
│       ├── current_metrics.csv
│       └── missing_or_pending.csv
├── docs/
│   ├── alignment_notes.md
│   ├── dependency_notes.md
│   ├── result_inventory.md
│   └── runbook.md
├── requirements.txt
└── README.md
```

Important files:

| File | Purpose |
| --- | --- |
| `scripts/run_experiment.sh` | Stable wrapper around model-specific `run.py` entrypoints. |
| `scripts/run_live.sh` | Runs commands with line-buffered terminal output and a saved log. |
| `results/tables/selected_rows.csv` | The result evidence selected for the current paper table. |
| `results/tables/current_metrics.csv` | Parsed metrics generated from the selected result files. |
| `results/tables/missing_or_pending.csv` | Known incomplete, unavailable, or pending results. |
| `docs/alignment_notes.md` | Mapping from paper components to implementation locations. |
| `docs/dependency_notes.md` | Notes on external evaluators and local model assets. |

## Method Summary

The pipeline has three stages.

1. Candidate generation

   Encoder-decoder models use beam-style generation. Instruction-tuned LLMs use dataset-specific prompts and decoding settings implemented in each model runner.

2. Budgeted candidate selection

   Candidate summaries are split into sentences. Exact duplicates are removed. Sentence-level utility combines coverage and factuality signals, and pairwise redundancy discourages repeated content. The release includes:

   | Selector | Implementation note |
   | --- | --- |
   | `baseline` | Direct generation output, no sentence-level recombination. |
   | `mmr` | Greedy relevance-diversity selection. |
   | `ilp` | Integer linear programming with utility and pairwise redundancy penalty. |
   | `dpp` | DPP-inspired greedy quality-diversity selection, not a probabilistically exact DPP sampler. |

3. Summary realization and evaluation

   Selected sentences are ordered by source similarity and concatenated without an additional rewriting model. Outputs are evaluated with ROUGE, BERTScore, FactCC, MiniCheck, AlignScore, and FactKB when the corresponding evaluator is available.

## How to Read the Source Code

The `src/` tree has one runner directory per generator. The directory layouts are intentionally similar, so the same reading path works for BART, PRIMERA, Llama, Qwen, and Gemma. A reviewer who wants to inspect how the methods are computed does not need to read every file; start with the files below.

| What to inspect | Where to look | What it shows |
| --- | --- | --- |
| Experiment entrypoint | `scripts/run_experiment.sh` | Maps `--model` and `--method` to the concrete `src/<model>/run.py` command. |
| Runner arguments | `src/<model>/run.py`, `src/<model>/cli/args.py` | Defines the CLI options, generator name, dataset choice, beam size, budget, and tri-metric flags. |
| Main pipeline | `src/<model>/core/orchestration.py` | Coordinates candidate generation, sentence-pool construction, utility and redundancy scoring, selector calls, ordering, checkpoints, evaluation, and result writing. |
| Encoder-decoder beam search | `src/bart/core/beam_search.py`, `src/primera_multinews/core/beam_search.py` | Current table runs call `batch_beam_search_decode()`, which uses Hugging Face `generate()` with `num_beams` and `num_return_sequences` to return beam candidates and their sequence scores. |
| LLM generation | `src/{llama3_8b,qwen3_5_9b,gemma4_e4b}/core/model_generation.py` | Defines prompts, truncation, sampling or beam-style candidate generation, stopping criteria, and output cleanup. |
| Sentence features | `src/<model>/core/features.py` | Computes coverage utility, MiniCheck factuality utility, ROUGE-L redundancy matrices, and tri-metric utility scores. Current sentence-pool tracing is in `core/orchestration.py`. |
| Selector registry | `src/<model>/opt_selectors/__init__.py` | Exposes the implemented sentence-level methods: `ilp`, `mmr`, and `dpp`. |
| ILP selection | `src/<model>/opt_selectors/sentence_level/ilp.py` | Implements the hard ILP and tri-metric soft ILP sentence-selection objectives. |
| MMR selection | `src/<model>/opt_selectors/sentence_level/mmr.py` | Implements greedy relevance-diversity selection. |
| DPP-inspired selection | `src/<model>/opt_selectors/sentence_level/dpp.py` | Builds a quality-similarity kernel and greedily selects a diverse high-quality subset. |
| Evaluation and results | `src/<model>/metrics/evaluation.py`, `src/<model>/output/result_saver.py` | Computes ROUGE, BERTScore, factuality metrics, and writes compact result files. |

For the ILP implementation, `x_i` indicates whether sentence `i` is selected and `R_ij` is the pairwise redundancy score. The hard ILP path maximizes:

```text
max sum_i u_i x_i
subject to sum_i x_i == budget
           x_i + x_j <= 1 when R_ij exceeds the redundancy threshold
```

The tri-metric soft ILP path uses a pairwise penalty:

```text
max sum_i u_i x_i - alpha sum_{i<j} R_ij y_ij
subject to 1 <= sum_i x_i <= budget
           y_ij <= x_i
           y_ij <= x_j
           y_ij >= x_i + x_j - 1
           x_i, y_ij in {0, 1}
```

Here `y_ij` is the standard linearized indicator for selecting both sentences `i` and `j`. The coefficient `alpha` is derived from the redundancy weight and `--ilp-penalty-scale`.

This release intentionally keeps only the sentence-level `ilp`, `mmr`, and `dpp` code paths used for the current paper tables. Historical local experiments such as LNS, submodular selection, summary-level selectors, and FactGraph wrappers are not part of this release code path.

## Claim Boundaries

The release is written to match the current implementation, not to overstate it.

- The budget is a sentence-count budget in the released experiments, not a strict token-level budget.
- DPP is implemented as a DPP-inspired greedy selector; the release does not claim exact DPP inference or a guaranteed positive semidefinite DPP kernel.
- Coverage and redundancy are primarily ROUGE-style lexical overlap signals in the current code.
- MiniCheck is used for factuality utility and evaluation where available.
- Some Multi-News LLM baseline MiniCheck values are unavailable in the committed compact evidence; these rows are tracked in `results/tables/missing_or_pending.csv`.
- Multi-News Llama CO rows are still pending unless completed result files are added to `results/raw/` and selected in `results/tables/selected_rows.csv`.

## Installation

```bash
git clone <repository-url>
cd bcfs

python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Some evaluators download or load large model checkpoints. Local paths can be configured with `NLM_ASSETS_DIR` or an untracked `src/.nlm_assets.json`; see `docs/dependency_notes.md`.

## Data

The runners use Hugging Face datasets:

- `cnn_dailymail`
- `multi_news`

Dataset files are not committed. They are downloaded or loaded through the normal Hugging Face `datasets` cache on the machine running the experiments.

## Running Experiments

All documented commands use `scripts/run_live.sh` so stdout/stderr are printed in real time and saved under `logs/`.

### Direct Baseline

```bash
PYTHON=python3 scripts/run_live.sh --name full_bart_cnn_baseline -- \
  bash scripts/run_experiment.sh \
    --model bart \
    --method baseline \
    --dataset cnn_dailymail \
    --num-samples 0 \
    --beam-size 4 \
    --output-tag full_bart_cnn_baseline
```

`--num-samples 0` means the full selected split for these scripts.

### CO Selection Run

```bash
PYTHON=python3 scripts/run_live.sh --name full_bart_cnn_ilp -- \
  bash scripts/run_experiment.sh \
    --model bart \
    --method ilp \
    --dataset cnn_dailymail \
    --num-samples 0 \
    --beam-size 5 \
    --output-tag full_bart_cnn_ilp
```

Supported `--method` values:

```text
baseline
mmr
ilp
dpp
```

Supported `--model` values:

```text
bart
primera_multinews
llama3_8b
qwen3_5_9b
gemma4_e4b
```

### Current Selected Run Set

The script below reproduces the run set represented by `results/tables/selected_rows.csv`, subject to GPU memory and local evaluator availability.

```bash
PYTHON=python3 scripts/run_live.sh --name current_selected_runs -- \
  bash scripts/current_runs/run_current_results.sh
```

For expensive full runs, use a machine with a suitable GPU and enough disk space for local model and dataset caches.

## Results

The committed table is generated from compact evidence files:

```bash
PYTHON=python3 scripts/run_live.sh --name collect_current_metrics -- \
  python3 scripts/collect_current_metrics.py
```

This writes:

```text
results/tables/current_metrics.csv
```

Current selected evidence includes:

| Dataset | Rows currently selected |
| --- | --- |
| CNN/DailyMail | BART baseline; BART+MMR/ILP/DPP; Qwen, Llama, Gemma baselines; Llama+MMR/ILP/DPP. |
| Multi-News | PRIMERA baseline; PRIMERA+MMR/ILP/DPP; Qwen, Llama, Gemma baselines. |

Known missing or unavailable items are tracked in:

```text
results/tables/missing_or_pending.csv
```

Do not edit `current_metrics.csv` by hand. Add or remove rows through `selected_rows.csv`, then rerun `scripts/collect_current_metrics.py`.

## Static Validation

Before committing release changes:

```bash
PYTHON=python3 scripts/run_live.sh --name validate_static -- \
  bash scripts/validate_static.sh

PYTHON=python3 scripts/run_live.sh --name collect_current_metrics -- \
  python3 scripts/collect_current_metrics.py
```

The static validator checks shell syntax, Python syntax, and runner `--help` entrypoints. It does not run full experiments.

## Code-Paper Alignment

| Paper component | Implementation location |
| --- | --- |
| Candidate generation | `src/{bart,primera_multinews}/core/beam_search.py::batch_beam_search_decode`, `src/{llama3_8b,qwen3_5_9b,gemma4_e4b}/core/model_generation.py::SummaryGenerator.generate_batch` |
| Sentence pool construction | `src/*/core/orchestration.py::build_candidate_pool_trace` |
| Sentence deduplication | `src/*/core/orchestration.py` |
| Coverage utility | `src/*/core/features.py` |
| MiniCheck factuality utility | `src/*/core/features.py`, `src/*/metrics/minicheck_eval_utils.py` |
| Pairwise redundancy | `src/*/core/features.py` |
| MMR selection | `src/*/opt_selectors/sentence_level/mmr.py` |
| ILP selection | `src/*/opt_selectors/sentence_level/ilp.py` |
| DPP-inspired selection | `src/*/opt_selectors/sentence_level/dpp.py` |
| Source-similarity ordering | `src/*/core/orchestration.py` |
| Evaluation | `src/*/metrics/evaluation.py` |
| Result saving | `src/*/output/result_saver.py` |

Additional alignment notes are in `docs/alignment_notes.md`.

## Adding New Results

1. Run the experiment with `scripts/run_live.sh`.
2. Copy only compact result evidence into `results/raw/` or keep non-selected evidence outside the release.
3. Add the result path to `results/tables/selected_rows.csv`.
4. Regenerate `results/tables/current_metrics.csv`.
5. Update `results/tables/missing_or_pending.csv` if a metric or row remains unavailable.

Large traces, full outputs, model checkpoints, cache directories, and private local paths should not be committed.

