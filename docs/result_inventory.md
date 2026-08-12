# Result Inventory

## Published Paper Layer

`results/paper/` contains eight CSV tables transcribed from arXiv v1. Its
README maps each CSV to the paper table and records source checksums. This layer
is the source for claims about published numbers.

## Compact Evidence Layer

`results/tables/current_metrics.csv` is generated from compact result files in
`results/raw/` and the selection list in `results/tables/selected_rows.csv`.
Verify it without changing files using:

```bash
python scripts/collect_current_metrics.py --check
```

### Selected Evidence

| Dataset | Family | Evidence root |
| --- | --- | --- |
| CNN/DailyMail | BART baseline and selectors | `results/raw/cnn_dailymail/bart/` |
| CNN/DailyMail | Qwen, Llama, Gemma baselines | `results/raw/cnn_dailymail/{qwen3_5_9b,llama3_8b,gemma4_e4b}/baseline/` |
| CNN/DailyMail | Llama selectors | `results/raw/cnn_dailymail/llama3_8b/{mmr,ilp,dpp}/` |
| Multi-News | PRIMERA baseline and selectors | `results/raw/multi_news/primera_multinews/` |
| Multi-News | Qwen, Llama, Gemma baselines | `results/raw/multi_news/{qwen3_5_9b,llama3_8b,gemma4_e4b}/baseline/` |

### Source Notes

- BART CNN/DailyMail evidence was copied from the older
  `NLP_generatesummary` experiment tree.
- PRIMERA, Llama, Qwen, and Gemma evidence was copied from
  `NLP_ilp_dpp_mmr_experiment`.
- Result files are compact extracts: configuration and metric sections are
  retained, long sample logs are omitted.
- Several rows happen to match values used in the paper, but the evidence layer
  as a whole uses historical beam widths, budgets, and weights and is not the
  complete provenance package for arXiv Table 1.

### Pending Items

See `results/tables/missing_or_pending.csv`.

## Deliberately Excluded

- per-example generation and metric-prediction arrays;
- model weights, dataset caches, and external evaluator checkpoints;
- raw human-participant responses and free-text comments;
- private absolute paths and credentials.
