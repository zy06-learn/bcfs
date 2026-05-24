# Result Inventory

`results/tables/current_metrics.csv` is generated from compact result files in
`results/raw/` and the configurable selection list in
`results/tables/selected_rows.csv`.

## Selected Evidence

| Dataset | Family | Evidence root |
| --- | --- | --- |
| CNN/DailyMail | BART baseline and selectors | `results/raw/cnn_dailymail/bart/` |
| CNN/DailyMail | Qwen, Llama, Gemma baselines | `results/raw/cnn_dailymail/{qwen3_5_9b,llama3_8b,gemma4_e4b}/baseline/` |
| CNN/DailyMail | Llama selectors | `results/raw/cnn_dailymail/llama3_8b/{mmr,ilp,dpp}/` |
| Multi-News | PRIMERA baseline and selectors | `results/raw/multi_news/primera_multinews/` |
| Multi-News | Qwen, Llama, Gemma baselines | `results/raw/multi_news/{qwen3_5_9b,llama3_8b,gemma4_e4b}/baseline/` |

## Source Notes

- BART CNN/DailyMail evidence was copied from the older
  `NLP_generatesummary` experiment tree.
- PRIMERA, Llama, Qwen, and Gemma evidence was copied from
  `NLP_ilp_dpp_mmr_experiment`.
- Result files are compact extracts: configuration and metric sections are
  retained, long sample logs are omitted.

## Pending Items

See `results/tables/missing_or_pending.csv`.
