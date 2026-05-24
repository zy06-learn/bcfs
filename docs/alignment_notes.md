# Alignment Notes

The current code and result table are organized around these implementation
components:

- Candidate generation: `src/*/core/model_generation.py` for instruction-tuned
  runners and `src/*/core/beam_search.py` for encoder-decoder runners.
- Candidate pool construction: `src/*/core/orchestration.py`.
- Coverage and factuality scoring: `src/*/core/features.py`.
- Redundancy scoring: `src/*/core/features.py`.
- Budgeted selection: `src/*/opt_selectors/sentence_level/{mmr,ilp,dpp}.py`.
- Realization ordering: `src/*/core/orchestration.py`.
- Evaluation: `src/*/metrics/evaluation.py`.

The current selected rows are controlled by
`results/tables/selected_rows.csv`. New runs should first be added as compact
evidence under `results/raw/` or retained under `results/auxiliary/` until they
are selected.
