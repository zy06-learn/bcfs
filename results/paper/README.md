# Published Results

This directory is a machine-readable transcription of the tables in
**arXiv:2608.03655v1**. The source of record is the v1 paper, not the current
runtime defaults and not the compact logs under `results/raw/`.

## Provenance

- Paper: <https://arxiv.org/abs/2608.03655v1>
- Source archive: <https://arxiv.org/src/2608.03655v1>
- Version submitted: 2026-08-04
- Source archive SHA-256:
  `3c0ac11b7fe82d56d2a2c4f8f5bbc2e2654f6b16760f4f367f098d57dfe4bab1`
- Author-provided reference PDF SHA-256 (the arXiv-served PDF may be rebuilt):
  `0362559da287c99bee1f24529828537522e56b61caefbb8cc6fc460903ce3596`

The CSVs contain only values stated in the paper. Formatting such as boldface
and explanatory prose remains in the paper. The main-results CSV stores point
estimates; paired deltas, confidence intervals, and adjusted p-values are in
the significance CSV. `artifact_manifest.sha256` covers every CSV in this
directory and the paper figure used by the repository README.

## Table Map

| Artifact | arXiv v1 source | Notes |
| --- | --- | --- |
| `arxiv_v1_main_results.csv` | Table 1 | Main benchmark results; values are percentages. |
| `arxiv_v1_faithfulness_benchmarks.csv` | Table 2 | FaithBench and TofuEval. |
| `arxiv_v1_selector_ablation.csv` | Table 3 | MMR, ILP, and DPP comparison. |
| `arxiv_v1_human_evaluation.csv` | Table 4 | Preference counts and mean ratings. |
| `arxiv_v1_weight_sensitivity.csv` | Weight-sensitivity appendix table | Beam 8; the paper does not state a row count for this table. |
| `arxiv_v1_budget_analysis.csv` | Sentence-budget appendix table | Sentence budgets 2 through 5. |
| `arxiv_v1_candidate_pool.csv` | Candidate-pool appendix table | Candidate widths 4, 8, and 12. |
| `arxiv_v1_significance.csv` | Statistical-significance appendix table | Paired differences, intervals, and Holm-adjusted tests. |

For human ratings, the paper reports the overall protocol as 100 comparisons
but does not report a non-missing rating count for each dimension. The `n`
cells for rating rows are therefore intentionally empty. Preference rows retain
their explicitly reported comparison counts.

## Important Reproduction Boundary

The paper states that its main optimization runs use candidate width 12,
sentence budget 3, and weights `(0.33, 0.33, 0.34)`. The repository's current
defaults and the committed compact raw evidence preserve earlier configurations
with different budgets, widths, and method-specific weights. Consequently:

- these CSV files are a faithful **paper transcription**;
- `results/tables/current_metrics.csv` is a reproducible **raw-evidence
  inventory**;
- the release does not claim that the current one-command launcher regenerates
  every published row exactly.

See [`../../docs/reproducibility.md`](../../docs/reproducibility.md) for the
full protocol boundary and validation instructions.
