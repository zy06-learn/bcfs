# Code-Paper Alignment

| Paper component | Release implementation | Boundary |
| --- | --- | --- |
| Candidate generation | `src/{bart,primera_multinews}/core/beam_search.py`; `src/{llama3_8b,qwen3_5_9b,gemma4_e4b}/core/model_generation.py` | Encoder-decoder and instruction-tuned runners have different decoding controls. |
| Sentence pooling and exact deduplication | `src/*/core/orchestration.py` | Generated sentences, not source sentences, form the candidate pool. |
| Coverage utility | `src/*/core/features.py` | The release computes ROUGE-1/ROUGE-2 recall-based coverage. |
| Factuality utility | `src/*/core/features.py`, `src/*/metrics/minicheck_eval_utils.py` | MiniCheck supplies the selection-time factuality signal. |
| Pairwise redundancy | `src/*/core/features.py` | The release uses pairwise ROUGE-L F1, despite broader semantic-similarity wording in arXiv v1. |
| MMR | `src/*/opt_selectors/sentence_level/mmr.py` | Greedy relevance-diversity selection. |
| ILP | `src/*/opt_selectors/sentence_level/ilp.py` | Hard constraints or a soft linearized pairwise penalty, depending on mode. |
| DPP-inspired selector | `src/*/opt_selectors/sentence_level/dpp.py` | Deterministic greedy log determinant; the heuristic kernel is not guaranteed PSD. |
| Source-aligned ordering | `src/*/core/orchestration.py` | Nearest source sentence is determined with ROUGE-L similarity in the release. |
| Evaluation | `src/*/metrics/evaluation.py` | Includes ROUGE, BERTScore, FactCC, MiniCheck, AlignScore, and FactKB where available; FaithLens is not included. |
| Result serialization | `src/*/output/result_saver.py` | Writes configuration and aggregate metrics used by the compact parser. |

The paper's abstract formulation deliberately permits different relevance and
similarity functions. The table above records the concrete released choices,
including two places where the paper uses broader "semantic similarity"
language but this implementation uses ROUGE-L.

Published values live in `results/paper/`. The selected compact evidence is
controlled by `results/tables/selected_rows.csv` and lives in `results/raw/`.
These layers must not be treated as interchangeable; see
[`reproducibility.md`](reproducibility.md).
