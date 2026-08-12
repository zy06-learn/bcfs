# Dependency and Asset Notes

`requirements.txt` contains bounded direct dependencies for the maintained
Python 3.11 release. `environment.yml` wraps that file for Conda. They are not
claimed to reconstruct an unavailable historical transitive lock.

## External Assets

Generator, dataset, and evaluator assets are not committed. Some are fetched
from Hugging Face and some must be installed locally:

| Component | Release expectation |
| --- | --- |
| BART / PRIMERA / Llama / Qwen / Gemma | Model access and a compatible Hugging Face cache. Gated models require acceptance of their terms. |
| CNN/DailyMail / Multi-News | Loaded through `datasets`; pin a dataset revision in any formal reproduction record. |
| FactCC | `manueldeprada/FactCC`. |
| FactKB | `bunsenfeng/FactKB` with the `roberta-base` tokenizer. |
| MiniCheck | A `minicheck_ckpts` directory under the configured asset root. |
| AlignScore | `AlignScore-main/src` plus `alignscore_ckpt/AlignScore-base.ckpt` under the asset root. |
| FaithLens | Reported in paper CSVs only; no release runner is included. |

Configure local assets with either:

```bash
export NLM_ASSETS_DIR=/absolute/path/to/assets
```

or an untracked `src/.nlm_assets.json`. The tracked
`src/.nlm_assets.example.json` documents its shape.

## Excluded Material

Do not commit model weights, dataset caches, raw participant responses,
complete generation traces, credentials, or machine-specific absolute paths.
The `.gitignore` excludes common local surfaces such as `outputs/`,
`results/runs/`, `.venv/`, `.wrangler/`, and `src/.nlm_assets.json`.

All third-party models, datasets, checkpoints, and source repositories retain
their own licenses and usage conditions; the repository's Apache-2.0 license
does not relicense them.
