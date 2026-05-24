# Dependency Notes

Core runtime dependencies are listed in `requirements.txt`.

Optional evaluators may require local model files or external repositories.
Use one of these local asset mechanisms:

- `NLM_ASSETS_DIR=/path/to/assets`
- an untracked `src/.nlm_assets.json`

The tracked `src/.nlm_assets.example.json` documents the expected shape. Do not
commit local model weights, dataset caches, generated traces, or virtual
environments.

Large directories intentionally excluded from version control include:

- `outputs/`
- `results/runs/`
- `archived_large_models/`
- `.venv/`
- dataset and model cache directories
