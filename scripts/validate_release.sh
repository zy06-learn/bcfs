#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODE="${1:-}"

if [[ -n "$MODE" && "$MODE" != "--lightweight" ]]; then
  echo "usage: bash scripts/validate_release.sh [--lightweight]" >&2
  exit 2
fi

cd "$ROOT"

echo "[release] shell syntax"
while IFS= read -r -d '' script; do
  sed $'s/\r$//' "$script" | bash -n
done < <(find scripts -type f -name '*.sh' -print0)

echo "[release] Python syntax"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" - <<'PY'
import ast
from pathlib import Path

files = sorted(path for base in (Path("scripts"), Path("src")) for path in base.rglob("*.py"))
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"[release] parsed {len(files)} Python files")
PY

echo "[release] JavaScript syntax"
while IFS= read -r -d '' script; do
  node --check "$script" >/dev/null
done < <(find scripts survey worker -type f \( -name '*.js' -o -name '*.mjs' \) -print0)

"$PYTHON" scripts/validate_paper_artifacts.py
"$PYTHON" scripts/collect_current_metrics.py --check
"$PYTHON" scripts/check_markdown_links.py
npm run worker:test

if [[ "$MODE" != "--lightweight" ]]; then
  normalized_static="$(mktemp "$SCRIPT_DIR/.validate_static.XXXXXX")"
  trap 'rm -f "$normalized_static"' EXIT
  sed $'s/\r$//' scripts/validate_static.sh > "$normalized_static"
  bash "$normalized_static"
fi

echo "[release] complete"
