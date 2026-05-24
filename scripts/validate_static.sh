#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=python
  fi
fi

echo "[validate] bash syntax"
script_list="$(mktemp)"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" - "." > "$script_list" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for base in (root / "scripts", root / "src"):
    if not base.exists():
        continue
    for path in sorted(base.rglob("*.sh")):
        sys.stdout.write(path.relative_to(root).as_posix() + "\n")
PY
while IFS= read -r script; do
  script="${script%$'\r'}"
  [[ -n "$script" ]] && bash -n "$script"
done < "$script_list"
rm -f "$script_list"

echo "[validate] python AST parse without bytecode writes"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" - "." <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
errors = []
files = []
for path in root.rglob("*.py"):
    if "__pycache__" in path.parts:
        continue
    files.append(path)
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as exc:
        errors.append((path, exc))

print(f"[validate] parsed_files={len(files)} errors={len(errors)}")
for path, exc in errors:
    print(f"{path}: {exc!r}", file=sys.stderr)
if errors:
    raise SystemExit(1)
PY

echo "[validate] runner --help imports"
for runner in \
  "src/bart/run.py" \
  "src/primera_multinews/run.py" \
  "src/llama3_8b/run.py" \
  "src/qwen3_5_9b/run.py" \
  "src/gemma4_e4b/run.py"
do
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$runner" --help >/dev/null
  echo "[validate] ok $runner --help"
done

echo "[validate] complete"
