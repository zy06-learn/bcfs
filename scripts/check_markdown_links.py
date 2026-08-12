#!/usr/bin/env python3
"""Check local links and images in tracked Markdown files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def tracked_markdown() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(ROOT / line for line in result.stdout.splitlines() if line)


def local_target(raw: str) -> str | None:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if " " in target and not target.startswith("#"):
        target = target.split(None, 1)[0]
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    return unquote(target.split("#", 1)[0])


def main() -> None:
    missing: list[str] = []
    checked = 0
    for markdown in tracked_markdown():
        for line_number, line in enumerate(markdown.read_text(encoding="utf-8").splitlines(), start=1):
            for match in LINK_RE.finditer(line):
                target = local_target(match.group(1))
                if not target:
                    continue
                checked += 1
                resolved = (markdown.parent / target).resolve()
                if not resolved.exists():
                    missing.append(
                        f"{markdown.relative_to(ROOT)}:{line_number}: missing {match.group(1)}"
                    )
    if missing:
        print("\n".join(missing), file=sys.stderr)
        raise SystemExit(1)
    print(f"[links] verified {checked} local Markdown links")


if __name__ == "__main__":
    main()
