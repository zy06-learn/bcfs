from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ASSETS_ENV_VAR = "NLM_ASSETS_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG_FILE = PROJECT_ROOT / ".nlm_assets.json"
DEFAULT_ASSETS_ROOT = PROJECT_ROOT.parent / "NLM_assets" / PROJECT_ROOT.name


def _expand_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    return Path(os.path.expandvars(os.path.expanduser(raw_path))).resolve()


def get_assets_root() -> Path:
    env_path = _expand_path(os.environ.get(ASSETS_ENV_VAR))
    if env_path is not None:
        return env_path

    if LOCAL_CONFIG_FILE.exists():
        try:
            data = json.loads(LOCAL_CONFIG_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        config_path = _expand_path(data.get("assets_root"))
        if config_path is not None:
            return config_path

    return DEFAULT_ASSETS_ROOT.resolve()


def get_asset_dir(name: str, fallback_to_project: bool = True) -> Path:
    external_path = get_assets_root() / name
    if external_path.exists():
        return external_path

    if fallback_to_project:
        local_path = PROJECT_ROOT / name
        if local_path.exists():
            return local_path

    return external_path


def require_asset_dir(name: str, fallback_to_project: bool = True) -> Path:
    asset_path = get_asset_dir(name, fallback_to_project=fallback_to_project)
    if asset_path.exists():
        return asset_path

    locations = [
        get_assets_root() / name,
        PROJECT_ROOT / name,
    ]
    checked_locations = ", ".join(str(path) for path in locations)
    raise FileNotFoundError(
        f"Required asset '{name}' was not found. Checked: {checked_locations}. "
        f"Set {ASSETS_ENV_VAR} or create {LOCAL_CONFIG_FILE.name} to point at the asset root."
    )


def ensure_path_on_sys_path(path: str | os.PathLike[str]) -> str:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return resolved


def ensure_asset_repo_on_sys_path(
    name: str,
    *,
    subdir: str | None = None,
    fallback_to_project: bool = True,
) -> Path:
    repo_path = require_asset_dir(name, fallback_to_project=fallback_to_project)
    target = repo_path / subdir if subdir else repo_path
    ensure_path_on_sys_path(target)
    return target
