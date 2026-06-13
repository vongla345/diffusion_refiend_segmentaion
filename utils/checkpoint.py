import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import load_dotenv  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - dotenv is an optional dependency
    load_dotenv = None

# Matches ${VAR} placeholders inside string values.
_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_DOTENV_LOADED = False


def _parse_dotenv_file(env_path: Path) -> None:
    """Minimal .env parser used when python-dotenv is unavailable.

    Existing environment variables take precedence and are never overwritten.
    """
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _ensure_dotenv_loaded() -> None:
    """Load the project .env once so ${VAR} placeholders can be resolved.

    Uses python-dotenv when available, otherwise falls back to a small built-in
    parser so token resolution never silently fails in environments where the
    optional dependency is missing.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    if load_dotenv is not None:
        load_dotenv(env_path)
    else:
        _parse_dotenv_file(env_path)


def _expand_env(value: Any) -> Any:
    """Recursively replace ${VAR} placeholders in strings using os.environ."""
    if isinstance(value, str):
        def repl(match: "re.Match[str]") -> str:
            name = match.group(1)
            return os.environ.get(name, match.group(0))

        return _ENV_PLACEHOLDER_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml_config(path: str, base_path: Optional[str] = None) -> Dict[str, Any]:
    _ensure_dotenv_loaded()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if base_path:
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        cfg = deep_merge(copy.deepcopy(base), cfg)
    return _expand_env(cfg)


def resolve_project_paths(cfg: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Turn relative paths in cfg['paths'] into absolute under project_root."""
    paths = cfg.setdefault("paths", {})
    for key in ("data_root", "patch_root", "checkpoint_dir", "log_dir"):
        if key in paths and paths[key]:
            p = Path(paths[key])
            if not p.is_absolute():
                paths[key] = str((project_root / p).resolve())
    return cfg
