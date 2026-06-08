import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml_config(path: str, base_path: Optional[str] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if base_path:
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        cfg = deep_merge(copy.deepcopy(base), cfg)
    return cfg


def resolve_project_paths(cfg: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Turn relative paths in cfg['paths'] into absolute under project_root."""
    paths = cfg.setdefault("paths", {})
    for key in ("data_root", "patch_root", "checkpoint_dir", "log_dir"):
        if key in paths and paths[key]:
            p = Path(paths[key])
            if not p.is_absolute():
                paths[key] = str((project_root / p).resolve())
    return cfg
