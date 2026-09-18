"""YAML 配置加载与覆盖。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 中的键覆盖 base。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    cfg = load_yaml(path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve_nested(cfg: dict, key: str, project_root: str | Path | None = None) -> Any:
    """按 'a.b.c' 取嵌套配置；若值是以 data/ 开头的相对路径则拼上 project_root。"""
    cur: Any = cfg
    for part in key.split("."):
        cur = cur[part]
    if isinstance(cur, str) and project_root and not Path(cur).is_absolute() \
            and cur.startswith(("data/", "code/", "logs/", "checkpoints/", "results/")):
        return str(Path(project_root) / cur)
    return cur
