from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from .utils import resolve_path

PATH_FIELDS = {
    ("dataset", "manifest"),
    ("runtime", "output_dir"),
    ("runtime", "checkpoint_root"),
}
MODEL_PATH_FIELDS = {"checkpoint", "local_pretrained_dir"}
HF_DEFAULT_DIR = {
    "efficientloftr_hf": "efficientloftr",
    "lightglue_hf": "lightglue_superpoint",
    "superglue_hf": "superglue_outdoor",
}



def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)!r}")
    data["__config_path__"] = os.path.abspath(path)
    data["__config_dir__"] = os.path.dirname(os.path.abspath(path))
    return data



def _deep_get(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = mapping
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur



def _deep_set(mapping: Dict[str, Any], keys: Iterable[str], value: Any) -> None:
    keys = list(keys)
    cur = mapping
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value



def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg_dir = cfg.get("__config_dir__", os.getcwd())

    for keys in PATH_FIELDS:
        value = _deep_get(cfg, keys)
        if isinstance(value, str) and value:
            _deep_set(cfg, keys, resolve_path(value, cfg_dir))

    runtime = cfg.setdefault("runtime", {})
    if "checkpoint_root" not in runtime:
        runtime["checkpoint_root"] = resolve_path("../checkpoints", cfg_dir)
    if "output_dir" not in runtime:
        runtime["output_dir"] = resolve_path("outputs/default", cfg_dir)

    models = cfg.get("models", []) or []
    for model in models:
        if not isinstance(model, dict):
            continue
        for field in MODEL_PATH_FIELDS:
            if field in model and isinstance(model[field], str) and model[field]:
                model[field] = resolve_path(model[field], cfg_dir)
        # Default local HF cache layout under checkpoint_root/hf/<kind-specific-dir>
        if str(model.get("kind", "")).endswith("_hf") and "local_pretrained_dir" not in model:
            local_dir = HF_DEFAULT_DIR.get(str(model.get("kind")), str(model.get("name") or model.get("kind")))
            model["local_pretrained_dir"] = os.path.join(runtime["checkpoint_root"], "hf", local_dir)
        if str(model.get("kind")) == "loftr_kornia" and "pretrained" not in model:
            model["pretrained"] = "outdoor"

    return cfg
