from __future__ import annotations

import csv
import importlib
import importlib.metadata
import json
import math
import os
import pathlib
import types
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - optional import guard
    torch = None  # type: ignore


def ensure_dir(path: os.PathLike[str] | str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def load_jsonl(path: os.PathLike[str] | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: os.PathLike[str] | str, data: Any) -> None:
    ensure_dir(pathlib.Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: os.PathLike[str] | str, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(pathlib.Path(path).parent)
    rows = list(rows)
    if not rows:
        if fieldnames is None:
            fieldnames = []
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames))
            writer.writeheader()
        return
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def import_object(path_or_module: str, attr: Optional[str] = None) -> Any:
    if attr is not None:
        module = importlib.import_module(path_or_module)
        return getattr(module, attr)
    if ":" in path_or_module:
        mod_name, obj_name = path_or_module.split(":", 1)
        module = importlib.import_module(mod_name)
        return getattr(module, obj_name)
    parts = path_or_module.split(".")
    if len(parts) < 2:
        return importlib.import_module(path_or_module)
    module = importlib.import_module(".".join(parts[:-1]))
    return getattr(module, parts[-1])


_INDEXABLE = (list, tuple)


def nested_get(obj: Any, path: Optional[str], default: Any = None) -> Any:
    if path is None or path == "":
        return obj
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(part, default)
        elif isinstance(cur, _INDEXABLE) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return default
        else:
            cur = getattr(cur, part, default)
        if cur is default:
            return default
    return cur


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    raise TypeError(f"Cannot convert type {type(x)!r} to numpy array")


def to_device(batch: Any, device: str) -> Any:
    if torch is None:
        return batch
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    return batch


def as_float_array(x: Any, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = to_numpy(x).astype(np.float64)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def read_image_rgb(path: os.PathLike[str] | str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def save_image_rgb(path: os.PathLike[str] | str, image: np.ndarray) -> None:
    ensure_dir(pathlib.Path(path).parent)
    Image.fromarray(image.astype(np.uint8)).save(path)


def resolve_path(path: str, base_dir: os.PathLike[str] | str) -> str:
    p = pathlib.Path(path)
    if p.is_absolute():
        return str(p)
    return str((pathlib.Path(base_dir) / p).resolve())


def load_array_file(path: os.PathLike[str] | str, npz_key: Optional[str] = None) -> np.ndarray:
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        npz = np.load(path)
        if npz_key is not None and npz_key in npz:
            return npz[npz_key]
        if len(npz.files) == 1:
            return npz[npz.files[0]]
        raise KeyError(f"Multiple arrays found in {path}; specify npz_key")
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return np.asarray(json.load(f))
    raise ValueError(f"Unsupported array file: {path}")


COMMON_STATE_KEYS = ["state_dict", "model_state_dict", "model", "net", "weights"]
COMMON_PREFIXES = ["module.", "model.", "net.", "matcher."]


def _extract_state_dict(checkpoint: Any, preferred_key: Optional[str] = None) -> Dict[str, Any]:
    if checkpoint is None:
        raise ValueError("Checkpoint is None")
    if preferred_key is not None:
        state = nested_get(checkpoint, preferred_key, None)
        if isinstance(state, Mapping):
            return dict(state)
    if isinstance(checkpoint, Mapping):
        if all(hasattr(v, "shape") or (torch is not None and torch.is_tensor(v)) for v in checkpoint.values()):
            return dict(checkpoint)
        for key in COMMON_STATE_KEYS:
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                return dict(checkpoint[key])
    if hasattr(checkpoint, "state_dict"):
        return dict(checkpoint.state_dict())
    raise ValueError("Could not extract state_dict from checkpoint")


def best_prefix_strip(state_dict: Mapping[str, Any], model_keys: Iterable[str], extra_prefixes: Optional[Sequence[str]] = None) -> Tuple[Dict[str, Any], str]:
    model_keys = list(model_keys)
    candidates = [""] + COMMON_PREFIXES + list(extra_prefixes or [])
    best_prefix = ""
    best_overlap = -1
    best_state = dict(state_dict)
    for prefix in candidates:
        if prefix:
            stripped = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
        else:
            stripped = dict(state_dict)
        overlap = sum(1 for k in stripped.keys() if k in model_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_prefix = prefix
            best_state = stripped
    return best_state, best_prefix


def load_checkpoint_into_model(
    model: Any,
    checkpoint_path: str,
    map_location: str = "cpu",
    checkpoint_key: Optional[str] = None,
    strict: bool = True,
    extra_strip_prefixes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load checkpoints")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint, preferred_key=checkpoint_key)
    model_keys = list(model.state_dict().keys())
    stripped_state, used_prefix = best_prefix_strip(state_dict, model_keys, extra_prefixes=extra_strip_prefixes)
    missing, unexpected = model.load_state_dict(stripped_state, strict=strict)
    return {
        "checkpoint_path": checkpoint_path,
        "used_prefix": used_prefix,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "num_checkpoint_tensors": len(stripped_state),
    }


def get_package_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_device(requested: Optional[str] = None) -> str:
    if torch is None:
        return "cpu"
    if requested is None or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def maybe_make_relative(path: str, base_dir: str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(pathlib.Path(base_dir).resolve()))
    except Exception:
        return path


def safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        x = float(x)
    except Exception:
        return float("nan")
    if math.isfinite(x):
        return x
    return float("nan")


def metric_mean(values: Sequence[Any]) -> float:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def metric_median(values: Sequence[Any]) -> float:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def metric_std(values: Sequence[Any]) -> float:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.std())
