from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import metric_keys_from_cfg
from .utils import metric_mean, metric_median, metric_std, safe_float


def aggregate_summary(rows: Sequence[Dict[str, Any]], metric_cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    metric_keys = metric_keys_from_cfg(metric_cfg)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["stage"]))].append(row)

    summary: List[Dict[str, Any]] = []
    for (model, stage), group in grouped.items():
        entry: Dict[str, Any] = {
            "model": model,
            "stage": stage,
            "num_samples": len(group),
        }
        for key in metric_keys:
            values = [g.get(key) for g in group]
            entry[key] = metric_mean(values)
            entry[f"{key}__median"] = metric_median(values)
            entry[f"{key}__std"] = metric_std(values)
        for aux_key in ["num_gt_valid", "num_pred_corr"]:
            entry[aux_key] = metric_mean([g.get(aux_key) for g in group])
        summary.append(entry)
    summary.sort(key=lambda d: (d["stage"] != "final", d["alignment_error_px"] if math.isfinite(safe_float(d["alignment_error_px"])) else 1e18))
    return summary



def aggregate_leaderboard(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [r for r in summary_rows if str(r.get("stage")) == "final"]
    rows = list(rows)
    rows.sort(
        key=lambda d: (
            safe_float(d.get("alignment_error_px")) if math.isfinite(safe_float(d.get("alignment_error_px"))) else 1e18,
            -(safe_float(d.get("pck@3px")) if math.isfinite(safe_float(d.get("pck@3px"))) else -1e18),
            -(safe_float(d.get("coverage")) if math.isfinite(safe_float(d.get("coverage"))) else -1e18),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows



def _rotation_bin_label(lo: float, hi: float) -> str:
    return f"[{lo:.1f},{hi:.1f})"



def aggregate_rotation_bins(
    rows: Sequence[Dict[str, Any]],
    metric_name: str,
    bin_edges: Sequence[float],
) -> List[Dict[str, Any]]:
    if len(bin_edges) < 2:
        return []
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    edges = list(map(float, bin_edges))
    for row in rows:
        rot = row.get("rotation_deg")
        if rot is None or not math.isfinite(safe_float(rot)):
            continue
        rot_val = safe_float(rot)
        assigned = False
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo <= abs(rot_val) < hi:
                label = _rotation_bin_label(lo, hi)
                grouped[(str(row["model"]), str(row["stage"]), label)].append(row)
                assigned = True
                break
        if not assigned and abs(rot_val) == edges[-1]:
            label = _rotation_bin_label(edges[-2], edges[-1])
            grouped[(str(row["model"]), str(row["stage"]), label)].append(row)

    out: List[Dict[str, Any]] = []
    for (model, stage, label), group in grouped.items():
        entry = {
            "model": model,
            "stage": stage,
            "rotation_bin": label,
            "num_samples": len(group),
            metric_name: metric_mean([g.get(metric_name) for g in group]),
        }
        out.append(entry)
    out.sort(key=lambda d: (d["model"], d["stage"], d["rotation_bin"]))
    return out



def aggregate_stage_deltas(
    rows: Sequence[Dict[str, Any]],
    metric_cfg: Optional[Dict[str, Any]] = None,
    stage_pairs: Optional[Sequence[Sequence[str]]] = None,
) -> List[Dict[str, Any]]:
    metric_keys = metric_keys_from_cfg(metric_cfg)
    if not stage_pairs:
        return []
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        by_key[(str(row["model"]), str(row["sample_id"]), str(row["stage"]))] = row

    deltas: List[Dict[str, Any]] = []
    for pair in stage_pairs:
        if len(pair) != 2:
            continue
        src_stage, dst_stage = pair
        all_keys = set((model, sample_id) for (model, sample_id, stage) in by_key.keys() if stage in {src_stage, dst_stage})
        for model, sample_id in sorted(all_keys):
            src = by_key.get((model, sample_id, src_stage))
            dst = by_key.get((model, sample_id, dst_stage))
            if src is None or dst is None:
                continue
            entry: Dict[str, Any] = {
                "model": model,
                "sample_id": sample_id,
                "src_stage": src_stage,
                "dst_stage": dst_stage,
            }
            for key in metric_keys:
                src_val = safe_float(src.get(key))
                dst_val = safe_float(dst.get(key))
                if not math.isfinite(src_val) or not math.isfinite(dst_val):
                    entry[f"delta__{key}"] = float("nan")
                    continue
                # error metrics: lower is better; coverage/PCK: higher is better.
                if key in {"coverage"} or key.startswith("pck@"):
                    entry[f"delta__{key}"] = float(dst_val - src_val)
                else:
                    entry[f"delta__{key}"] = float(src_val - dst_val)
            deltas.append(entry)
    return deltas



def aggregate_stage_delta_summary(
    delta_rows: Sequence[Dict[str, Any]],
    metric_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    metric_keys = metric_keys_from_cfg(metric_cfg)
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(str(row["model"]), str(row["src_stage"]), str(row["dst_stage"]))].append(row)
    summary: List[Dict[str, Any]] = []
    for (model, src_stage, dst_stage), group in grouped.items():
        entry = {
            "model": model,
            "src_stage": src_stage,
            "dst_stage": dst_stage,
            "num_samples": len(group),
        }
        for key in metric_keys:
            entry[f"delta__{key}"] = metric_mean([g.get(f"delta__{key}") for g in group])
        summary.append(entry)
    summary.sort(key=lambda d: (d["model"], d["src_stage"], d["dst_stage"]))
    return summary
