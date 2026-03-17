#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from evalkit.config_io import load_yaml_config, normalize_config
from evalkit.dataset import ManifestDataset
from evalkit.evaluator import evaluate_bundle_tree
from evalkit.models.registry import build_matcher
from evalkit.reporting import (
    aggregate_leaderboard,
    aggregate_rotation_bins,
    aggregate_stage_delta_summary,
    aggregate_stage_deltas,
    aggregate_summary,
)
from evalkit.utils import ensure_dir, get_package_version, write_csv, write_json

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore



def iter_progress(items: Iterable[Any], total: Optional[int] = None, desc: str = "") -> Iterable[Any]:
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc)



def _package_versions() -> Dict[str, Optional[str]]:
    packages = [
        "numpy",
        "opencv-python",
        "opencv-python-headless",
        "scipy",
        "Pillow",
        "PyYAML",
        "transformers",
        "kornia",
        "romatch",
        "dkm",
        "torch",
    ]
    versions = {pkg: get_package_version(pkg) for pkg in packages}
    return versions



def _select_models(models: List[Dict[str, Any]], only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not only:
        return models
    only_set = set(only)
    selected = [m for m in models if str(m.get("name") or m.get("kind")) in only_set or str(m.get("kind")) in only_set]
    missing = sorted(only_set - {str(m.get("name") or m.get("kind")) for m in selected} - {str(m.get("kind")) for m in selected})
    if missing:
        print(f"[WARN] Requested models not found in config: {missing}", file=sys.stderr)
    return selected



def _cleanup_model(match_obj: Any) -> None:
    try:
        if match_obj is not None:
            match_obj.close()
    except Exception:
        pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass



def main() -> int:
    parser = argparse.ArgumentParser(description="Unified evaluation suite for homography / dense matching models.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of samples")
    parser.add_argument("--only", nargs="*", default=None, help="Run only the named models from the config")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if any sample/model fails")
    args = parser.parse_args()

    raw_cfg = load_yaml_config(args.config)
    cfg = normalize_config(raw_cfg)
    metric_cfg = cfg.get("metrics", {}) or {}
    runtime_cfg = cfg.get("runtime", {}) or {}
    output_dir = str(runtime_cfg.get("output_dir"))
    ensure_dir(output_dir)

    write_json(os.path.join(output_dir, "resolved_config.json"), cfg)
    write_json(os.path.join(output_dir, "package_versions.json"), _package_versions())

    dataset_cfg = cfg.get("dataset", {}) or {}
    manifest_path = dataset_cfg.get("manifest")
    if not manifest_path:
        raise ValueError("dataset.manifest is required in the config")
    dataset = ManifestDataset(str(manifest_path))
    dataset_size = len(dataset)
    if args.max_samples is not None:
        dataset_size = min(dataset_size, int(args.max_samples))

    models = _select_models(list(cfg.get("models", []) or []), only=args.only)
    if not models:
        raise ValueError("No models selected. Check the config or --only names.")

    all_rows: List[Dict[str, Any]] = []
    model_reports: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    overall_start = time.time()
    for model_cfg in models:
        model_name = str(model_cfg.get("name") or model_cfg.get("kind"))
        model_cfg = dict(model_cfg)
        model_cfg.setdefault("checkpoint_root", runtime_cfg.get("checkpoint_root"))

        model_start = time.time()
        matcher = None
        model_report: Dict[str, Any] = {
            "model": model_name,
            "kind": model_cfg.get("kind"),
            "status": "unknown",
            "num_rows": 0,
            "elapsed_sec": float("nan"),
        }
        try:
            print(f"\n[INFO] Loading model: {model_name} ({model_cfg.get('kind')})")
            matcher = build_matcher(model_cfg)
            per_model_rows: List[Dict[str, Any]] = []
            sample_iter = iter(dataset)
            if args.max_samples is not None:
                from itertools import islice

                sample_iter = islice(sample_iter, int(args.max_samples))
            for sample in iter_progress(sample_iter, total=dataset_size, desc=model_name):
                try:
                    bundle = matcher.predict(sample)
                    rows = evaluate_bundle_tree(sample, bundle, model_name=model_name, metric_cfg=metric_cfg)
                    per_model_rows.extend(rows)
                except Exception as exc:
                    err = {
                        "model": model_name,
                        "sample_id": getattr(sample, "sample_id", "<unknown>"),
                        "error": repr(exc),
                    }
                    error_rows.append(err)
                    print(f"[ERROR] {model_name} failed on sample {err['sample_id']}: {exc}", file=sys.stderr)
                    if args.fail_fast or bool(runtime_cfg.get("fail_fast", False)):
                        raise
            all_rows.extend(per_model_rows)
            model_report["status"] = "ok"
            model_report["num_rows"] = len(per_model_rows)
        except Exception as exc:
            model_report["status"] = "failed"
            model_report["error"] = repr(exc)
            error_rows.append({"model": model_name, "sample_id": None, "error": repr(exc)})
            print(f"[ERROR] Model failed to load/run: {model_name}: {exc}", file=sys.stderr)
            if args.fail_fast or bool(runtime_cfg.get("fail_fast", False)):
                _cleanup_model(matcher)
                raise
        finally:
            model_report["elapsed_sec"] = round(time.time() - model_start, 3)
            model_reports.append(model_report)
            _cleanup_model(matcher)

    summary_rows = aggregate_summary(all_rows, metric_cfg=metric_cfg)
    leaderboard_rows = aggregate_leaderboard(summary_rows)

    rotation_cfg = metric_cfg.get("rotation", {}) or {}
    rotation_metric = str(rotation_cfg.get("metric", "alignment_error_px"))
    rotation_bins = list(rotation_cfg.get("bins", [0, 5, 15, 30, 60, 180]))
    rotation_rows = aggregate_rotation_bins(all_rows, metric_name=rotation_metric, bin_edges=rotation_bins)

    stage_pairs = metric_cfg.get("stage_pairs", [["after_transformer", "final"]])
    stage_delta_rows = aggregate_stage_deltas(all_rows, metric_cfg=metric_cfg, stage_pairs=stage_pairs)
    stage_delta_summary_rows = aggregate_stage_delta_summary(stage_delta_rows, metric_cfg=metric_cfg)

    write_csv(os.path.join(output_dir, "per_sample.csv"), all_rows)
    write_csv(os.path.join(output_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(output_dir, "leaderboard.csv"), leaderboard_rows)
    write_csv(os.path.join(output_dir, "rotation_bins.csv"), rotation_rows)
    write_csv(os.path.join(output_dir, "stage_deltas.csv"), stage_delta_rows)
    write_csv(os.path.join(output_dir, "stage_delta_summary.csv"), stage_delta_summary_rows)
    write_csv(os.path.join(output_dir, "model_reports.csv"), model_reports)
    write_csv(os.path.join(output_dir, "errors.csv"), error_rows, fieldnames=["model", "sample_id", "error"])

    write_json(
        os.path.join(output_dir, "summary_bundle.json"),
        {
            "elapsed_sec": round(time.time() - overall_start, 3),
            "num_models": len(models),
            "num_samples": dataset_size,
            "summary": summary_rows,
            "leaderboard": leaderboard_rows,
            "rotation_bins": rotation_rows,
            "stage_delta_summary": stage_delta_summary_rows,
            "model_reports": model_reports,
            "errors": error_rows,
        },
    )

    print("\n[INFO] Evaluation finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Leaderboard rows: {len(leaderboard_rows)}")
    print(f"[INFO] Errors: {len(error_rows)}")
    return 0 if not any(r.get("status") == "failed" for r in model_reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
