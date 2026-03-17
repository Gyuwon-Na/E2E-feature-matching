#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

from evalkit.config_io import load_yaml_config, normalize_config
from evalkit.models.registry import get_matcher_class
from evalkit.utils import ensure_dir, write_csv, write_json



def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether all evaluation models can be imported and loaded.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--load-models", action="store_true", help="Actually instantiate and load each model")
    args = parser.parse_args()

    raw_cfg = load_yaml_config(args.config)
    cfg = normalize_config(raw_cfg)
    runtime_cfg = cfg.get("runtime", {}) or {}
    output_dir = os.path.join(str(runtime_cfg.get("output_dir", ".")), "verify_env")
    ensure_dir(output_dir)

    rows: List[Dict[str, Any]] = []
    overall_ok = True
    for model_cfg in list(cfg.get("models", []) or []):
        model_cfg = dict(model_cfg)
        model_cfg.setdefault("checkpoint_root", runtime_cfg.get("checkpoint_root"))
        name = str(model_cfg.get("name") or model_cfg.get("kind"))
        kind = str(model_cfg.get("kind"))
        row: Dict[str, Any] = {
            "model": name,
            "kind": kind,
            "check_ok": False,
            "check_details": "",
            "load_ok": None,
            "load_error": "",
        }
        try:
            cls = get_matcher_class(kind)
            rep = cls.check_environment(model_cfg)
            row["check_ok"] = bool(rep.get("ok", False))
            row["check_details"] = str(rep.get("details", ""))
            if not row["check_ok"]:
                overall_ok = False
            if args.load_models and row["check_ok"]:
                matcher = None
                try:
                    matcher = cls(model_cfg)
                    matcher.load()
                    row["load_ok"] = True
                except Exception as exc:
                    row["load_ok"] = False
                    row["load_error"] = repr(exc)
                    overall_ok = False
                finally:
                    try:
                        if matcher is not None:
                            matcher.close()
                    except Exception:
                        pass
        except Exception as exc:
            row["check_ok"] = False
            row["check_details"] = repr(exc)
            overall_ok = False
        rows.append(row)
        status = "OK" if row["check_ok"] and (row["load_ok"] in {None, True}) else "FAIL"
        print(f"[{status}] {name}: {row['check_details']}")
        if row["load_ok"] is False:
            print(f"      load_error={row['load_error']}")

    write_csv(os.path.join(output_dir, "verify_env.csv"), rows)
    write_json(os.path.join(output_dir, "verify_env.json"), rows)
    print(f"\n[INFO] Wrote environment report to: {output_dir}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
