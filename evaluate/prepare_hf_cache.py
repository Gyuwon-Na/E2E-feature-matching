#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List

from evalkit.utils import ensure_dir, write_json

MODELS = [
    {"name": "efficientloftr", "model_id": "zju-community/efficientloftr"},
    {"name": "lightglue_superpoint", "model_id": "ETH-CVG/lightglue_superpoint"},
    {"name": "superglue_outdoor", "model_id": "magic-leap-community/superglue_outdoor"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/save stable Hugging Face matching baselines under ../checkpoints/hf")
    parser.add_argument("--output-root", default="../checkpoints/hf", help="Root directory for saved HF model snapshots")
    parser.add_argument("--revision", default=None, help="Optional HF revision/commit/tag to pin")
    parser.add_argument(
        "--include-bin",
        action="store_true",
        help="Also download *.bin files. By default only safetensors/config/processor files are downloaded.",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    out_root = os.path.abspath(args.output_root)
    ensure_dir(out_root)

    ignore_patterns = ["*.h5", "*.ot", "*.msgpack"]
    if not args.include_bin:
        ignore_patterns.append("*.bin")

    manifest: List[Dict[str, str]] = []
    for item in MODELS:
        name = item["name"]
        model_id = item["model_id"]
        dst = os.path.join(out_root, name)
        ensure_dir(dst)
        print(f"[INFO] Snapshotting {model_id} -> {dst}")
        snapshot_path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            revision=args.revision,
            local_dir=dst,
            ignore_patterns=ignore_patterns,
        )
        manifest.append({"name": name, "model_id": model_id, "path": dst, "snapshot_path": snapshot_path})

    write_json(os.path.join(out_root, "cache_manifest.json"), manifest)
    print(f"[INFO] Done. Cached {len(manifest)} model repos under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
