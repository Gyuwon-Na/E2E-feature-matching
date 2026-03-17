from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .common import Sample
from .geometry import estimate_rotation_from_homography, flow_from_homography, homography_from_corners, inside_image_mask
from .utils import load_array_file, load_jsonl, read_image_rgb, resolve_path


@dataclass
class DatasetConfig:
    manifest: str
    base_dir: str


class ManifestDataset:
    """JSONL manifest dataset.

    Each line minimally needs:
      {"sample_id": "...", "image0": "path/to/a.png", "image1": "path/to/b.png", "gt": {...}}

    Supported gt fields inside `gt`:
      - homography_0to1: inline 3x3 list or path to .npy/.npz/.json
      - corners0, corners1: inline 4x2 or path; used to derive homography
      - flow01: inline HxWx2 list or path to .npy/.npz/.json
      - valid_mask: inline HxW list or path to .npy/.npz/.json
      - rotation_deg
    """

    def __init__(self, manifest_path: str):
        self.manifest_path = os.path.abspath(manifest_path)
        self.base_dir = os.path.dirname(self.manifest_path)
        self.rows = load_jsonl(self.manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Sample]:
        for row in self.rows:
            yield self._build_sample(row)

    def _resolve_row_path(self, path: str) -> str:
        return resolve_path(path, self.base_dir)

    def _parse_array_or_path(self, value: Any, npz_key: Optional[str] = None) -> Optional[np.ndarray]:
        if value is None:
            return None
        if isinstance(value, str):
            return np.asarray(load_array_file(self._resolve_row_path(value), npz_key=npz_key))
        return np.asarray(value)

    def _build_sample(self, row: Dict[str, Any]) -> Sample:
        row = copy.deepcopy(row)
        sample_id = str(row.get("sample_id", row.get("id", f"sample_{id(row)}")))
        image0_path = self._resolve_row_path(row["image0"])
        image1_path = self._resolve_row_path(row["image1"])
        image0 = read_image_rgb(image0_path)
        image1 = read_image_rgb(image1_path)
        h0, w0 = image0.shape[:2]
        h1, w1 = image1.shape[:2]

        gt = row.get("gt", {})

        H_gt = self._parse_array_or_path(gt.get("homography_0to1"))
        if H_gt is not None:
            H_gt = H_gt.reshape(3, 3).astype(np.float64)

        corners0 = self._parse_array_or_path(gt.get("corners0"))
        corners1 = self._parse_array_or_path(gt.get("corners1"))
        if H_gt is None and corners0 is not None and corners1 is not None:
            H_gt = homography_from_corners(corners0.reshape(4, 2), corners1.reshape(4, 2))

        flow = self._parse_array_or_path(gt.get("flow01") or gt.get("correspondence_path"))
        if flow is not None:
            flow = flow.astype(np.float64)
            if flow.ndim != 3 or flow.shape[-1] != 2:
                raise ValueError(f"flow01 for sample {sample_id} must have shape [H, W, 2], got {flow.shape}")
        valid_mask = self._parse_array_or_path(gt.get("valid_mask"), npz_key="valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.astype(bool)

        if flow is None and H_gt is not None:
            flow, valid_mask_from_h = flow_from_homography(H_gt, (h0, w0), (h1, w1))
            if valid_mask is None:
                valid_mask = valid_mask_from_h
            else:
                valid_mask = valid_mask.astype(bool) & valid_mask_from_h.astype(bool)

        if flow is not None and valid_mask is None:
            valid_mask = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1]) & inside_image_mask(flow, (h1, w1))

        if flow is None and valid_mask is None:
            valid_mask = np.zeros((h0, w0), dtype=bool)

        if valid_mask.shape != (h0, w0):
            raise ValueError(
                f"valid_mask shape mismatch for sample {sample_id}: expected {(h0, w0)}, got {valid_mask.shape}"
            )

        rotation_deg = gt.get("rotation_deg", row.get("rotation_deg"))
        if rotation_deg is None:
            rotation_deg = row.get("meta", {}).get("rotation_deg")
        if rotation_deg is None and H_gt is not None:
            rotation_deg = estimate_rotation_from_homography(H_gt)
        rotation_deg = None if rotation_deg is None else float(rotation_deg)

        meta = row.get("meta", {})
        if not isinstance(meta, dict):
            meta = {"meta": meta}
        meta["manifest_row"] = row

        return Sample(
            sample_id=sample_id,
            image0_path=image0_path,
            image1_path=image1_path,
            image0=image0,
            image1=image1,
            gt_flow01=flow,
            gt_valid_mask=np.asarray(valid_mask, dtype=bool),
            gt_homography_0to1=H_gt,
            rotation_deg=rotation_deg,
            meta=meta,
        )
