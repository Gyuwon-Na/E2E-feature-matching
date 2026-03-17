from __future__ import annotations

import importlib.util
from typing import Any, Dict

import numpy as np

from ..common import DenseCorrespondence, PredictionBundle, Sample, SparseCorrespondence
from ..geometry import normalized_coords_to_pixel
from ..utils import get_package_version, to_numpy
from .base import BaseMatcher


class RoMaMatcher(BaseMatcher):
    @classmethod
    def check_environment(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        version_str = get_package_version("romatch")
        if version_str is None and importlib.util.find_spec("romatch") is None:
            return {
                "ok": False,
                "details": "romatch is not installed. Official README suggests uv pip install romatch (or romatch[fused-local-corr]).",
            }
        return {"ok": True, "details": f"romatch=={version_str or 'installed'}"}

    def load(self) -> None:
        from romatch import roma_outdoor

        self.model = roma_outdoor(device=self.device)
        self.loaded = True

    def _dense_from_warp(self, warp: np.ndarray, certainty: np.ndarray, sample: Sample) -> DenseCorrespondence:
        h0, w0 = sample.image0.shape[:2]
        h1, w1 = sample.image1.shape[:2]
        if warp.ndim == 4 and warp.shape[0] == 1:
            warp = warp[0]
        if certainty.ndim == 3 and certainty.shape[0] == 1:
            certainty = certainty[0]
        certainty = np.asarray(certainty, dtype=np.float64)
        if warp.shape[-1] == 4:
            src_xy = normalized_coords_to_pixel(warp[..., 0:2], h0, w0)
            dst_xy = normalized_coords_to_pixel(warp[..., 2:4], h1, w1)
        elif warp.shape[-1] == 2:
            ys, xs = np.meshgrid(np.arange(warp.shape[0]), np.arange(warp.shape[1]), indexing="ij")
            src_xy = np.stack([xs, ys], axis=-1).astype(np.float64)
            dst_xy = normalized_coords_to_pixel(warp[..., 0:2], h1, w1)
        else:
            raise ValueError(f"Unsupported RoMa warp shape: {warp.shape}")
        flow = np.full((h0, w0, 2), np.nan, dtype=np.float64)
        valid = np.zeros((h0, w0), dtype=bool)
        conf = np.zeros((h0, w0), dtype=np.float64)
        src_flat = src_xy.reshape(-1, 2)
        dst_flat = dst_xy.reshape(-1, 2)
        cert_flat = certainty.reshape(-1)
        order = np.argsort(-cert_flat)
        for idx in order:
            x0, y0 = src_flat[idx]
            xi = int(np.rint(x0))
            yi = int(np.rint(y0))
            if 0 <= xi < w0 and 0 <= yi < h0 and cert_flat[idx] >= conf[yi, xi]:
                flow[yi, xi] = dst_flat[idx]
                conf[yi, xi] = float(cert_flat[idx])
                valid[yi, xi] = True
        return DenseCorrespondence(flow01=flow, valid_mask=valid, confidence=conf)

    def predict(self, sample: Sample) -> PredictionBundle:
        import torch

        with torch.inference_mode():
            try:
                warp, certainty = self.model.match(sample.image0_path, sample.image1_path, device=self.device)
            except TypeError:
                warp, certainty = self.model.match(sample.image0_path, sample.image1_path)
        warp_np = to_numpy(warp)
        certainty_np = to_numpy(certainty)

        if warp_np.ndim >= 3 and warp_np.shape[-1] in {2, 4}:
            dense = self._dense_from_warp(warp_np, certainty_np, sample)
            return PredictionBundle(dense=dense)

        sample_count = int(self.config.get("sample_count", 10000))
        try:
            matches, scores = self.model.sample(warp, certainty, sample_count)
        except TypeError:
            matches, scores = self.model.sample(warp, certainty)
        h0, w0 = sample.image0.shape[:2]
        h1, w1 = sample.image1.shape[:2]
        kpts0, kpts1 = self.model.to_pixel_coordinates(matches, h0, w0, h1, w1)
        kpts0 = to_numpy(kpts0).reshape(-1, 2)
        kpts1 = to_numpy(kpts1).reshape(-1, 2)
        scores = to_numpy(scores).reshape(-1)
        return PredictionBundle(sparse=SparseCorrespondence(matches0=kpts0, matches1=kpts1, confidence=scores))
