from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from ..common import DenseCorrespondence, PredictionBundle, Sample
from ..geometry import normalized_coords_to_pixel
from ..utils import get_package_version, to_numpy
from .base import BaseMatcher


class DKMMatcher(BaseMatcher):
    @classmethod
    def check_environment(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        version_str = get_package_version("dkm")
        if version_str is None:
            return {
                "ok": False,
                "details": "dkm is not installed. Official repo install is clone + pip install -e .",
            }
        return {"ok": True, "details": f"dkm=={version_str}"}

    def load(self) -> None:
        import torch
        from dkm import dkm_base

        version_name = str(self.config.get("version", "v11"))
        pretrained = bool(self.config.get("pretrained", True))
        self.model = dkm_base(pretrained=pretrained, version=version_name)
        self.model.to(self.device)
        self.model.eval()
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
            raise ValueError(f"Unsupported DKM warp shape: {warp.shape}")

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

        img0 = Image.fromarray(sample.image0.astype(np.uint8)).convert("RGB")
        img1 = Image.fromarray(sample.image1.astype(np.uint8)).convert("RGB")
        with torch.inference_mode():
            warp, certainty = self.model.match(img0, img1)
        warp_np = to_numpy(warp)
        certainty_np = to_numpy(certainty)
        dense = self._dense_from_warp(warp_np, certainty_np, sample)
        return PredictionBundle(dense=dense)
