from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np

from ..common import DenseCorrespondence, PredictionBundle, Sample
from .base import BaseMatcher



def _seed_from_sample(sample_id: str, base_seed: int) -> int:
    h = hashlib.sha1(f"{sample_id}|{base_seed}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)



def _make_noisy_dense(
    gt_flow01: np.ndarray,
    gt_valid_mask: np.ndarray,
    rng: np.random.Generator,
    noise_px: float,
    keep_prob: float,
) -> DenseCorrespondence:
    flow = np.asarray(gt_flow01, dtype=np.float64).copy()
    valid = np.asarray(gt_valid_mask, dtype=bool).copy()
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected gt_flow01 shaped [H,W,2], got {flow.shape}")
    if valid.shape != flow.shape[:2]:
        raise ValueError(f"valid_mask shape mismatch: {valid.shape} vs {flow.shape[:2]}")

    if noise_px > 0:
        noise = rng.normal(loc=0.0, scale=float(noise_px), size=flow.shape)
        flow[valid] = flow[valid] + noise[valid]

    if keep_prob < 1.0:
        keep = rng.random(valid.shape) < float(keep_prob)
        valid = valid & keep

    conf = np.zeros(valid.shape, dtype=np.float64)
    if np.any(valid):
        if noise_px <= 0:
            conf[valid] = 1.0
        else:
            residual = np.linalg.norm(flow - gt_flow01, axis=-1)
            conf[valid] = np.exp(-(residual[valid] / max(noise_px, 1e-6)))
    return DenseCorrespondence(flow01=flow, valid_mask=valid, confidence=conf)


class MockGTStageMatcher(BaseMatcher):
    """Smoke-test matcher built from GT correspondences.

    It synthesizes an intermediate transformer-stage output and a refined final
    output, so the stage-delta reporting can be exercised without any external
    dependencies or checkpoints.
    """

    @classmethod
    def check_environment(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "details": "numpy-only smoke-test matcher"}

    def load(self) -> None:
        self.base_seed = int(self.config.get("seed", 1337))
        self.stage_name = str(self.config.get("intermediate_stage_name", "after_transformer"))
        self.intermediate_noise_px = float(self.config.get("intermediate_noise_px", 4.0))
        self.intermediate_keep_prob = float(self.config.get("intermediate_keep_prob", 0.55))
        self.final_noise_px = float(self.config.get("final_noise_px", 1.0))
        self.final_keep_prob = float(self.config.get("final_keep_prob", 0.9))
        self.emit_direct_h = bool(self.config.get("emit_direct_homography", False))
        self.loaded = True

    def predict(self, sample: Sample) -> PredictionBundle:
        if sample.gt_flow01 is None:
            raise ValueError(
                "MockGTStageMatcher requires gt_flow01 in the manifest so that it can synthesize predictions."
            )
        rng = np.random.default_rng(_seed_from_sample(sample.sample_id, self.base_seed))

        final_dense = _make_noisy_dense(
            sample.gt_flow01,
            sample.gt_valid_mask,
            rng,
            noise_px=self.final_noise_px,
            keep_prob=self.final_keep_prob,
        )
        stage_dense = _make_noisy_dense(
            sample.gt_flow01,
            sample.gt_valid_mask,
            rng,
            noise_px=self.intermediate_noise_px,
            keep_prob=self.intermediate_keep_prob,
        )

        final_bundle = PredictionBundle(
            homography_0to1=np.asarray(sample.gt_homography_0to1, dtype=np.float64).copy()
            if (self.emit_direct_h and sample.gt_homography_0to1 is not None)
            else None,
            dense=final_dense,
            stages={
                self.stage_name: PredictionBundle(
                    homography_0to1=np.asarray(sample.gt_homography_0to1, dtype=np.float64).copy()
                    if (self.emit_direct_h and sample.gt_homography_0to1 is not None)
                    else None,
                    dense=stage_dense,
                )
            },
            extras={
                "mock": True,
                "stage_name": self.stage_name,
            },
        )
        return final_bundle
