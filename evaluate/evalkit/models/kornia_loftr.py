from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..common import PredictionBundle, Sample, SparseCorrespondence
from ..utils import get_package_version
from .base import BaseMatcher


class KorniaLoFTRMatcher(BaseMatcher):
    @classmethod
    def check_environment(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        version_str = get_package_version("kornia")
        if version_str is None:
            return {"ok": False, "details": "kornia is not installed. Install with: pip install 'kornia>=0.7.0'"}
        return {"ok": True, "details": f"kornia=={version_str}"}

    def load(self) -> None:
        import kornia as K
        import kornia.feature as KF

        self.K = K
        self.matcher = KF.LoFTR(pretrained=str(self.config.get("pretrained", "outdoor")))
        self.matcher.to(self.device)
        self.matcher.eval()
        self.loaded = True

    def predict(self, sample: Sample) -> PredictionBundle:
        import torch

        img0 = torch.from_numpy(sample.image0).permute(2, 0, 1).float()[None] / 255.0
        img1 = torch.from_numpy(sample.image1).permute(2, 0, 1).float()[None] / 255.0
        img0 = img0.to(self.device)
        img1 = img1.to(self.device)
        gray0 = self.K.color.rgb_to_grayscale(img0)
        gray1 = self.K.color.rgb_to_grayscale(img1)
        with torch.inference_mode():
            out = self.matcher({"image0": gray0, "image1": gray1})
        kpts0 = np.asarray(out["keypoints0"].detach().cpu().numpy(), dtype=np.float64)
        kpts1 = np.asarray(out["keypoints1"].detach().cpu().numpy(), dtype=np.float64)
        conf = np.asarray(out.get("confidence", torch.ones((kpts0.shape[0],), device=gray0.device)).detach().cpu().numpy(), dtype=np.float64)
        return PredictionBundle(
            sparse=SparseCorrespondence(matches0=kpts0, matches1=kpts1, confidence=conf),
            extras={"pretrained": self.config.get("pretrained", "outdoor")},
        )
