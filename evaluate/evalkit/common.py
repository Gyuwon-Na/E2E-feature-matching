from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

Array = np.ndarray


@dataclass
class DenseCorrespondence:
    """Dense correspondence map from image0 pixels to image1 pixels.

    flow01 uses absolute pixel coordinates in image1 with shape [H, W, 2].
    valid_mask is bool [H, W]. confidence is float [H, W].
    """

    flow01: Array
    valid_mask: Optional[Array] = None
    confidence: Optional[Array] = None


@dataclass
class SparseCorrespondence:
    """Sparse matched points in pixel coordinates."""

    matches0: Array
    matches1: Array
    confidence: Optional[Array] = None


@dataclass
class PredictionBundle:
    """Unified model output.

    A model may output any subset of:
      - homography_0to1
      - dense correspondences
      - sparse correspondences
      - additional stage outputs (same structure)
    """

    homography_0to1: Optional[Array] = None
    dense: Optional[DenseCorrespondence] = None
    sparse: Optional[SparseCorrespondence] = None
    stages: Dict[str, "PredictionBundle"] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Sample:
    sample_id: str
    image0_path: str
    image1_path: str
    image0: Array
    image1: Array
    gt_flow01: Optional[Array]
    gt_valid_mask: Array
    gt_homography_0to1: Optional[Array]
    rotation_deg: Optional[float]
    meta: Dict[str, Any] = field(default_factory=dict)
