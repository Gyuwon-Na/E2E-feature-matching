from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.linalg import polar

from .utils import safe_float


def corners_from_shape(height: int, width: int) -> np.ndarray:
    return np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )


def warp_points(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([pts, ones], axis=1)
    warped = homo @ H.T
    denom = np.where(np.abs(warped[:, 2:3]) < 1e-12, 1e-12, warped[:, 2:3])
    return warped[:, :2] / denom


def make_pixel_grid(height: int, width: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    return np.stack([xs, ys], axis=-1)


def warp_grid(H: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    grid = make_pixel_grid(h, w).reshape(-1, 2)
    warped = warp_points(grid, H).reshape(h, w, 2)
    return warped


def inside_image_mask(coords_xy: np.ndarray, image_shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape_hw
    x = coords_xy[..., 0]
    y = coords_xy[..., 1]
    return (x >= 0.0) & (x <= (w - 1.0)) & (y >= 0.0) & (y <= (h - 1.0))


def homography_from_corners(corners0: np.ndarray, corners1: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(
        np.asarray(corners0, dtype=np.float32), np.asarray(corners1, dtype=np.float32)
    ).astype(np.float64)


METHOD_MAP = {
    "ransac": getattr(cv2, "RANSAC", 0),
    "lmeds": getattr(cv2, "LMEDS", 4),
    "usac_magsac": getattr(cv2, "USAC_MAGSAC", getattr(cv2, "RANSAC", 0)),
    "usac_default": getattr(cv2, "USAC_DEFAULT", getattr(cv2, "RANSAC", 0)),
    "usac_fast": getattr(cv2, "USAC_FAST", getattr(cv2, "RANSAC", 0)),
}


def fit_homography(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    method: str = "usac_magsac",
    ransac_reproj_threshold: float = 3.0,
    max_iters: int = 10000,
    confidence: float = 0.999,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    src = np.asarray(src_xy, dtype=np.float32)
    dst = np.asarray(dst_xy, dtype=np.float32)
    info: Dict[str, Any] = {
        "fit_method": method,
        "fit_success": False,
        "fit_num_points": int(src.shape[0]),
        "fit_num_inliers": 0,
        "fit_inlier_ratio": float("nan"),
    }
    if src.shape[0] < 4 or dst.shape[0] < 4:
        return None, None, info
    flag = METHOD_MAP.get(method, METHOD_MAP["usac_magsac"])
    try:
        H, mask = cv2.findHomography(
            src,
            dst,
            method=flag,
            ransacReprojThreshold=float(ransac_reproj_threshold),
            maxIters=int(max_iters),
            confidence=float(confidence),
        )
    except TypeError:
        H, mask = cv2.findHomography(
            src,
            dst,
            method=flag,
            ransacReprojThreshold=float(ransac_reproj_threshold),
        )
    if H is None:
        return None, None, info
    H = np.asarray(H, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask).reshape(-1).astype(bool)
        info["fit_num_inliers"] = int(mask.sum())
        info["fit_inlier_ratio"] = float(mask.mean()) if mask.size > 0 else float("nan")
    info["fit_success"] = True
    return H, mask, info


def estimate_rotation_from_homography(H: np.ndarray) -> float:
    H = np.asarray(H, dtype=np.float64)
    A = H[:2, :2]
    try:
        R, _ = polar(A)
        theta = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        return float(theta)
    except Exception:
        theta = math.degrees(math.atan2(A[1, 0], A[0, 0]))
        return float(theta)


def normalized_coords_to_pixel(coords_norm_xy: np.ndarray, height: int, width: int) -> np.ndarray:
    coords = np.asarray(coords_norm_xy, dtype=np.float64)
    x = (coords[..., 0] + 1.0) * 0.5 * max(width - 1, 1)
    y = (coords[..., 1] + 1.0) * 0.5 * max(height - 1, 1)
    return np.stack([x, y], axis=-1)


def displacement_to_absolute(displacement_xy: np.ndarray) -> np.ndarray:
    disp = np.asarray(displacement_xy, dtype=np.float64)
    h, w = disp.shape[:2]
    grid = make_pixel_grid(h, w)
    return grid + disp


def flow_from_homography(H: np.ndarray, image0_shape_hw: Tuple[int, int], image1_shape_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    flow = warp_grid(H, image0_shape_hw)
    valid = inside_image_mask(flow, image1_shape_hw)
    return flow, valid.astype(bool)


def sample_dense_points(
    flow01: np.ndarray,
    valid_mask: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    max_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    flow = np.asarray(flow01, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    h, w = valid.shape
    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64), None
    src = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=-1)
    dst = flow[ys, xs]
    conf = None if confidence is None else np.asarray(confidence, dtype=np.float64)[ys, xs]
    if src.shape[0] > max_points:
        if conf is None:
            idx = np.linspace(0, src.shape[0] - 1, max_points).astype(int)
        else:
            idx = np.argsort(-conf)[:max_points]
        src = src[idx]
        dst = dst[idx]
        if conf is not None:
            conf = conf[idx]
    return src, dst, conf
