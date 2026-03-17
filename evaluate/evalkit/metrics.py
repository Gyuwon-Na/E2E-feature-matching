from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .common import DenseCorrespondence, PredictionBundle, Sample, SparseCorrespondence
from .geometry import (
    corners_from_shape,
    fit_homography,
    flow_from_homography,
    sample_dense_points,
    warp_grid,
    warp_points,
)
from .utils import metric_mean, safe_float


DEFAULT_FIT_CFG = {
    "method": "usac_magsac",
    "ransac_reproj_threshold": 3.0,
    "max_iters": 10000,
    "confidence": 0.999,
    "max_fit_points": 5000,
}


def _dense_valid_mask(dense: DenseCorrespondence) -> np.ndarray:
    flow = np.asarray(dense.flow01, dtype=np.float64)
    if dense.valid_mask is not None:
        valid = np.asarray(dense.valid_mask, dtype=bool)
    else:
        valid = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
    return valid.astype(bool)



def _dense_confidence(dense: DenseCorrespondence) -> np.ndarray:
    flow = np.asarray(dense.flow01, dtype=np.float64)
    if dense.confidence is not None:
        return np.asarray(dense.confidence, dtype=np.float64)
    return np.ones(flow.shape[:2], dtype=np.float64)



def rasterize_sparse_to_dense(
    sparse: SparseCorrespondence,
    image0_shape_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h0, w0 = image0_shape_hw
    flow = np.full((h0, w0, 2), np.nan, dtype=np.float64)
    valid = np.zeros((h0, w0), dtype=bool)
    conf = np.full((h0, w0), -np.inf, dtype=np.float64)

    matches0 = np.asarray(sparse.matches0, dtype=np.float64)
    matches1 = np.asarray(sparse.matches1, dtype=np.float64)
    if sparse.confidence is None:
        scores = np.ones((matches0.shape[0],), dtype=np.float64)
    else:
        scores = np.asarray(sparse.confidence, dtype=np.float64).reshape(-1)

    for (x0, y0), (x1, y1), score in zip(matches0, matches1, scores):
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if 0 <= xi < w0 and 0 <= yi < h0 and score >= conf[yi, xi]:
            flow[yi, xi] = np.asarray([x1, y1], dtype=np.float64)
            conf[yi, xi] = float(score)
            valid[yi, xi] = True

    conf[~valid] = 0.0
    return flow, valid, conf



def get_direct_dense(bundle: PredictionBundle, image0_shape_hw: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if bundle.dense is not None:
        flow = np.asarray(bundle.dense.flow01, dtype=np.float64)
        valid = _dense_valid_mask(bundle.dense)
        conf = _dense_confidence(bundle.dense)
        return flow, valid, conf
    if bundle.sparse is not None:
        return rasterize_sparse_to_dense(bundle.sparse, image0_shape_hw)
    return None



def get_fit_matches(bundle: PredictionBundle, image0_shape_hw: Tuple[int, int], fit_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if bundle.sparse is not None:
        matches0 = np.asarray(bundle.sparse.matches0, dtype=np.float64)
        matches1 = np.asarray(bundle.sparse.matches1, dtype=np.float64)
        conf = None if bundle.sparse.confidence is None else np.asarray(bundle.sparse.confidence, dtype=np.float64).reshape(-1)
        max_points = int(fit_cfg.get("max_fit_points", DEFAULT_FIT_CFG["max_fit_points"]))
        if matches0.shape[0] > max_points:
            if conf is None:
                idx = np.linspace(0, matches0.shape[0] - 1, max_points).astype(int)
            else:
                idx = np.argsort(-conf)[:max_points]
            matches0 = matches0[idx]
            matches1 = matches1[idx]
            conf = None if conf is None else conf[idx]
        return matches0, matches1, conf
    if bundle.dense is not None:
        flow = np.asarray(bundle.dense.flow01, dtype=np.float64)
        valid = _dense_valid_mask(bundle.dense)
        conf = None if bundle.dense.confidence is None else np.asarray(bundle.dense.confidence, dtype=np.float64)
        return sample_dense_points(
            flow,
            valid,
            confidence=conf,
            max_points=int(fit_cfg.get("max_fit_points", DEFAULT_FIT_CFG["max_fit_points"])),
        )
    return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64), None



def get_pred_homography(bundle: PredictionBundle, image0_shape_hw: Tuple[int, int], fit_cfg: Optional[Dict[str, Any]] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    fit_cfg = {**DEFAULT_FIT_CFG, **(fit_cfg or {})}
    if bundle.homography_0to1 is not None:
        return np.asarray(bundle.homography_0to1, dtype=np.float64).reshape(3, 3), {
            "fit_source": "direct_homography",
            "fit_success": True,
            "fit_num_points": float("nan"),
            "fit_num_inliers": float("nan"),
            "fit_inlier_ratio": float("nan"),
            "fit_method": "direct",
        }
    src, dst, _ = get_fit_matches(bundle, image0_shape_hw, fit_cfg)
    H, _, info = fit_homography(
        src,
        dst,
        method=str(fit_cfg.get("method", DEFAULT_FIT_CFG["method"])),
        ransac_reproj_threshold=float(fit_cfg.get("ransac_reproj_threshold", DEFAULT_FIT_CFG["ransac_reproj_threshold"])),
        max_iters=int(fit_cfg.get("max_iters", DEFAULT_FIT_CFG["max_iters"])),
        confidence=float(fit_cfg.get("confidence", DEFAULT_FIT_CFG["confidence"])),
    )
    info["fit_source"] = "fitted_from_correspondences"
    return H, info



def get_gt_homography(sample: Sample, fit_cfg: Optional[Dict[str, Any]] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    fit_cfg = {**DEFAULT_FIT_CFG, **(fit_cfg or {})}
    if sample.gt_homography_0to1 is not None:
        return np.asarray(sample.gt_homography_0to1, dtype=np.float64).reshape(3, 3), {"gt_h_source": "direct_gt_homography"}
    if sample.gt_flow01 is None:
        return None, {"gt_h_source": "missing"}
    src, dst, _ = sample_dense_points(
        np.asarray(sample.gt_flow01, dtype=np.float64),
        np.asarray(sample.gt_valid_mask, dtype=bool),
        confidence=None,
        max_points=int(fit_cfg.get("max_fit_points", DEFAULT_FIT_CFG["max_fit_points"])),
    )
    H, _, info = fit_homography(
        src,
        dst,
        method=str(fit_cfg.get("method", DEFAULT_FIT_CFG["method"])),
        ransac_reproj_threshold=float(fit_cfg.get("ransac_reproj_threshold", DEFAULT_FIT_CFG["ransac_reproj_threshold"])),
        max_iters=int(fit_cfg.get("max_iters", DEFAULT_FIT_CFG["max_iters"])),
        confidence=float(fit_cfg.get("confidence", DEFAULT_FIT_CFG["confidence"])),
    )
    info["gt_h_source"] = "fitted_from_gt_flow"
    return H, info



def compute_alignment_error(sample: Sample, bundle: PredictionBundle, fit_cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    h0, w0 = sample.image0.shape[:2]
    h1, w1 = sample.image1.shape[:2]
    H_pred, info = get_pred_homography(bundle, (h0, w0), fit_cfg)
    if H_pred is None:
        info["alignment_error_px"] = float("nan")
        return float("nan"), info

    pred_flow, pred_valid = flow_from_homography(H_pred, (h0, w0), (h1, w1))
    if sample.gt_flow01 is None:
        info["alignment_error_px"] = float("nan")
        return float("nan"), info

    valid = np.asarray(sample.gt_valid_mask, dtype=bool) & pred_valid.astype(bool)
    if valid.sum() == 0:
        info["alignment_error_px"] = float("nan")
        return float("nan"), info

    err = np.linalg.norm(pred_flow[valid] - np.asarray(sample.gt_flow01, dtype=np.float64)[valid], axis=-1)
    value = float(err.mean()) if err.size > 0 else float("nan")
    info["alignment_error_px"] = value
    return value, info



def compute_mace(sample: Sample, bundle: PredictionBundle, fit_cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    h0, w0 = sample.image0.shape[:2]
    H_pred, pred_info = get_pred_homography(bundle, (h0, w0), fit_cfg)
    H_gt, gt_info = get_gt_homography(sample, fit_cfg)
    info: Dict[str, Any] = {**pred_info, **gt_info}
    if H_pred is None or H_gt is None:
        info["mace_px"] = float("nan")
        return float("nan"), info
    corners = corners_from_shape(h0, w0)
    pred_c = warp_points(corners, H_pred)
    gt_c = warp_points(corners, H_gt)
    err = np.linalg.norm(pred_c - gt_c, axis=-1)
    value = float(err.mean()) if err.size > 0 else float("nan")
    info["mace_px"] = value
    return value, info



def compute_aepe(sample: Sample, bundle: PredictionBundle) -> Tuple[float, Dict[str, Any]]:
    h0, w0 = sample.image0.shape[:2]
    direct = get_direct_dense(bundle, (h0, w0))
    info: Dict[str, Any] = {}
    if direct is None or sample.gt_flow01 is None:
        info["aepe_px"] = float("nan")
        info["num_pred_corr"] = 0
        return float("nan"), info
    pred_flow, pred_valid, _ = direct
    valid = np.asarray(sample.gt_valid_mask, dtype=bool) & pred_valid.astype(bool)
    info["num_pred_corr"] = int(pred_valid.sum())
    if valid.sum() == 0:
        info["aepe_px"] = float("nan")
        return float("nan"), info
    err = np.linalg.norm(pred_flow[valid] - np.asarray(sample.gt_flow01, dtype=np.float64)[valid], axis=-1)
    value = float(err.mean()) if err.size > 0 else float("nan")
    info["aepe_px"] = value
    return value, info



def compute_coverage(sample: Sample, bundle: PredictionBundle) -> Tuple[float, Dict[str, Any]]:
    h0, w0 = sample.image0.shape[:2]
    total_pixels = int(h0 * w0)
    direct = get_direct_dense(bundle, (h0, w0))
    info: Dict[str, Any] = {
        "num_gt_valid": int(np.asarray(sample.gt_valid_mask, dtype=bool).sum()),
        "num_total_pixels": total_pixels,
    }
    if direct is None:
        info["coverage"] = 0.0
        info["num_pred_corr"] = 0
        return 0.0, info
    _, pred_valid, _ = direct
    pred_count = int(pred_valid.sum())
    value = float(pred_count / total_pixels) if total_pixels > 0 else float("nan")
    info["coverage"] = value
    info["num_pred_corr"] = pred_count
    return value, info



def compute_pck(sample: Sample, bundle: PredictionBundle, thresholds: Sequence[float]) -> Dict[str, float]:
    h0, w0 = sample.image0.shape[:2]
    direct = get_direct_dense(bundle, (h0, w0))
    results: Dict[str, float] = {}
    if direct is None or sample.gt_flow01 is None:
        for thr in thresholds:
            results[f"pck@{int(thr)}px"] = float("nan")
        return results
    pred_flow, pred_valid, _ = direct
    valid = np.asarray(sample.gt_valid_mask, dtype=bool) & pred_valid.astype(bool)
    if valid.sum() == 0:
        for thr in thresholds:
            results[f"pck@{int(thr)}px"] = float("nan")
        return results
    err = np.linalg.norm(pred_flow[valid] - np.asarray(sample.gt_flow01, dtype=np.float64)[valid], axis=-1)
    for thr in thresholds:
        results[f"pck@{int(thr)}px"] = float((err <= float(thr)).mean())
    return results



def compute_all_metrics(sample: Sample, bundle: PredictionBundle, metric_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metric_cfg = metric_cfg or {}
    fit_cfg = {**DEFAULT_FIT_CFG, **metric_cfg.get("homography_fit", {})}
    thresholds = metric_cfg.get("pck_thresholds", [1, 3, 5])

    alignment_error, alignment_info = compute_alignment_error(sample, bundle, fit_cfg)
    mace, mace_info = compute_mace(sample, bundle, fit_cfg)
    aepe, aepe_info = compute_aepe(sample, bundle)
    coverage, coverage_info = compute_coverage(sample, bundle)
    pck = compute_pck(sample, bundle, thresholds)

    row: Dict[str, Any] = {
        "sample_id": sample.sample_id,
        "rotation_deg": None if sample.rotation_deg is None else float(sample.rotation_deg),
        "alignment_error_px": alignment_error,
        "mace_px": mace,
        "aepe_px": aepe,
        "coverage": coverage,
        "num_gt_valid": coverage_info.get("num_gt_valid", int(sample.gt_valid_mask.sum())),
        "num_pred_corr": coverage_info.get("num_pred_corr", aepe_info.get("num_pred_corr", 0)),
    }
    row.update(pck)
    for k, v in alignment_info.items():
        row[f"align__{k}"] = v
    for k, v in mace_info.items():
        row[f"mace__{k}"] = v
    return row



def metric_keys_from_cfg(metric_cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    metric_cfg = metric_cfg or {}
    thresholds = metric_cfg.get("pck_thresholds", [1, 3, 5])
    return [
        "alignment_error_px",
        "mace_px",
        "aepe_px",
        "coverage",
        *[f"pck@{int(t)}px" for t in thresholds],
    ]
