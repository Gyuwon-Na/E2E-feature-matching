from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
import torch

from evalkit.common import DenseCorrespondence, PredictionBundle
from evalkit.geometry import flow_from_homography


def _add_project_root(package_root: str | None = None) -> str:
    if package_root is None:
        # default: evaluate/ 바로 위를 프로젝트 루트로 가정
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        project_root = os.path.abspath(package_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def _norm_affine_to_pixel_homography(W: Any, width: int, height: int) -> np.ndarray:
    """normalized 2x3 affine(A->B) -> pixel-domain 3x3 homography"""
    if torch.is_tensor(W):
        W = W.detach().cpu().numpy()
    W = np.asarray(W, dtype=np.float64)

    while W.ndim >= 3 and W.shape[0] == 1:
        W = W[0]
    if W.shape == (2, 3):
        Hn = np.eye(3, dtype=np.float64)
        Hn[:2, :3] = W
    elif W.shape == (3, 3):
        Hn = W
    else:
        raise ValueError(f"Expected affine [2,3] or homography [3,3], got {W.shape}")

    N = np.array(
        [[2.0 / width, 0.0, -1.0], [0.0, 2.0 / height, -1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    N_inv = np.linalg.inv(N)
    return N_inv @ Hn @ N


def _bundle_from_homography(H: np.ndarray, sample) -> PredictionBundle:
    flow, valid = flow_from_homography(H, sample.image0.shape[:2], sample.image1.shape[:2])
    conf = np.ones(valid.shape, dtype=np.float64)
    return PredictionBundle(
        homography_0to1=H,
        dense=DenseCorrespondence(flow01=flow, valid_mask=valid, confidence=conf),
    )


def build_model(config: Dict[str, Any], device: str, **factory_kwargs) -> Dict[str, Any]:
    """현재 fine_tune.py 체크포인트 형식(embedder/transformer/refiner)에 맞춘 factory."""
    package_root = factory_kwargs.get("package_root") or config.get("package_root") or ".."
    project_root = _add_project_root(package_root)

    # 네 프로젝트 구조에 맞게 import 경로를 조정해도 됨.
    from pipeline.phase1 import MathGeometricPreprocessor
    from pipeline.phase2 import CliffordPyramidEmbedder
    from pipeline.phase3 import Phase3Transformer
    from pipeline.phase4 import IterativeRefinementLoop
    from train.fine_tune import HIDDEN_DIM, FEATURE_DIM

    hidden_dim = int(factory_kwargs.get("hidden_dim", config.get("hidden_dim", HIDDEN_DIM)))
    feature_dim = int(factory_kwargs.get("feature_dim", config.get("feature_dim", FEATURE_DIM)))
    use_refiner = bool(factory_kwargs.get("use_refiner", config.get("use_refiner", True)))

    preprocessor = MathGeometricPreprocessor()
    embedder = CliffordPyramidEmbedder(hidden_dim=hidden_dim).to(device)
    transformer = Phase3Transformer(feature_dim=feature_dim, embed_dim=hidden_dim).to(device)
    refiner = IterativeRefinementLoop(feature_dim=feature_dim).to(device) if use_refiner else None

    ckpt_path = str(config["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    embedder.load_state_dict(ckpt["embedder"], strict=True)
    transformer.load_state_dict(ckpt["transformer"], strict=True)
    if refiner is not None and "refiner" in ckpt:
        refiner.load_state_dict(ckpt["refiner"], strict=False)

    embedder.eval()
    transformer.eval()
    if refiner is not None:
        refiner.eval()

    return {
        "project_root": project_root,
        "preprocessor": preprocessor,
        "embedder": embedder,
        "transformer": transformer,
        "refiner": refiner,
        "phase_levels": int(factory_kwargs.get("phase_levels", config.get("phase_levels", 5))),
    }


def predict_sample(obj: Dict[str, Any], sample, config: Dict[str, Any], device: str, **predictor_kwargs) -> PredictionBundle:
    """이미지 쌍 -> phase1/2/3/4 -> final + after_transformer stage bundle"""
    _add_project_root(obj.get("project_root"))

    from train.fine_tune import build_phase4_pyramid_features, invert_affine_2x3

    image0 = sample.image0.astype(np.uint8)
    image1 = sample.image1.astype(np.uint8)

    levels = int(obj.get("phase_levels", 5))
    pyramid_a = obj["preprocessor"].process_pyramid(image0, levels=levels)
    pyramid_b = obj["preprocessor"].process_pyramid(image1, levels=levels)

    with torch.inference_mode():
        phase2_a = obj["embedder"](pyramid_a, device)
        phase2_b = obj["embedder"](pyramid_b, device)
        phase3_results = obj["transformer"](phase2_a, phase2_b)

        results_sorted = sorted(phase3_results, key=lambda d: d.get("level", 0))
        finest = results_sorted[0]
        H_after = _norm_affine_to_pixel_homography(
            finest["W_AB"], width=sample.image0.shape[1], height=sample.image0.shape[0]
        )
        after_bundle = _bundle_from_homography(H_after, sample)

        refiner = obj.get("refiner")
        if refiner is None:
            final_bundle = _bundle_from_homography(H_after, sample)
            final_bundle.stages["after_transformer"] = after_bundle
            final_bundle.extras["note"] = "refiner disabled; final == after_transformer"
            return final_bundle

        feats_a, feats_b = build_phase4_pyramid_features(obj["transformer"], phase2_a, phase2_b, detach=True)
        pred_W4_B2A, hist = refiner(feats_a, feats_b, phase3_results=phase3_results, device=device)
        pred_W4_A2B = invert_affine_2x3(pred_W4_B2A)
        H_final = _norm_affine_to_pixel_homography(
            pred_W4_A2B, width=sample.image0.shape[1], height=sample.image0.shape[0]
        )

    final_bundle = _bundle_from_homography(H_final, sample)
    final_bundle.stages["after_transformer"] = after_bundle
    final_bundle.extras["refine_history_len"] = len(hist) if hist is not None else 0
    return final_bundle
