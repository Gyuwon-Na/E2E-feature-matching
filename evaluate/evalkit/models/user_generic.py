from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..common import DenseCorrespondence, PredictionBundle, Sample, SparseCorrespondence
from ..geometry import displacement_to_absolute, normalized_coords_to_pixel
from ..utils import (
    as_float_array,
    import_object,
    load_checkpoint_into_model,
    nested_get,
    read_image_rgb,
    to_numpy,
)
from .base import BaseMatcher



def _squeeze_to_hwc2(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 2 and arr.shape[-1] != 2:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Expected flow-like array shaped [H,W,2] or [2,H,W], got {arr.shape}")
    return arr.astype(np.float64)



def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected array shaped [H,W], got {arr.shape}")
    return arr



def _prepare_input(sample: Sample, mode: str, device: str) -> Tuple[Any, Any]:
    if mode == "path":
        return sample.image0_path, sample.image1_path
    if mode == "pil":
        return Image.fromarray(sample.image0.astype(np.uint8)).convert("RGB"), Image.fromarray(sample.image1.astype(np.uint8)).convert("RGB")
    if mode == "numpy_uint8":
        return sample.image0.copy(), sample.image1.copy()
    import torch

    if mode == "tensor_rgb_0_1":
        t0 = torch.from_numpy(sample.image0).permute(2, 0, 1).float()[None] / 255.0
        t1 = torch.from_numpy(sample.image1).permute(2, 0, 1).float()[None] / 255.0
        return t0.to(device), t1.to(device)
    if mode == "tensor_gray_0_1":
        t0 = torch.from_numpy(sample.image0.mean(axis=2, keepdims=False)).float()[None, None] / 255.0
        t1 = torch.from_numpy(sample.image1.mean(axis=2, keepdims=False)).float()[None, None] / 255.0
        return t0.to(device), t1.to(device)
    raise ValueError(f"Unsupported input_mode: {mode}")



def _dense_from_pair4(raw_pair4: np.ndarray, confidence: Optional[np.ndarray], sample: Sample) -> DenseCorrespondence:
    h0, w0 = sample.image0.shape[:2]
    h1, w1 = sample.image1.shape[:2]
    pair4 = np.asarray(raw_pair4)
    while pair4.ndim >= 4 and pair4.shape[0] == 1:
        pair4 = pair4[0]
    if pair4.ndim == 3 and pair4.shape[0] == 4 and pair4.shape[-1] != 4:
        pair4 = np.transpose(pair4, (1, 2, 0))
    if pair4.ndim != 3 or pair4.shape[-1] != 4:
        raise ValueError(f"Expected pair4 array shaped [H,W,4] or [4,H,W], got {pair4.shape}")

    src_xy = normalized_coords_to_pixel(pair4[..., 0:2], h0, w0)
    dst_xy = normalized_coords_to_pixel(pair4[..., 2:4], h1, w1)
    if confidence is None:
        confidence = np.ones(pair4.shape[:2], dtype=np.float64)
    else:
        confidence = _squeeze_hw(np.asarray(confidence)).astype(np.float64)

    flow = np.full((h0, w0, 2), np.nan, dtype=np.float64)
    valid = np.zeros((h0, w0), dtype=bool)
    conf = np.zeros((h0, w0), dtype=np.float64)
    src_flat = src_xy.reshape(-1, 2)
    dst_flat = dst_xy.reshape(-1, 2)
    score_flat = confidence.reshape(-1)
    order = np.argsort(-score_flat)
    for idx in order:
        x0, y0 = src_flat[idx]
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if 0 <= xi < w0 and 0 <= yi < h0 and score_flat[idx] >= conf[yi, xi]:
            flow[yi, xi] = dst_flat[idx]
            conf[yi, xi] = float(score_flat[idx])
            valid[yi, xi] = True
    return DenseCorrespondence(flow01=flow, valid_mask=valid, confidence=conf)



def _build_dense(raw_output: Any, spec: Dict[str, Any], sample: Sample) -> Optional[DenseCorrespondence]:
    flow_path = spec.get("flow")
    if flow_path is None:
        return None
    flow_raw = nested_get(raw_output, flow_path)
    if flow_raw is None:
        return None
    flow_mode = str(spec.get("flow_mode", "absolute"))
    valid_path = spec.get("valid_mask")
    conf_path = spec.get("confidence")
    valid_raw = nested_get(raw_output, valid_path) if valid_path else None
    conf_raw = nested_get(raw_output, conf_path) if conf_path else None

    if flow_mode == "normalized_pair4":
        return _dense_from_pair4(to_numpy(flow_raw), None if conf_raw is None else to_numpy(conf_raw), sample)

    flow = _squeeze_to_hwc2(to_numpy(flow_raw))
    if flow_mode == "absolute":
        dense_flow = flow
    elif flow_mode == "displacement":
        dense_flow = displacement_to_absolute(flow)
    elif flow_mode == "normalized_target":
        h1, w1 = sample.image1.shape[:2]
        dense_flow = normalized_coords_to_pixel(flow, h1, w1)
    else:
        raise ValueError(f"Unsupported flow_mode: {flow_mode}")

    valid = None if valid_raw is None else _squeeze_hw(to_numpy(valid_raw)).astype(bool)
    conf = None if conf_raw is None else _squeeze_hw(to_numpy(conf_raw)).astype(np.float64)
    return DenseCorrespondence(flow01=dense_flow, valid_mask=valid, confidence=conf)



def _build_sparse(raw_output: Any, spec: Dict[str, Any]) -> Optional[SparseCorrespondence]:
    m0_path = spec.get("matches0")
    m1_path = spec.get("matches1")
    if m0_path is None or m1_path is None:
        return None
    m0 = nested_get(raw_output, m0_path)
    m1 = nested_get(raw_output, m1_path)
    if m0 is None or m1 is None:
        return None
    conf_path = spec.get("sparse_confidence") or spec.get("confidence")
    conf = nested_get(raw_output, conf_path) if conf_path else None
    m0_np = np.asarray(to_numpy(m0), dtype=np.float64).reshape(-1, 2)
    m1_np = np.asarray(to_numpy(m1), dtype=np.float64).reshape(-1, 2)
    conf_np = None if conf is None else np.asarray(to_numpy(conf), dtype=np.float64).reshape(-1)
    return SparseCorrespondence(matches0=m0_np, matches1=m1_np, confidence=conf_np)



def _build_homography(raw_output: Any, spec: Dict[str, Any]) -> Optional[np.ndarray]:
    h_path = spec.get("homography")
    if h_path is None:
        return None
    H = nested_get(raw_output, h_path)
    if H is None:
        return None
    return np.asarray(to_numpy(H), dtype=np.float64).reshape(3, 3)



def bundle_from_raw_output(raw_output: Any, output_spec: Dict[str, Any], sample: Sample) -> PredictionBundle:
    if isinstance(raw_output, PredictionBundle):
        return raw_output
    default_spec = output_spec.get("default", output_spec)
    dense = _build_dense(raw_output, default_spec, sample)
    sparse = _build_sparse(raw_output, default_spec)
    homography = _build_homography(raw_output, default_spec)
    bundle = PredictionBundle(homography_0to1=homography, dense=dense, sparse=sparse)
    for stage_name, stage_spec in output_spec.get("stages", {}).items():
        bundle.stages[str(stage_name)] = bundle_from_raw_output(raw_output, stage_spec, sample)
    return bundle


class UserTorchModuleMatcher(BaseMatcher):
    @classmethod
    def check_environment(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint_path = config.get("checkpoint")
        if checkpoint_path and not os.path.exists(checkpoint_path):
            return {"ok": False, "details": f"checkpoint not found: {checkpoint_path}"}
        module_name = str(config.get("module", ""))
        class_name = str(config.get("class_name", ""))
        if not module_name or not class_name:
            return {"ok": False, "details": "module and class_name are required for user_torch_module"}
        try:
            import_object(module_name, class_name)
        except Exception as exc:
            return {"ok": False, "details": f"cannot import {module_name}.{class_name}: {exc}"}
        return {"ok": True, "details": "module/class import succeeded"}

    def load(self) -> None:
        import torch

        cls = import_object(str(self.config["module"]), str(self.config["class_name"]))
        init_args = dict(self.config.get("init_args", {}))
        self.model = cls(**init_args)
        checkpoint_path = self.config.get("checkpoint")
        self.load_report = None
        if checkpoint_path:
            self.load_report = load_checkpoint_into_model(
                self.model,
                checkpoint_path=str(checkpoint_path),
                checkpoint_key=self.config.get("checkpoint_key"),
                strict=bool(self.config.get("strict_load", True)),
                extra_strip_prefixes=self.config.get("strip_prefixes", []),
            )
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def predict(self, sample: Sample) -> PredictionBundle:
        import torch

        fcfg = self.config.get("forward", {})
        input_mode = str(fcfg.get("input_mode", "tensor_rgb_0_1"))
        arg0, arg1 = _prepare_input(sample, input_mode, self.device)
        method_name = str(fcfg.get("method", "forward"))
        use_kwargs = bool(fcfg.get("call_with_kwargs", True))
        input_names = fcfg.get("input_names", ["image0", "image1"])
        extra_kwargs = dict(fcfg.get("extra_kwargs", {}))
        callable_obj = self.model if method_name in {"__call__", "call"} else getattr(self.model, method_name)
        with torch.inference_mode():
            if use_kwargs:
                kwargs = {str(input_names[0]): arg0, str(input_names[1]): arg1}
                kwargs.update(extra_kwargs)
                raw_output = callable_obj(**kwargs)
            else:
                raw_output = callable_obj(arg0, arg1, **extra_kwargs)
        bundle = bundle_from_raw_output(raw_output, self.config.get("output_spec", {}), sample)
        if self.load_report is not None:
            bundle.extras["load_report"] = self.load_report
        return bundle


class CustomCallableMatcher(BaseMatcher):
    @classmethod
    def check_environment(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        factory = config.get("factory")
        predictor = config.get("predictor")
        if not factory or not predictor:
            return {"ok": False, "details": "factory and predictor are required for user_custom_callable"}
        try:
            import_object(str(factory))
            import_object(str(predictor))
        except Exception as exc:
            return {"ok": False, "details": f"cannot import factory/predictor: {exc}"}
        return {"ok": True, "details": "factory/predictor import succeeded"}

    def load(self) -> None:
        self.factory = import_object(str(self.config["factory"]))
        self.predictor = import_object(str(self.config["predictor"]))
        self.obj = self.factory(config=self.config, device=self.device, **dict(self.config.get("factory_kwargs", {})))
        self.loaded = True

    def predict(self, sample: Sample) -> PredictionBundle:
        raw_output = self.predictor(
            self.obj,
            sample=sample,
            config=self.config,
            device=self.device,
            **dict(self.config.get("predictor_kwargs", {})),
        )
        return bundle_from_raw_output(raw_output, self.config.get("output_spec", {}), sample)
