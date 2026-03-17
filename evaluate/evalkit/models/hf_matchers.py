from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
from PIL import Image

from ..common import PredictionBundle, Sample, SparseCorrespondence
from ..utils import get_package_version, to_device
from .base import BaseMatcher


class HFKeypointMatcher(BaseMatcher):
    min_transformers_version = "4.0.0"
    default_model_id: str = ""
    default_local_dir_name: str = ""
    model_class_name: str | None = None
    fallback_auto_class_name: str | None = "AutoModelForKeypointMatching"

    @classmethod
    def check_environment(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        version_str = get_package_version("transformers")
        if version_str is None:
            return {
                "ok": False,
                "details": "transformers is not installed. Install with: pip install 'transformers>=5.3.0' safetensors packaging",
            }
        try:
            from packaging import version

            ok = version.parse(version_str) >= version.parse(cls.min_transformers_version)
        except Exception:
            ok = True
        if not ok:
            return {
                "ok": False,
                "details": f"transformers>={cls.min_transformers_version} required, found {version_str}",
            }
        return {"ok": True, "details": f"transformers=={version_str}"}

    def _resolve_source(self) -> str:
        explicit = self.config.get("local_pretrained_dir")
        require_local = bool(self.config.get("require_local_pretrained", False))
        if explicit:
            explicit = os.path.abspath(str(explicit))
            if os.path.isdir(explicit):
                return explicit
            if require_local:
                return explicit
        checkpoint_root = self.config.get("checkpoint_root")
        if checkpoint_root:
            candidate = os.path.abspath(os.path.join(str(checkpoint_root), "hf", self.default_local_dir_name))
            if os.path.isdir(candidate):
                return candidate
        return str(self.config.get("model_id", self.default_model_id))

    def _resolve_processor_source(self, model_source: str) -> str:
        explicit = self.config.get("processor_local_pretrained_dir") or self.config.get("local_pretrained_dir")
        if explicit:
            explicit = os.path.abspath(str(explicit))
            if os.path.isdir(explicit):
                return explicit
        explicit_model_id = self.config.get("processor_model_id")
        if explicit_model_id:
            return str(explicit_model_id)
        return model_source

    def _resolve_model_loader(self):
        import transformers

        candidates = []
        if self.model_class_name:
            candidates.append(self.model_class_name)
        if self.fallback_auto_class_name:
            candidates.append(self.fallback_auto_class_name)
        candidates.append("AutoModel")

        tried = []
        for class_name in candidates:
            cls = getattr(transformers, class_name, None)
            if cls is not None:
                return cls, candidates
            tried.append(class_name)
        raise ImportError(
            f"None of the expected transformers model classes are available: {', '.join(candidates)}"
        )

    def load(self) -> None:
        from transformers import AutoImageProcessor

        source = self._resolve_source()
        processor_source = self._resolve_processor_source(source)
        local_model = os.path.isdir(source)
        local_processor = os.path.isdir(processor_source)
        local_files_only = bool(self.config.get("local_files_only", local_model and local_processor))
        use_fast = bool(self.config.get("use_fast_processor", False))

        self.processor = AutoImageProcessor.from_pretrained(
            processor_source,
            use_fast=use_fast,
            local_files_only=local_files_only,
        )

        model_loader, loader_candidates = self._resolve_model_loader()
        try:
            self.model = model_loader.from_pretrained(source, local_files_only=local_files_only)
        except TypeError:
            self.model = model_loader.from_pretrained(source)

        self.model.to(self.device)
        self.model.eval()
        self.source = source
        self.processor_source = processor_source
        self.loaded = True
        self.extras = {
            "pretrained_source": self.source,
            "processor_source": self.processor_source,
            "model_loader": getattr(model_loader, "__name__", str(model_loader)),
            "use_fast_processor": use_fast,
            "local_files_only": local_files_only,
            "loader_candidates": loader_candidates,
        }

    def _predict_sparse(self, sample: Sample) -> PredictionBundle:
        import torch

        img0 = Image.fromarray(sample.image0.astype(np.uint8)).convert("RGB")
        img1 = Image.fromarray(sample.image1.astype(np.uint8)).convert("RGB")
        inputs = self.processor([img0, img1], return_tensors="pt")
        inputs = to_device(inputs, self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = [[(img0.height, img0.width), (img1.height, img1.width)]]
        processed = self.processor.post_process_keypoint_matching(
            outputs,
            target_sizes,
            threshold=float(self.config.get("postprocess_threshold", 0.2)),
        )[0]
        kpts0 = np.asarray(processed["keypoints0"], dtype=np.float64)
        kpts1 = np.asarray(processed["keypoints1"], dtype=np.float64)
        scores = np.asarray(processed.get("matching_scores", np.ones((kpts0.shape[0],), dtype=np.float64)), dtype=np.float64)
        return PredictionBundle(
            sparse=SparseCorrespondence(matches0=kpts0, matches1=kpts1, confidence=scores),
            extras=dict(self.extras),
        )

    def predict(self, sample: Sample) -> PredictionBundle:
        return self._predict_sparse(sample)


class EfficientLoFTRHFMatcher(HFKeypointMatcher):
    default_model_id = "zju-community/efficientloftr"
    default_local_dir_name = "efficientloftr"
    model_class_name = "EfficientLoFTRForKeypointMatching"


class LightGlueHFMatcher(HFKeypointMatcher):
    default_model_id = "ETH-CVG/lightglue_superpoint"
    default_local_dir_name = "lightglue_superpoint"
    model_class_name = "LightGlueForKeypointMatching"


class SuperGlueHFMatcher(HFKeypointMatcher):
    default_model_id = "magic-leap-community/superglue_outdoor"
    default_local_dir_name = "superglue_outdoor"
    model_class_name = "SuperGlueForKeypointMatching"
