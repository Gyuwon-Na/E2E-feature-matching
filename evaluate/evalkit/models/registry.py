from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseMatcher
from .dkm_matcher import DKMMatcher
from .hf_matchers import EfficientLoFTRHFMatcher, LightGlueHFMatcher, SuperGlueHFMatcher
from .kornia_loftr import KorniaLoFTRMatcher
from .mock_gt_stage import MockGTStageMatcher
from .roma_matcher import RoMaMatcher
from .user_generic import CustomCallableMatcher, UserTorchModuleMatcher

REGISTRY: Dict[str, Type[BaseMatcher]] = {
    "efficientloftr_hf": EfficientLoFTRHFMatcher,
    "lightglue_hf": LightGlueHFMatcher,
    "superglue_hf": SuperGlueHFMatcher,
    "loftr_kornia": KorniaLoFTRMatcher,
    "roma": RoMaMatcher,
    "dkm": DKMMatcher,
    "user_torch_module": UserTorchModuleMatcher,
    "user_custom_callable": CustomCallableMatcher,
    "mock_gt_stage": MockGTStageMatcher,
}


def get_matcher_class(kind: str) -> Type[BaseMatcher]:
    if kind not in REGISTRY:
        raise KeyError(f"Unknown matcher kind: {kind}. Available: {sorted(REGISTRY.keys())}")
    return REGISTRY[kind]


def build_matcher(config: Dict[str, Any]) -> BaseMatcher:
    cls = get_matcher_class(str(config["kind"]))
    matcher = cls(config)
    matcher.load()
    return matcher
