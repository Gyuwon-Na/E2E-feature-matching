from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..common import PredictionBundle, Sample
from ..utils import resolve_device


class BaseMatcher(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = str(config.get("name", config.get("kind", self.__class__.__name__)))
        self.device = resolve_device(config.get("device", "auto"))
        self.loaded = False

    @classmethod
    def check_environment(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "details": "no additional dependency checks"}

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, sample: Sample) -> PredictionBundle:
        raise NotImplementedError

    def close(self) -> None:
        return None
