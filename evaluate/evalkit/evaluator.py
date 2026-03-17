from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import PredictionBundle, Sample
from .metrics import compute_all_metrics



def evaluate_bundle_tree(
    sample: Sample,
    bundle: PredictionBundle,
    model_name: str,
    metric_cfg: Optional[Dict[str, Any]] = None,
    stage_name: str = "final",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    current = compute_all_metrics(sample, bundle, metric_cfg=metric_cfg)
    current["model"] = model_name
    current["stage"] = stage_name
    rows.append(current)
    for child_stage, child_bundle in bundle.stages.items():
        rows.extend(evaluate_bundle_tree(sample, child_bundle, model_name, metric_cfg=metric_cfg, stage_name=child_stage))
    return rows
