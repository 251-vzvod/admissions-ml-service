"""Optional review-routing sidecar run in shadow mode.

This layer is intentionally non-authoritative:
- it does not override deterministic recommendation logic
- it does not change public score semantics
- it provides an offline-trained routing hint for internal comparison only
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import CONFIG
from app.services.offline_ranker import build_offline_ranker_feature_map


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


@dataclass(slots=True)
class ReviewRoutingShadowResult:
    enabled: bool
    available: bool
    artifact_name: str | None = None
    artifact_version: str | None = None
    target_name: str | None = None
    model_name: str | None = None
    probability: float | None = None
    threshold: float | None = None
    predicted_positive: bool | None = None
    note: str | None = None

    def as_public_debug_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "artifact_name": self.artifact_name,
            "artifact_version": self.artifact_version,
            "target_name": self.target_name,
            "model_name": self.model_name,
            "probability": self.probability,
            "threshold": self.threshold,
            "predicted_positive": self.predicted_positive,
            "note": self.note,
        }


@dataclass(slots=True)
class ReviewRoutingArtifact:
    artifact_name: str
    artifact_version: str
    target_name: str
    model_name: str
    threshold: float
    feature_names: list[str]
    model: Any


def _artifact_paths() -> tuple[Path, Path]:
    artifact_name = CONFIG.review_routing_sidecar.artifact_name
    return (
        ASSET_DIR / f"{artifact_name}.json",
        ASSET_DIR / f"{artifact_name}.joblib",
    )


@lru_cache(maxsize=1)
def _load_artifact() -> ReviewRoutingArtifact:
    metadata_path, model_path = _artifact_paths()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - exercised only in missing-runtime-dep environments
        raise RuntimeError("joblib / scikit-learn is not installed for review routing sidecar") from exc

    model = joblib.load(model_path)
    return ReviewRoutingArtifact(
        artifact_name=str(metadata.get("artifact_name", CONFIG.review_routing_sidecar.artifact_name)),
        artifact_version=str(metadata.get("artifact_version", "review-routing-sidecar-v1")),
        target_name=str(metadata.get("target_name", "nonstandard_route")),
        model_name=str(metadata.get("model_name", "unknown")),
        threshold=float(metadata.get("threshold", 0.5)),
        feature_names=[str(name) for name in metadata.get("feature_names", [])],
        model=model,
    )


def score_review_routing_shadow(result: object) -> ReviewRoutingShadowResult:
    if not CONFIG.review_routing_sidecar.enabled:
        return ReviewRoutingShadowResult(
            enabled=False,
            available=False,
            artifact_name=CONFIG.review_routing_sidecar.artifact_name,
            note="shadow_sidecar_disabled",
        )

    try:
        artifact = _load_artifact()
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        return ReviewRoutingShadowResult(
            enabled=True,
            available=False,
            artifact_name=CONFIG.review_routing_sidecar.artifact_name,
            note=f"shadow_sidecar_unavailable: {exc}",
        )

    feature_map = build_offline_ranker_feature_map(result)
    if not artifact.feature_names:
        return ReviewRoutingShadowResult(
            enabled=True,
            available=False,
            artifact_name=artifact.artifact_name,
            artifact_version=artifact.artifact_version,
            target_name=artifact.target_name,
            model_name=artifact.model_name,
            note="shadow_sidecar_missing_feature_names",
        )

    vector = np.asarray([[float(feature_map.get(name, 0.0)) for name in artifact.feature_names]], dtype=np.float32)
    probability = float(artifact.model.predict_proba(vector)[0][1])
    predicted_positive = probability >= artifact.threshold

    return ReviewRoutingShadowResult(
        enabled=True,
        available=True,
        artifact_name=artifact.artifact_name,
        artifact_version=artifact.artifact_version,
        target_name=artifact.target_name,
        model_name=artifact.model_name,
        probability=probability,
        threshold=artifact.threshold,
        predicted_positive=predicted_positive,
        note="shadow_only_no_runtime_override",
    )
