"""Offline-trained shortlist ranker loaded as a static runtime artifact."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any


RANKER_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "assets" / "offline_shortlist_ranker_v1.json"


@dataclass(slots=True)
class OfflineRankerArtifact:
    version: str
    bias: float
    feature_weights: dict[str, float]


@lru_cache(maxsize=1)
def _load_artifact() -> OfflineRankerArtifact:
    payload = json.loads(RANKER_ARTIFACT_PATH.read_text(encoding="utf-8"))
    weights = payload.get("feature_weights", {})
    if not isinstance(weights, dict):
        weights = {}
    return OfflineRankerArtifact(
        version=str(payload.get("version", "offline-ranker-v1")),
        bias=float(payload.get("bias", 0.0)),
        feature_weights={str(key): float(value) for key, value in weights.items()},
    )


def _get_axis(result: object, axis_name: str) -> float:
    merit_breakdown = getattr(result, "merit_breakdown", {}) or {}
    if isinstance(merit_breakdown, dict):
        return float(merit_breakdown.get(axis_name, 0.0)) / 100.0
    return 0.0


def build_offline_ranker_feature_map(result: object) -> dict[str, float]:
    return {
        "merit_score": float(getattr(result, "merit_score", 0.0)) / 100.0,
        "confidence_score": float(getattr(result, "confidence_score", 0.0)) / 100.0,
        "authenticity_risk_neg": 1.0 - (float(getattr(result, "authenticity_risk", 0.0)) / 100.0),
        "hidden_potential_score": float(getattr(result, "hidden_potential_score", 0.0)) / 100.0,
        "support_needed_score": float(getattr(result, "support_needed_score", 0.0)) / 100.0,
        "shortlist_priority_score": float(getattr(result, "shortlist_priority_score", 0.0)) / 100.0,
        "evidence_coverage_score": float(getattr(result, "evidence_coverage_score", 0.0)) / 100.0,
        "trajectory_score": float(getattr(result, "trajectory_score", 0.0)) / 100.0,
        "community_values_axis": _get_axis(result, "community_values"),
        "leadership_axis": _get_axis(result, "leadership_agency"),
        "potential_axis": _get_axis(result, "potential"),
        "trust_axis": _get_axis(result, "trust_completeness"),
    }


def score_result_with_offline_ranker(result: object, artifact: OfflineRankerArtifact | None = None) -> float:
    artifact = artifact or _load_artifact()
    features = build_offline_ranker_feature_map(result)
    score = artifact.bias
    for feature_name, weight in artifact.feature_weights.items():
        score += features.get(feature_name, 0.0) * weight
    return score


def rank_results_with_offline_ranker(results: list[object]) -> list[object]:
    artifact = _load_artifact()
    return sorted(
        results,
        key=lambda item: (
            score_result_with_offline_ranker(item, artifact),
            float(getattr(item, "shortlist_priority_score", 0.0)),
            float(getattr(item, "hidden_potential_score", 0.0)),
            float(getattr(item, "trajectory_score", 0.0)),
            float(getattr(item, "confidence_score", 0.0)),
            -float(getattr(item, "authenticity_risk", 100.0)),
        ),
        reverse=True,
    )


def get_offline_ranker_metadata() -> dict[str, Any]:
    artifact = _load_artifact()
    return {
        "version": artifact.version,
        "feature_count": len(artifact.feature_weights),
    }
