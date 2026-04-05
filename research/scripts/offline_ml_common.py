from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from app.config import CONFIG
from app.services.offline_ranker import score_result_with_offline_ranker
from app.services.pipeline import ScoringPipeline
from app.services.shortlist import build_shortlist_signals
from app.services.text_representation import (
    TextRepresentationConfig,
    TextRepresentationResult,
    build_text_representation,
)


DATA_ROOT = ROOT / "data" / "ml_workbench"
EXPORTS_DIR = DATA_ROOT / "exports"
LABELS_DIR = DATA_ROOT / "labels"
TRAINING_CSV = EXPORTS_DIR / "training_dataset_v3.csv"
OFFLINE_LAYER_DIR = EXPORTS_DIR / "models" / "offline_ml_layer_v1_training_dataset_v3"
REPRESENTATION_CACHE_JOBLIB = OFFLINE_LAYER_DIR / "text_representation_cache.joblib"
REPRESENTATION_CACHE_METADATA_JSON = OFFLINE_LAYER_DIR / "text_representation_cache_metadata.json"
FEATURE_CACHE_JOBLIB = OFFLINE_LAYER_DIR / "candidate_feature_cache.joblib"
FEATURE_CACHE_METADATA_JSON = OFFLINE_LAYER_DIR / "candidate_feature_cache_metadata.json"
BATCH_JSONL = LABELS_DIR / "batch_shortlist_tasks.jsonl"
PAIRWISE_CSV = LABELS_DIR / "pairwise_labels.csv"

SEED_JSONL = DATA_ROOT / "processed" / "english_candidates_api_input_v1.jsonl"
SYNTHETIC_V1_JSONL = DATA_ROOT / "raw" / "generated" / "batch_v1" / "synthetic_batch_v1_api_input.jsonl"
CONTRASTIVE_V2_JSONL = DATA_ROOT / "raw" / "generated" / "contrastive_batch_v2" / "contrastive_batch_v2_api_input.jsonl"
TRANSLATED_V3_JSONL = DATA_ROOT / "raw" / "generated" / "translated_batch_v3" / "translated_batch_v3_api_input.jsonl"
MESSY_V4_JSONL = DATA_ROOT / "raw" / "generated" / "messy_batch_v4" / "messy_batch_v4_api_input.jsonl"
MESSY_V5_JSONL = DATA_ROOT / "raw" / "generated" / "messy_batch_v5" / "messy_batch_v5_api_input.jsonl"
MESSY_V5_EXTENSION_JSONL = DATA_ROOT / "raw" / "generated" / "messy_batch_v5_extension" / "messy_batch_v5_extension_api_input.jsonl"
ORDINARY_V6_JSONL = DATA_ROOT / "raw" / "generated" / "ordinary_batch_v6" / "ordinary_batch_v6_api_input.jsonl"
GAP_FILL_V7_JSONL = DATA_ROOT / "raw" / "generated" / "gap_fill_batch_v7" / "gap_fill_batch_v7_api_input.jsonl"

PAYLOAD_JSONL_PATHS = [
    SEED_JSONL,
    SYNTHETIC_V1_JSONL,
    CONTRASTIVE_V2_JSONL,
    TRANSLATED_V3_JSONL,
    MESSY_V4_JSONL,
    MESSY_V5_JSONL,
    MESSY_V5_EXTENSION_JSONL,
    ORDINARY_V6_JSONL,
    GAP_FILL_V7_JSONL,
]

DEFAULT_REPR_CONFIG = TextRepresentationConfig()


@dataclass(frozen=True, slots=True)
class TrainingRow:
    candidate_id: str
    source_group: str
    origin_language_slice: str
    split: str
    has_interview_text: bool
    has_transcript: bool
    motivation_question_count: int
    total_text_word_count: int
    final_recommendation: str
    final_committee_priority: int
    final_shortlist_band: bool
    final_hidden_potential_band: bool
    final_support_needed_band: bool
    final_authenticity_review_band: bool


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_payloads() -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in PAYLOAD_JSONL_PATHS:
        for record in load_jsonl(path):
            candidate_id = str(record["candidate_id"])
            if candidate_id in payloads:
                raise ValueError(f"Duplicate candidate_id in payload pool: {candidate_id}")
            payloads[candidate_id] = record
    return payloads


def load_training_rows() -> list[TrainingRow]:
    rows: list[TrainingRow] = []
    with TRAINING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                TrainingRow(
                    candidate_id=row["candidate_id"],
                    source_group=row["source_group"],
                    origin_language_slice=row["origin_language_slice"],
                    split=row["split"],
                    has_interview_text=parse_bool(row["has_interview_text"]),
                    has_transcript=parse_bool(row["has_transcript"]),
                    motivation_question_count=int(row["motivation_question_count"]),
                    total_text_word_count=int(float(row["total_text_word_count"] or 0)),
                    final_recommendation=row["final_recommendation"],
                    final_committee_priority=int(row["final_committee_priority"]),
                    final_shortlist_band=parse_bool(row["final_shortlist_band"]),
                    final_hidden_potential_band=parse_bool(row["final_hidden_potential_band"]),
                    final_support_needed_band=parse_bool(row["final_support_needed_band"]),
                    final_authenticity_review_band=parse_bool(row["final_authenticity_review_band"]),
                )
            )
    return rows


def load_batch_tasks() -> list[dict[str, Any]]:
    return load_jsonl(BATCH_JSONL)


def load_pairwise_rows() -> list[dict[str, str]]:
    with PAIRWISE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@contextmanager
def offline_runtime_guard():
    original_llm_enabled = CONFIG.llm.enabled
    original_ai_enabled = CONFIG.ai_detector.enabled
    try:
        CONFIG.llm.enabled = False
        CONFIG.ai_detector.enabled = False
        yield
    finally:
        CONFIG.llm.enabled = original_llm_enabled
        CONFIG.ai_detector.enabled = original_ai_enabled


def _representation_cache_key(config: TextRepresentationConfig) -> dict[str, Any]:
    return {
        "backend": config.backend,
        "model_name": config.model_name,
        "device": config.device,
        "top_k_mean": config.top_k_mean,
        "max_chunks_per_source": config.max_chunks_per_source,
        "max_words_per_chunk": config.max_words_per_chunk,
    }


def _baseline_ranker_score(*, merit_breakdown: dict[str, int], shortlist_signals: Any, scoring_result: Any) -> float:
    pseudo_result = SimpleNamespace(
        merit_score=scoring_result.merit_score,
        confidence_score=scoring_result.confidence_score,
        authenticity_risk=scoring_result.authenticity_risk,
        hidden_potential_score=shortlist_signals.hidden_potential_score,
        support_needed_score=shortlist_signals.support_needed_score,
        shortlist_priority_score=shortlist_signals.shortlist_priority_score,
        evidence_coverage_score=shortlist_signals.evidence_coverage_score,
        trajectory_score=shortlist_signals.trajectory_score,
        merit_breakdown=merit_breakdown,
    )
    return float(score_result_with_offline_ranker(pseudo_result))


def _slice_flags(row: TrainingRow, merged_features: dict[str, float | bool], bundle_stats: dict[str, float | int]) -> dict[str, bool]:
    logical_sources_present = int(bundle_stats.get("logical_source_groups_present", 0))
    low_polish_signal = (
        float(merged_features.get("specificity_score", 0.0)) < 0.42
        and float(merged_features.get("evidence_count", 0.0)) >= 0.18
    )
    hobby_reflective = (
        logical_sources_present <= 1
        and row.motivation_question_count == 0
        and float(merged_features.get("trajectory_reflection_score", 0.0)) >= 0.40
        and float(merged_features.get("community_value_orientation", 0.0)) < 0.42
    )
    return {
        "translated_thinking_english": row.origin_language_slice.startswith("translated"),
        "messy_or_low_polish": row.source_group.startswith("messy") or low_polish_signal,
        "single_source_application": logical_sources_present <= 1,
        "hidden_potential_slice": row.final_hidden_potential_band,
        "support_needed_slice": row.final_support_needed_band,
        "hobby_heavy_reflective": hobby_reflective,
    }


def _merged_feature_map_with_baseline(
    *,
    context: dict[str, Any],
    representation: TextRepresentationResult,
) -> tuple[dict[str, float], dict[str, float]]:
    merged_features = context["merged_features"]
    scoring_result = context["scoring_result"]
    merit_breakdown_display = context["merit_breakdown"]
    recommendation = context["recommendation_result"].recommendation
    shortlist_signals = build_shortlist_signals(
        feature_map=merged_features,
        semantic_scores={key: int(round(value * 100)) for key, value in context["semantic_snapshot"].items()},
        merit_score=scoring_result.merit_score,
        confidence_score=scoring_result.confidence_score,
        authenticity_risk=scoring_result.authenticity_risk,
        recommendation=recommendation,
    )

    feature_map: dict[str, float] = {}
    for key, value in merged_features.items():
        if isinstance(value, bool):
            feature_map[key] = 1.0 if value else 0.0
        else:
            feature_map[key] = float(value)

    feature_map.update(
        {
            "baseline_merit_raw": float(scoring_result.merit_raw),
            "baseline_confidence_raw": float(scoring_result.confidence_raw),
            "baseline_authenticity_risk_raw": float(scoring_result.authenticity_risk_raw),
            "baseline_merit_score": scoring_result.merit_score / 100.0,
            "baseline_confidence_score": scoring_result.confidence_score / 100.0,
            "baseline_authenticity_risk": scoring_result.authenticity_risk / 100.0,
            "baseline_hidden_potential_score": shortlist_signals.hidden_potential_score / 100.0,
            "baseline_support_needed_score": shortlist_signals.support_needed_score / 100.0,
            "baseline_shortlist_priority_score": shortlist_signals.shortlist_priority_score / 100.0,
            "baseline_evidence_coverage_score": shortlist_signals.evidence_coverage_score / 100.0,
            "baseline_trajectory_score": shortlist_signals.trajectory_score / 100.0,
            "baseline_recommendation_review_priority": 1.0 if str(recommendation) == "review_priority" else 0.0,
            "baseline_recommendation_manual_review_required": 1.0 if str(recommendation) == "manual_review_required" else 0.0,
        }
    )
    for key, value in scoring_result.merit_breakdown_raw.items():
        feature_map[f"baseline_axis_raw_{key}"] = float(value)
    for key, value in merit_breakdown_display.items():
        feature_map[f"baseline_axis_score_{key}"] = float(value) / 100.0
    for key, value in context["reviewer_signals"].items():
        feature_map[f"review_signal_{key}"] = float(value)
    feature_map.update(representation.feature_map)

    baseline_outputs = {
        "committee_priority_score": float(scoring_result.merit_raw),
        "shortlist_band_score": shortlist_signals.shortlist_priority_score / 100.0,
        "hidden_potential_band_score": shortlist_signals.hidden_potential_score / 100.0,
        "support_needed_band_score": shortlist_signals.support_needed_score / 100.0,
        "authenticity_review_band_score": scoring_result.authenticity_risk / 100.0,
        "pairwise_ranker_score": _baseline_ranker_score(
            merit_breakdown=merit_breakdown_display,
            shortlist_signals=shortlist_signals,
            scoring_result=scoring_result,
        ),
    }
    return feature_map, baseline_outputs


def build_or_load_text_representation_cache(
    *,
    payloads: dict[str, dict[str, Any]] | None = None,
    repr_config: TextRepresentationConfig | None = None,
    rebuild: bool = False,
) -> dict[str, TextRepresentationResult]:
    repr_config = repr_config or DEFAULT_REPR_CONFIG
    OFFLINE_LAYER_DIR.mkdir(parents=True, exist_ok=True)

    if not rebuild and REPRESENTATION_CACHE_JOBLIB.exists() and REPRESENTATION_CACHE_METADATA_JSON.exists():
        metadata = json.loads(REPRESENTATION_CACHE_METADATA_JSON.read_text(encoding="utf-8"))
        if metadata.get("representation_config") == _representation_cache_key(repr_config):
            return joblib.load(REPRESENTATION_CACHE_JOBLIB)

    payloads = payloads or load_payloads()
    pipeline = ScoringPipeline()
    cache: dict[str, TextRepresentationResult] = {}
    with offline_runtime_guard():
        for idx, candidate_id in enumerate(sorted(payloads), start=1):
            context = pipeline._prepare_scoring_context(payloads[candidate_id], enable_llm_explainability=False)
            if context.get("early_exit"):
                continue
            cache[candidate_id] = build_text_representation(context["bundle"], config=repr_config)
            if idx % 100 == 0:
                print(f"[offline-ml] Built text representation for {idx} candidates")

    joblib.dump(cache, REPRESENTATION_CACHE_JOBLIB)
    REPRESENTATION_CACHE_METADATA_JSON.write_text(
        json.dumps(
            {
                "artifact_name": "text_representation_cache_v1",
                "row_count": len(cache),
                "representation_config": _representation_cache_key(repr_config),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return cache


def build_or_load_candidate_feature_cache(
    *,
    repr_config: TextRepresentationConfig | None = None,
    rebuild: bool = False,
) -> list[dict[str, Any]]:
    repr_config = repr_config or DEFAULT_REPR_CONFIG
    OFFLINE_LAYER_DIR.mkdir(parents=True, exist_ok=True)

    if not rebuild and FEATURE_CACHE_JOBLIB.exists() and FEATURE_CACHE_METADATA_JSON.exists():
        metadata = json.loads(FEATURE_CACHE_METADATA_JSON.read_text(encoding="utf-8"))
        if metadata.get("representation_config") == _representation_cache_key(repr_config):
            return joblib.load(FEATURE_CACHE_JOBLIB)

    payloads = load_payloads()
    rows = load_training_rows()
    representations = build_or_load_text_representation_cache(
        payloads=payloads,
        repr_config=repr_config,
        rebuild=rebuild,
    )

    pipeline = ScoringPipeline()
    cached_rows: list[dict[str, Any]] = []
    with offline_runtime_guard():
        for idx, row in enumerate(rows, start=1):
            payload = payloads.get(row.candidate_id)
            if payload is None:
                raise ValueError(f"Missing payload for training candidate {row.candidate_id}")
            context = pipeline._prepare_scoring_context(payload, enable_llm_explainability=False)
            if context.get("early_exit"):
                continue
            representation = representations[row.candidate_id]
            feature_map, baseline_outputs = _merged_feature_map_with_baseline(
                context=context,
                representation=representation,
            )
            cached_rows.append(
                {
                    "candidate_id": row.candidate_id,
                    "split": row.split,
                    "source_group": row.source_group,
                    "origin_language_slice": row.origin_language_slice,
                    "labels": {
                        "final_recommendation": row.final_recommendation,
                        "final_committee_priority": row.final_committee_priority,
                        "final_shortlist_band": row.final_shortlist_band,
                        "final_hidden_potential_band": row.final_hidden_potential_band,
                        "final_support_needed_band": row.final_support_needed_band,
                        "final_authenticity_review_band": row.final_authenticity_review_band,
                    },
                    "slices": _slice_flags(row, context["merged_features"], context["bundle"].stats),
                    "feature_map": feature_map,
                    "baseline_outputs": baseline_outputs,
                    "representation_diagnostics": representation.diagnostics,
                }
            )
            if idx % 100 == 0:
                print(f"[offline-ml] Prepared candidate feature rows for {idx} labels")

    feature_names = sorted({key for item in cached_rows for key in item["feature_map"].keys()})
    FEATURE_CACHE_METADATA_JSON.write_text(
        json.dumps(
            {
                "artifact_name": "offline_candidate_feature_cache_v1",
                "row_count": len(cached_rows),
                "feature_count": len(feature_names),
                "representation_config": _representation_cache_key(repr_config),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    joblib.dump(cached_rows, FEATURE_CACHE_JOBLIB)
    return cached_rows


def feature_names_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({key for item in rows for key in item["feature_map"].keys()})


def build_feature_row_for_payload(
    candidate_payload: dict[str, Any],
    *,
    repr_config: TextRepresentationConfig | None = None,
) -> dict[str, Any]:
    repr_config = repr_config or DEFAULT_REPR_CONFIG
    pipeline = ScoringPipeline()
    with offline_runtime_guard():
        context = pipeline._prepare_scoring_context(candidate_payload, enable_llm_explainability=False)
        if context.get("early_exit"):
            raise ValueError("payload_not_eligible_for_offline_feature_row")
        representation = build_text_representation(context["bundle"], config=repr_config)
        feature_map, baseline_outputs = _merged_feature_map_with_baseline(
            context=context,
            representation=representation,
        )
        return {
            "candidate_id": str(context["projected"].get("candidate_id", "")),
            "feature_map": feature_map,
            "baseline_outputs": baseline_outputs,
            "representation_diagnostics": representation.diagnostics,
            "raw_recommendation": str(context["recommendation_result"].recommendation),
        }
