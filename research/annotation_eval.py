"""Utilities for comparing scored candidates to human annotations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CandidateAnnotation:
    candidate_id: str
    leadership_potential: int
    growth_trajectory: int
    motivation_authenticity: int
    evidence_strength: int
    committee_priority: int
    hidden_potential_flag: bool = False
    needs_support_flag: bool = False
    authenticity_review_flag: bool = False


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0

    while idx < len(ordered):
        end = idx
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[idx][1]:
            end += 1
        avg_rank = (idx + end + 2) / 2.0
        for rank_idx in range(idx, end + 1):
            original_index = ordered[rank_idx][0]
            ranks[original_index] = avg_rank
        idx = end + 1

    return ranks


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    numerator = sum(x_val * y_val for x_val, y_val in zip(centered_x, centered_y, strict=False))
    denom_x = sqrt(sum(x_val * x_val for x_val in centered_x))
    denom_y = sqrt(sum(y_val * y_val for y_val in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return round(numerator / (denom_x * denom_y), 6)


def _spearman_correlation(xs: list[float], ys: list[float]) -> float:
    return _pearson_correlation(_average_ranks(xs), _average_ranks(ys))


def _pairwise_accuracy(predicted_scores: list[float], target_scores: list[float]) -> float:
    correct = 0
    total = 0
    for left_idx in range(len(predicted_scores)):
        for right_idx in range(left_idx + 1, len(predicted_scores)):
            target_delta = target_scores[left_idx] - target_scores[right_idx]
            if target_delta == 0:
                continue
            predicted_delta = predicted_scores[left_idx] - predicted_scores[right_idx]
            total += 1
            if predicted_delta == 0:
                continue
            if (predicted_delta > 0) == (target_delta > 0):
                correct += 1
    return round(correct / total, 6) if total else 0.0


def _hidden_potential_recall_at_k(
    candidate_ids: list[str],
    predicted_scores: list[float],
    annotations: dict[str, CandidateAnnotation],
) -> float:
    positives = [candidate_id for candidate_id in candidate_ids if annotations[candidate_id].hidden_potential_flag]
    if not positives:
        return 0.0

    top_k = len(positives)
    ranked_ids = [
        candidate_id
        for candidate_id, _score in sorted(
            zip(candidate_ids, predicted_scores, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    hits = sum(1 for candidate_id in ranked_ids[:top_k] if candidate_id in positives)
    return round(hits / len(positives), 6)


def _human_label_score(annotation: CandidateAnnotation) -> float:
    return (
        (annotation.committee_priority * 0.34)
        + (annotation.leadership_potential * 0.18)
        + (annotation.growth_trajectory * 0.18)
        + (annotation.motivation_authenticity * 0.16)
        + (annotation.evidence_strength * 0.14)
    )


def _semantic_alignment(
    scored_items: Iterable[dict[str, object]],
    annotations: dict[str, CandidateAnnotation],
) -> dict[str, float]:
    dimension_mappings = {
        "leadership_potential": "leadership_potential",
        "growth_trajectory": "growth_trajectory",
        "motivation_authenticity": "motivation_authenticity",
        "authenticity_groundedness": "evidence_strength",
        "hidden_potential": "hidden_potential_flag",
    }
    alignment: dict[str, float] = {}

    for semantic_dimension, annotation_field in dimension_mappings.items():
        semantic_scores: list[float] = []
        human_scores: list[float] = []
        for item in scored_items:
            candidate_id = str(item.get("candidate_id") or "")
            annotation = annotations.get(candidate_id)
            if annotation is None:
                continue
            semantic = item.get("semantic_rubric_scores")
            if not isinstance(semantic, dict):
                continue
            semantic_scores.append(float(semantic.get(semantic_dimension, 0.0)))
            human_value = getattr(annotation, annotation_field)
            human_scores.append(float(int(human_value)) if isinstance(human_value, bool) else float(human_value))

        alignment[f"{semantic_dimension}_spearman"] = _spearman_correlation(
            semantic_scores,
            human_scores,
        )

    return alignment


def build_label_evaluation(
    scored_candidates: list[dict[str, object]],
    annotations: dict[str, CandidateAnnotation],
) -> dict[str, object]:
    annotated_scored = [
        candidate
        for candidate in scored_candidates
        if str(candidate.get("candidate_id") or "") in annotations
    ]
    candidate_ids = [str(candidate["candidate_id"]) for candidate in annotated_scored]
    merit_scores = [float(candidate.get("merit_score", 0.0)) for candidate in annotated_scored]
    human_scores = [_human_label_score(annotations[candidate_id]) for candidate_id in candidate_ids]

    return {
        "annotated_candidate_count": len(annotated_scored),
        "spearman_merit_vs_labels": _spearman_correlation(merit_scores, human_scores),
        "pairwise_accuracy": _pairwise_accuracy(merit_scores, human_scores),
        "hidden_potential_recall_at_k": _hidden_potential_recall_at_k(candidate_ids, merit_scores, annotations),
        "semantic_dimension_alignment": _semantic_alignment(annotated_scored, annotations),
    }

