"""Utilities for evaluating ranking quality against committee annotations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any

from app.utils.math_utils import safe_div


ANNOTATION_DIMENSIONS = (
    "leadership_potential",
    "growth_trajectory",
    "motivation_authenticity",
    "evidence_strength",
    "committee_priority",
)


@dataclass(slots=True)
class CandidateAnnotation:
    candidate_id: str
    leadership_potential: float | None = None
    growth_trajectory: float | None = None
    motivation_authenticity: float | None = None
    evidence_strength: float | None = None
    committee_priority: float | None = None
    hidden_potential_flag: bool = False
    needs_support_flag: bool = False
    authenticity_review_flag: bool = False

    @property
    def composite_label(self) -> float:
        if self.committee_priority is not None:
            return float(self.committee_priority)

        weighted_sum = 0.0
        total_weight = 0.0
        weights = {
            "leadership_potential": 0.30,
            "growth_trajectory": 0.25,
            "motivation_authenticity": 0.20,
            "evidence_strength": 0.25,
        }
        for key, weight in weights.items():
            value = getattr(self, key)
            if value is None:
                continue
            weighted_sum += float(value) * weight
            total_weight += weight
        return safe_div(weighted_sum, total_weight, default=0.0)


def load_annotations(path: str | Path) -> dict[str, CandidateAnnotation]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("annotations", [])
    if not isinstance(items, list):
        raise ValueError("annotations file must contain an 'annotations' list")

    annotations: dict[str, CandidateAnnotation] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        rubric = item.get("rubric", {})
        if not isinstance(rubric, dict):
            rubric = {}
        annotations[candidate_id] = CandidateAnnotation(
            candidate_id=candidate_id,
            leadership_potential=_to_optional_float(item.get("leadership_potential", rubric.get("leadership_potential"))),
            growth_trajectory=_to_optional_float(item.get("growth_trajectory", rubric.get("growth_trajectory"))),
            motivation_authenticity=_to_optional_float(item.get("motivation_authenticity", rubric.get("motivation_authenticity"))),
            evidence_strength=_to_optional_float(item.get("evidence_strength", rubric.get("evidence_strength"))),
            committee_priority=_to_optional_float(item.get("committee_priority", rubric.get("committee_priority"))),
            hidden_potential_flag=bool(item.get("hidden_potential_flag", rubric.get("hidden_potential_flag", False))),
            needs_support_flag=bool(item.get("needs_support_flag", rubric.get("needs_support_flag", False))),
            authenticity_review_flag=bool(item.get("authenticity_review_flag", rubric.get("authenticity_review_flag", False))),
        )
    return annotations


def build_label_evaluation(scored: list[dict[str, Any]], annotations: dict[str, CandidateAnnotation]) -> dict[str, Any]:
    aligned = [
        (row, annotations[row["candidate_id"]])
        for row in scored
        if str(row.get("candidate_id", "")) in annotations
    ]
    if not aligned:
        return {"status": "no_overlap_with_annotations"}

    score_values = [float(row["merit_score"]) for row, _ in aligned]
    label_values = [annotation.composite_label for _, annotation in aligned]

    top_k = min(max(len(aligned) // 5, 1), len(aligned))
    hidden_positive_ids = {annotation.candidate_id for _, annotation in aligned if annotation.hidden_potential_flag}
    priority_positive_ids = {
        annotation.candidate_id for _, annotation in aligned if (annotation.committee_priority or annotation.composite_label) >= 4.0
    }
    ranked_ids = [
        row["candidate_id"]
        for row, _ in sorted(aligned, key=lambda item: float(item[0]["merit_score"]), reverse=True)
    ]
    top_k_ids = set(ranked_ids[:top_k])

    return {
        "annotated_candidate_count": len(aligned),
        "top_k": top_k,
        "spearman_merit_vs_labels": round(_spearman(score_values, label_values), 4),
        "pairwise_accuracy": round(_pairwise_accuracy(score_values, label_values), 4),
        "precision_at_k_priority": round(
            safe_div(sum(1 for cid in ranked_ids[:top_k] if cid in priority_positive_ids), top_k, default=0.0),
            4,
        ),
        "hidden_potential_recall_at_k": round(
            safe_div(sum(1 for cid in hidden_positive_ids if cid in top_k_ids), len(hidden_positive_ids), default=0.0),
            4,
        ),
        "support_flag_rate_in_top_k": round(
            safe_div(
                sum(1 for cid in ranked_ids[:top_k] if annotations[cid].needs_support_flag),
                top_k,
                default=0.0,
            ),
            4,
        ),
        "semantic_dimension_alignment": _semantic_dimension_alignment(aligned),
    }


def _to_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for original_index, _ in indexed[idx:end]:
            ranks[original_index] = avg_rank
        idx = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def _pairwise_accuracy(score_values: list[float], label_values: list[float]) -> float:
    comparisons = 0
    correct = 0
    for left, right in combinations(range(len(score_values)), 2):
        label_delta = label_values[left] - label_values[right]
        if label_delta == 0:
            continue
        score_delta = score_values[left] - score_values[right]
        comparisons += 1
        if (score_delta > 0 and label_delta > 0) or (score_delta < 0 and label_delta < 0):
            correct += 1
    return safe_div(correct, comparisons, default=0.0)


def _semantic_dimension_alignment(
    aligned: list[tuple[dict[str, Any], CandidateAnnotation]]
) -> dict[str, float]:
    dimension_map = {
        "leadership_potential": "leadership_potential",
        "growth_trajectory": "growth_trajectory",
        "motivation_authenticity": "motivation_authenticity",
        "authenticity_groundedness": None,
        "hidden_potential": None,
    }

    report: dict[str, float] = {}
    for semantic_key, annotation_key in dimension_map.items():
        xs: list[float] = []
        ys: list[float] = []
        for row, annotation in aligned:
            semantic_scores = row.get("semantic_rubric_scores", {})
            if not isinstance(semantic_scores, dict) or semantic_key not in semantic_scores:
                continue
            xs.append(float(semantic_scores[semantic_key]))
            if annotation_key is None:
                if semantic_key == "hidden_potential":
                    ys.append(5.0 if annotation.hidden_potential_flag else 1.0)
                else:
                    ys.append(5.0 if not annotation.authenticity_review_flag else 1.0)
            else:
                value = getattr(annotation, annotation_key)
                if value is None:
                    xs.pop()
                    continue
                ys.append(float(value))

        if xs and ys:
            report[f"{semantic_key}_spearman"] = round(_spearman(xs, ys), 4)
    return report
