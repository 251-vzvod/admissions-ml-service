"""English-first validation slices for shortlist fairness and ranking robustness.

This script focuses on the current practical deployment assumption:
most real applicant text will be in English.

It evaluates the scorer on:
- polished vs plain English presentation slices
- verbose vs concise English slices
- evidence-strong vs evidence-thin English slices
- transcript-present vs transcript-absent English slices

It complements the existing overall evaluation pack with more targeted,
English-first fairness and robustness views.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.annotation_eval import build_label_evaluation, load_annotations
from app.services.pipeline import ScoringPipeline
from app.services.preprocessing import preprocess_text_inputs


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("JSON must contain a 'candidates' list")
    return candidates


def _mean(values: list[int | float]) -> float:
    return round(float(statistics.mean(values)), 3) if values else 0.0


def _quantile_bucket(value: float, lower: float, upper: float, low_label: str, mid_label: str, high_label: str) -> str:
    if value <= lower:
        return low_label
    if value >= upper:
        return high_label
    return mid_label


def _quantile_cutoffs(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    lower_idx = max(0, int(round((len(ordered) - 1) * 0.33)))
    upper_idx = max(0, int(round((len(ordered) - 1) * 0.67)))
    return ordered[lower_idx], ordered[upper_idx]


def _english_first_candidate(candidate: dict[str, Any]) -> bool:
    text_inputs = candidate.get("text_inputs", {})
    if not isinstance(text_inputs, dict):
        return False
    bundle = preprocess_text_inputs(text_inputs=text_inputs)
    content_profile = candidate.get("content_profile", {})
    if not isinstance(content_profile, dict):
        content_profile = {}
    language_profile = str(content_profile.get("language_profile", "")).strip().lower()
    latin_share = float(bundle.stats.get("latin_text_share", 0.0))
    cyrillic_share = float(bundle.stats.get("cyrillic_text_share", 0.0))
    if language_profile == "english":
        return True
    if language_profile == "mixed" and latin_share >= cyrillic_share:
        return True
    return latin_share >= 0.65


def _word_count(candidate: dict[str, Any]) -> int:
    text_inputs = candidate.get("text_inputs", {})
    if not isinstance(text_inputs, dict):
        return 0
    bundle = preprocess_text_inputs(text_inputs=text_inputs)
    return int(bundle.stats.get("word_count", 0))


def _has_transcript(candidate: dict[str, Any]) -> bool:
    text_inputs = candidate.get("text_inputs", {})
    if not isinstance(text_inputs, dict):
        return False
    return any(
        bool(text_inputs.get(key))
        for key in ("interview_text", "video_interview_transcript_text", "video_presentation_transcript_text")
    )


def _presentation_metric(trace: dict[str, Any]) -> float:
    text_features = trace.get("text_features", {})
    if not isinstance(text_features, dict):
        return 0.0
    return (
        (float(text_features.get("motivation_clarity", 0.0)) * 0.34)
        + (float(text_features.get("program_fit", 0.0)) * 0.18)
        + ((1.0 - float(text_features.get("genericness_score", 0.0))) * 0.18)
        + ((1.0 - float(text_features.get("polished_but_empty_score", 0.0))) * 0.12)
        + (float(text_features.get("completeness_score", 0.0)) * 0.18)
    )


def _evidence_metric(trace: dict[str, Any]) -> float:
    text_features = trace.get("text_features", {})
    if not isinstance(text_features, dict):
        return 0.0
    return (
        (float(text_features.get("evidence_count", 0.0)) * 0.45)
        + (float(text_features.get("specificity_score", 0.0)) * 0.30)
        + (float(text_features.get("consistency_score", 0.0)) * 0.25)
    )


def _transcript_slice(candidate: dict[str, Any]) -> str:
    return "transcript_present" if _has_transcript(candidate) else "transcript_absent"


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    top_k = min(max(len(rows) // 5, 1), len(rows))
    ranked = sorted(rows, key=lambda item: int(item["merit_score"]), reverse=True)
    return {
        "count": len(rows),
        "avg_merit_score": _mean([int(item["merit_score"]) for item in rows]),
        "avg_confidence_score": _mean([int(item["confidence_score"]) for item in rows]),
        "avg_authenticity_risk": _mean([int(item["authenticity_risk"]) for item in rows]),
        "avg_hidden_potential_score": _mean([int(item.get("hidden_potential_score", 0)) for item in rows]),
        "avg_shortlist_priority_score": _mean([int(item.get("shortlist_priority_score", 0)) for item in rows]),
        "top_k": top_k,
        "top_k_candidate_ids": [item["candidate_id"] for item in ranked[:top_k]],
        "recommendation_distribution": dict(Counter(item["recommendation"] for item in rows)),
    }


def build_english_first_validation_report(
    candidates: list[dict[str, Any]],
    annotations_path: Path,
) -> dict[str, Any]:
    annotations = load_annotations(annotations_path)
    pipeline = ScoringPipeline()

    english_candidates = [candidate for candidate in candidates if _english_first_candidate(candidate)]
    score_rows: list[dict[str, Any]] = []
    trace_rows: dict[str, dict[str, Any]] = {}
    candidate_by_id: dict[str, dict[str, Any]] = {}
    presentation_metrics: dict[str, float] = {}
    verbosity_metrics: dict[str, float] = {}
    evidence_metrics: dict[str, float] = {}

    for candidate in english_candidates:
        result = pipeline.score_candidate(candidate, enable_llm_explainability=False).model_dump()
        trace = pipeline.score_candidate_trace(candidate)
        cid = str(result["candidate_id"])
        score_rows.append(result)
        trace_rows[cid] = trace
        candidate_by_id[cid] = candidate
        presentation_metrics[cid] = _presentation_metric(trace)
        verbosity_metrics[cid] = float(_word_count(candidate))
        evidence_metrics[cid] = _evidence_metric(trace)

    presentation_lower, presentation_upper = _quantile_cutoffs(list(presentation_metrics.values()))
    verbosity_lower, verbosity_upper = _quantile_cutoffs(list(verbosity_metrics.values()))
    evidence_lower, evidence_upper = _quantile_cutoffs(list(evidence_metrics.values()))

    slices: dict[str, dict[str, list[dict[str, Any]]]] = {
        "presentation_style": defaultdict(list),
        "verbosity": defaultdict(list),
        "evidence_profile": defaultdict(list),
        "transcript_presence": defaultdict(list),
    }

    for row in score_rows:
        cid = str(row["candidate_id"])
        slices["presentation_style"][
            _quantile_bucket(
                presentation_metrics[cid],
                presentation_lower,
                presentation_upper,
                "plain",
                "middle",
                "polished",
            )
        ].append(row)
        slices["verbosity"][
            _quantile_bucket(
                verbosity_metrics[cid],
                verbosity_lower,
                verbosity_upper,
                "concise",
                "medium",
                "verbose",
            )
        ].append(row)
        slices["evidence_profile"][
            _quantile_bucket(
                evidence_metrics[cid],
                evidence_lower,
                evidence_upper,
                "evidence_thin",
                "evidence_middle",
                "evidence_strong",
            )
        ].append(row)
        slices["transcript_presence"][_transcript_slice(candidate_by_id[cid])].append(row)

    slice_reports: dict[str, dict[str, Any]] = {}
    for axis, groups in slices.items():
        axis_report: dict[str, Any] = {}
        for label, rows in sorted(groups.items()):
            axis_report[label] = {
                "summary": _summarize_rows(rows),
                "label_alignment": build_label_evaluation(rows, annotations),
            }
        slice_reports[axis] = axis_report

    return {
        "meta": {
            "candidate_count_total": len(candidates),
            "english_first_candidate_count": len(english_candidates),
            "annotations_path": str(annotations_path),
            "focus": "english_first_shortlist_fairness_and_validation",
        },
        "slices": slice_reports,
        "notes": [
            "This report intentionally focuses on English-first shortlist behavior rather than multilingual expansion.",
            "Fairness here is treated primarily as presentation-style bias within English, not only language-family bias.",
            "Slice-level results should be read together with the overall evaluation pack and family-aware validation.",
        ],
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# English-First Validation",
        "",
        "## Scope",
        "",
        f"- total candidates: `{report['meta']['candidate_count_total']}`",
        f"- english-first candidates analyzed: `{report['meta']['english_first_candidate_count']}`",
        "- focus: `polish bias + shortlist robustness within English-first inputs`",
        "",
    ]

    for axis, groups in report["slices"].items():
        lines.extend([f"## {axis.replace('_', ' ').title()}", ""])
        for label, payload in groups.items():
            summary = payload["summary"]
            alignment = payload["label_alignment"]
            lines.extend(
                [
                    f"### {label}",
                    "",
                    f"- count: `{summary.get('count', 0)}`",
                    f"- avg merit: `{summary.get('avg_merit_score', 0)}`",
                    f"- avg confidence: `{summary.get('avg_confidence_score', 0)}`",
                    f"- avg authenticity risk: `{summary.get('avg_authenticity_risk', 0)}`",
                    f"- avg hidden potential: `{summary.get('avg_hidden_potential_score', 0)}`",
                    f"- avg shortlist priority: `{summary.get('avg_shortlist_priority_score', 0)}`",
                    f"- label pairwise accuracy: `{alignment.get('pairwise_accuracy', 'n/a')}`",
                    f"- label precision@k priority: `{alignment.get('precision_at_k_priority', 'n/a')}`",
                    f"- hidden potential recall@k: `{alignment.get('hidden_potential_recall_at_k', 'n/a')}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Use this report to detect whether shortlist behavior changes materially across English-first presentation slices.",
            "- The most important comparison is no longer `RU vs EN`, but `polished vs plain`, `verbose vs concise`, and `evidence-strong vs evidence-thin`.",
            "- Any future mitigation should improve these slices without making the scorer opaque.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run English-first shortlist fairness validation")
    parser.add_argument("--input", default="research/data/candidates_expanded_v1.json")
    parser.add_argument("--annotations", default="research/data/final_hackathon_annotations_v1.json")
    parser.add_argument("--output-json", default="research/reports/english_first_validation.json")
    parser.add_argument("--output-md", default="research/reports/english_first_validation.md")
    args = parser.parse_args()

    candidates = load_candidates(Path(args.input))
    report = build_english_first_validation_report(candidates, Path(args.annotations))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"English-first validation JSON -> {output_json}")
    print(f"English-first validation markdown -> {output_md}")


if __name__ == "__main__":
    main()
