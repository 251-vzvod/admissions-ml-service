"""Stress-test shortlist stability on English-first candidate perturbations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.pipeline import ScoringPipeline
from app.services.preprocessing import preprocess_text_inputs
from app.utils.text import normalize_whitespace


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("JSON must contain a 'candidates' list")
    return candidates


def _mean(values: list[int | float]) -> float:
    return round(float(statistics.mean(values)), 3) if values else 0.0


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


def _first_sentence(text: str) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""
    parts = cleaned.split(". ")
    return parts[0].strip() + ("." if parts and not parts[0].endswith(".") else "")


def _concise_variant(candidate: dict[str, Any]) -> dict[str, Any]:
    variant = deepcopy(candidate)
    text_inputs = variant.setdefault("text_inputs", {})
    if isinstance(text_inputs.get("motivation_letter_text"), str):
        text_inputs["motivation_letter_text"] = _first_sentence(text_inputs["motivation_letter_text"])
    qas = text_inputs.get("motivation_questions")
    if isinstance(qas, list) and qas:
        first = qas[0]
        if isinstance(first, dict):
            text_inputs["motivation_questions"] = [
                {
                    "question": first.get("question"),
                    "answer": _first_sentence(str(first.get("answer") or "")),
                }
            ]
    return variant


def _polished_wrapper_variant(candidate: dict[str, Any]) -> dict[str, Any]:
    variant = deepcopy(candidate)
    text_inputs = variant.setdefault("text_inputs", {})
    wrapper = (
        "I care deeply about meaningful impact, collaborative growth, and responsible leadership. "
        "I try to turn challenges into structured learning and long-term value for others. "
    )
    if isinstance(text_inputs.get("motivation_letter_text"), str):
        text_inputs["motivation_letter_text"] = wrapper + text_inputs["motivation_letter_text"]
    return variant


def _evidence_removed_variant(candidate: dict[str, Any]) -> dict[str, Any]:
    variant = deepcopy(candidate)
    text_inputs = variant.setdefault("text_inputs", {})
    if isinstance(text_inputs.get("motivation_letter_text"), str):
        text_inputs["motivation_letter_text"] = (
            "I tried to help and improve things, but I cannot provide many concrete details here."
        )
    qas = text_inputs.get("motivation_questions")
    if isinstance(qas, list):
        rebuilt = []
        for qa in qas[:2]:
            if isinstance(qa, dict):
                rebuilt.append(
                    {
                        "question": qa.get("question"),
                        "answer": "I was motivated to contribute and learn, but the example is described only at a high level.",
                    }
                )
        text_inputs["motivation_questions"] = rebuilt
    return variant


def _transcript_removed_variant(candidate: dict[str, Any]) -> dict[str, Any]:
    variant = deepcopy(candidate)
    text_inputs = variant.setdefault("text_inputs", {})
    for key in ("interview_text", "video_interview_transcript_text", "video_presentation_transcript_text"):
        if key in text_inputs:
            text_inputs[key] = ""
    return variant


PERTURBATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "concise": _concise_variant,
    "polished_wrapper": _polished_wrapper_variant,
    "evidence_removed": _evidence_removed_variant,
    "transcript_removed": _transcript_removed_variant,
}


def _pick_focus_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    english = [candidate for candidate in candidates if _english_first_candidate(candidate)]
    return english[: min(20, len(english))]


def build_stress_test_report(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    pipeline = ScoringPipeline()
    focus_candidates = _pick_focus_candidates(candidates)

    per_perturbation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aggregate_shortlist_shift: dict[str, Counter[str]] = defaultdict(Counter)

    for candidate in focus_candidates:
        base = pipeline.score_candidate(candidate, enable_llm_explainability=False).model_dump()
        base_cohorts = set(base.get("committee_cohorts", []))
        base_top_claims = base.get("supported_claims", []) or base.get("weakly_supported_claims", [])

        for name, builder in PERTURBATIONS.items():
            variant = builder(candidate)
            variant_result = pipeline.score_candidate(variant, enable_llm_explainability=False).model_dump()
            aggregate_shortlist_shift[name]["cohort_changed"] += int(set(variant_result.get("committee_cohorts", [])) != base_cohorts)
            aggregate_shortlist_shift[name]["claim_layer_changed"] += int(
                (variant_result.get("supported_claims", []) or variant_result.get("weakly_supported_claims", [])) != base_top_claims
            )

            per_perturbation[name].append(
                {
                    "candidate_id": base["candidate_id"],
                    "base_shortlist_priority": int(base["shortlist_priority_score"]),
                    "variant_shortlist_priority": int(variant_result["shortlist_priority_score"]),
                    "base_hidden_potential": int(base["hidden_potential_score"]),
                    "variant_hidden_potential": int(variant_result["hidden_potential_score"]),
                    "base_confidence": int(base["confidence_score"]),
                    "variant_confidence": int(variant_result["confidence_score"]),
                    "base_authenticity_risk": int(base["authenticity_risk"]),
                    "variant_authenticity_risk": int(variant_result["authenticity_risk"]),
                    "base_recommendation": base["recommendation"],
                    "variant_recommendation": variant_result["recommendation"],
                    "shortlist_priority_delta": int(variant_result["shortlist_priority_score"]) - int(base["shortlist_priority_score"]),
                    "hidden_potential_delta": int(variant_result["hidden_potential_score"]) - int(base["hidden_potential_score"]),
                    "confidence_delta": int(variant_result["confidence_score"]) - int(base["confidence_score"]),
                    "authenticity_risk_delta": int(variant_result["authenticity_risk"]) - int(base["authenticity_risk"]),
                }
            )

    report: dict[str, Any] = {
        "meta": {
            "focus_candidate_count": len(focus_candidates),
            "perturbations": list(PERTURBATIONS.keys()),
            "focus": "english_first_stress_test",
        },
        "perturbation_summary": {},
        "notes": [
            "This report tests how strongly shortlist behavior changes when the same English-first candidate is presented differently.",
            "The goal is not to freeze scores completely, but to detect over-sensitivity to style, verbosity, missing transcript, or weakened evidence.",
        ],
    }

    for name, rows in per_perturbation.items():
        report["perturbation_summary"][name] = {
            "count": len(rows),
            "avg_shortlist_priority_delta": _mean([row["shortlist_priority_delta"] for row in rows]),
            "avg_hidden_potential_delta": _mean([row["hidden_potential_delta"] for row in rows]),
            "avg_confidence_delta": _mean([row["confidence_delta"] for row in rows]),
            "avg_authenticity_risk_delta": _mean([row["authenticity_risk_delta"] for row in rows]),
            "recommendation_change_rate": round(
                sum(1 for row in rows if row["base_recommendation"] != row["variant_recommendation"]) / max(len(rows), 1),
                3,
            ),
            "cohort_change_rate": round(
                aggregate_shortlist_shift[name]["cohort_changed"] / max(len(rows), 1),
                3,
            ),
            "claim_layer_change_rate": round(
                aggregate_shortlist_shift[name]["claim_layer_changed"] / max(len(rows), 1),
                3,
            ),
            "examples": rows[:5],
        }

    return report


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# English-First Stress Test",
        "",
        "## Scope",
        "",
        f"- focus candidates: `{report['meta']['focus_candidate_count']}`",
        f"- perturbations: `{', '.join(report['meta']['perturbations'])}`",
        "",
    ]
    for name, summary in report["perturbation_summary"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- avg shortlist priority delta: `{summary['avg_shortlist_priority_delta']}`",
                f"- avg hidden potential delta: `{summary['avg_hidden_potential_delta']}`",
                f"- avg confidence delta: `{summary['avg_confidence_delta']}`",
                f"- avg authenticity risk delta: `{summary['avg_authenticity_risk_delta']}`",
                f"- recommendation change rate: `{summary['recommendation_change_rate']}`",
                f"- cohort change rate: `{summary['cohort_change_rate']}`",
                f"- claim layer change rate: `{summary['claim_layer_change_rate']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- `concise` tests whether shorter English answers are punished too aggressively.",
            "- `polished_wrapper` tests sensitivity to presentation polish alone.",
            "- `evidence_removed` tests whether the system appropriately drops confidence and shortlist priority when grounding disappears.",
            "- `transcript_removed` tests reliance on transcript channels.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run English-first shortlist stress tests")
    parser.add_argument("--input", default="data/candidates_expanded_v1.json")
    parser.add_argument("--output-json", default="data/evaluation_pack_final_hackathon_v3/english_first_stress_test.json")
    parser.add_argument("--output-md", default="docs/english_first_stress_test.md")
    args = parser.parse_args()

    candidates = load_candidates(Path(args.input))
    report = build_stress_test_report(candidates)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"English-first stress test JSON -> {output_json}")
    print(f"English-first stress test markdown -> {output_md}")


if __name__ == "__main__":
    main()
