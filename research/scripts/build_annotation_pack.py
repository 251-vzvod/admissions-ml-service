"""Build committee annotation pack for ranking-oriented baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("JSON must contain a 'candidates' list")
    return candidates


def build_annotation_pack(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    annotations = []
    for candidate in candidates:
        text_inputs = candidate.get("text_inputs", {})
        if not isinstance(text_inputs, dict):
            text_inputs = {}

        annotations.append(
            {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "scenario_meta": candidate.get("scenario_meta", {}),
                "content_profile": candidate.get("content_profile", {}),
                "rubric": {
                    "leadership_potential": None,
                    "growth_trajectory": None,
                    "motivation_authenticity": None,
                    "evidence_strength": None,
                    "committee_priority": None,
                    "hidden_potential_flag": False,
                    "needs_support_flag": False,
                    "authenticity_review_flag": False,
                    "review_notes": "",
                },
                "evidence_payload": {
                    "motivation_letter_text": text_inputs.get("motivation_letter_text", ""),
                    "motivation_questions": text_inputs.get("motivation_questions", []),
                    "interview_text": text_inputs.get("interview_text", ""),
                    "video_interview_transcript_text": text_inputs.get("video_interview_transcript_text", ""),
                    "video_presentation_transcript_text": text_inputs.get("video_presentation_transcript_text", ""),
                },
            }
        )

    return {
        "annotation_instructions": {
            "scale": "Use 1..5 where 1=very weak and 5=very strong.",
            "committee_priority_definition": "How strongly should this candidate be prioritized for human review/interview.",
            "hidden_potential_flag_definition": "True when growth, resilience, initiative, or values are strong despite weak self-presentation or formal profile.",
            "needs_support_flag_definition": "True when candidate seems promising but may require language, academic, or onboarding support.",
            "authenticity_review_flag_definition": "True when text should be manually checked for genericness or unsupported claims.",
        },
        "annotations": annotations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build annotation pack for committee review")
    parser.add_argument("--input", default="data/archive/candidates.json", help="Path to candidates JSON")
    parser.add_argument(
        "--output",
        default="data/archive/annotation_pack.json",
        help="Where to write annotation pack JSON",
    )
    args = parser.parse_args()

    candidates = load_candidates(Path(args.input))
    output = build_annotation_pack(candidates)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Annotation pack generated: {args.output}")


if __name__ == "__main__":
    main()
