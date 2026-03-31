"""Build family-aware validation artifacts for synthetic candidate sets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.annotation_eval import CandidateAnnotation, build_label_evaluation, load_annotations


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("JSON must contain a 'candidates' list")
    return candidates


def load_scored(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("scored results JSON must contain a 'results' list")
    return results


def family_id_for_candidate(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    derived = metadata.get("derived_from_candidate_id")
    candidate_id = str(candidate.get("candidate_id", "")).strip()
    return str(derived).strip() if isinstance(derived, str) and derived.strip() else candidate_id


def choose_root_representative(family_id: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    for member in members:
        if str(member.get("candidate_id", "")).strip() == family_id:
            return member
    return sorted(members, key=lambda item: str(item.get("candidate_id", "")))[0]


def aggregate_family_annotations(
    family_members: list[dict[str, Any]],
    annotations: dict[str, CandidateAnnotation],
) -> CandidateAnnotation:
    present = [annotations[str(item.get("candidate_id", ""))] for item in family_members if str(item.get("candidate_id", "")) in annotations]
    if not present:
        raise ValueError("family has no annotations")

    def _max_optional(values: list[float | None]) -> float | None:
        present_values = [float(value) for value in values if value is not None]
        if not present_values:
            return None
        return max(present_values)

    family_id = family_id_for_candidate(family_members[0])
    return CandidateAnnotation(
        candidate_id=family_id,
        leadership_potential=_max_optional([item.leadership_potential for item in present]),
        growth_trajectory=_max_optional([item.growth_trajectory for item in present]),
        motivation_authenticity=_max_optional([item.motivation_authenticity for item in present]),
        evidence_strength=_max_optional([item.evidence_strength for item in present]),
        committee_priority=_max_optional([item.committee_priority for item in present]),
        hidden_potential_flag=any(item.hidden_potential_flag for item in present),
        needs_support_flag=any(item.needs_support_flag for item in present),
        authenticity_review_flag=any(item.authenticity_review_flag for item in present),
    )


def build_family_aware_report(
    candidates: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    annotations: dict[str, CandidateAnnotation],
) -> dict[str, Any]:
    candidates_by_id = {str(candidate.get("candidate_id", "")): candidate for candidate in candidates}
    scored_by_id = {str(row.get("candidate_id", "")): row for row in scored}

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        families[family_id_for_candidate(candidate)].append(candidate)

    family_sizes = Counter({family_id: len(items) for family_id, items in families.items()})

    root_representative_rows: list[dict[str, Any]] = []
    root_representative_annotations: dict[str, CandidateAnnotation] = {}

    aggregated_family_rows: list[dict[str, Any]] = []
    aggregated_family_annotations: dict[str, CandidateAnnotation] = {}

    high_priority_families: list[dict[str, Any]] = []

    for family_id, members in sorted(families.items()):
        annotated_members = [member for member in members if str(member.get("candidate_id", "")) in annotations]
        scored_members = [scored_by_id[str(member.get("candidate_id", ""))] for member in members if str(member.get("candidate_id", "")) in scored_by_id]
        if not annotated_members or not scored_members:
            continue

        representative_candidate = choose_root_representative(family_id, annotated_members)
        representative_id = str(representative_candidate.get("candidate_id", ""))
        if representative_id in scored_by_id and representative_id in annotations:
            representative_row = dict(scored_by_id[representative_id])
            representative_row["candidate_id"] = family_id
            root_representative_rows.append(representative_row)
            root_representative_annotations[family_id] = annotations[representative_id]

        top_scored = max(scored_members, key=lambda item: (float(item.get("merit_score", 0)), float(item.get("confidence_score", 0))))
        aggregated_row = dict(top_scored)
        aggregated_row["candidate_id"] = family_id
        aggregated_family_rows.append(aggregated_row)
        aggregated_family_annotations[family_id] = aggregate_family_annotations(members, annotations)

        top_label = aggregated_family_annotations[family_id].committee_priority or aggregated_family_annotations[family_id].composite_label
        if top_label >= 4:
            high_priority_families.append(
                {
                    "family_id": family_id,
                    "family_size": len(members),
                    "max_committee_priority": top_label,
                    "top_scored_candidate_id": str(top_scored.get("candidate_id", "")),
                    "top_scored_merit_score": int(top_scored.get("merit_score", 0)),
                    "representative_candidate_id": representative_id,
                }
            )

    candidate_level_annotations = {cid: annotation for cid, annotation in annotations.items() if cid in scored_by_id}
    candidate_level_rows = [row for row in scored if str(row.get("candidate_id", "")) in candidate_level_annotations]

    return {
        "meta": {
            "candidate_count": len(candidates),
            "annotated_candidate_count": len(candidate_level_rows),
            "family_count": len(families),
            "largest_family_size": max((len(items) for items in families.values()), default=0),
        },
        "family_summary": {
            "size_distribution": dict(Counter(len(items) for items in families.values())),
            "families_with_counterfactuals": sum(1 for items in families.values() if len(items) > 1),
            "largest_families": [
                {"family_id": family_id, "size": len(items)}
                for family_id, items in sorted(families.items(), key=lambda item: (-len(item[1]), item[0]))[:10]
            ],
        },
        "candidate_level_metrics": build_label_evaluation(candidate_level_rows, candidate_level_annotations),
        "root_representative_metrics": build_label_evaluation(root_representative_rows, root_representative_annotations),
        "family_aggregated_metrics": build_label_evaluation(aggregated_family_rows, aggregated_family_annotations),
        "high_priority_family_snapshot": high_priority_families[:15],
    }


def render_markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    family_summary = report["family_summary"]
    candidate_level = report["candidate_level_metrics"]
    root_level = report["root_representative_metrics"]
    family_level = report["family_aggregated_metrics"]

    def _fmt_metrics(title: str, metrics: dict[str, Any]) -> list[str]:
        return [
            f"## {title}",
            f"- annotated_candidate_count: {metrics.get('annotated_candidate_count')}",
            f"- top_k: {metrics.get('top_k')}",
            f"- spearman_merit_vs_labels: {metrics.get('spearman_merit_vs_labels')}",
            f"- pairwise_accuracy: {metrics.get('pairwise_accuracy')}",
            f"- precision_at_k_priority: {metrics.get('precision_at_k_priority')}",
            f"- hidden_potential_recall_at_k: {metrics.get('hidden_potential_recall_at_k')}",
            f"- support_flag_rate_in_top_k: {metrics.get('support_flag_rate_in_top_k')}",
            "",
        ]

    lines = [
        "# Family-Aware Validation",
        "",
        "## Why This Exists",
        "",
        "Counterfactual variants can create leakage-like inflation in synthetic evaluation.",
        "This report adds family-aware views so near-neighbor candidate variants do not dominate validation.",
        "",
        "## Dataset Summary",
        f"- candidate_count: {meta['candidate_count']}",
        f"- annotated_candidate_count: {meta['annotated_candidate_count']}",
        f"- family_count: {meta['family_count']}",
        f"- largest_family_size: {meta['largest_family_size']}",
        f"- families_with_counterfactuals: {family_summary['families_with_counterfactuals']}",
        f"- family_size_distribution: {family_summary['size_distribution']}",
        "",
    ]
    lines.extend(_fmt_metrics("Candidate-Level Metrics", candidate_level))
    lines.extend(_fmt_metrics("Root-Representative Metrics", root_level))
    lines.extend(_fmt_metrics("Family-Aggregated Metrics", family_level))

    lines.extend(
        [
            "## Interpretation",
            "- Candidate-level metrics are the most optimistic because counterfactual relatives appear separately.",
            "- Root-representative metrics are stricter because each family contributes only one canonical candidate.",
            "- Family-aggregated metrics are shortlist-oriented: a family counts once, using the best surfaced candidate in that family.",
            "",
            "## Largest Families",
        ]
    )
    for item in family_summary["largest_families"]:
        lines.append(f"- {item['family_id']}: size={item['size']}")

    lines.extend(["", "## High-Priority Family Snapshot"])
    for item in report["high_priority_family_snapshot"]:
        lines.append(
            f"- {item['family_id']}: family_size={item['family_size']}, max_committee_priority={item['max_committee_priority']}, "
            f"top_scored_candidate_id={item['top_scored_candidate_id']}, top_scored_merit_score={item['top_scored_merit_score']}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="research/data/candidates_expanded_v1.json")
    parser.add_argument("--scored", default="data/archive/evaluation_pack_final_hackathon_v3/hybrid_scored_results.json")
    parser.add_argument("--annotations", default="research/data/final_hackathon_annotations_v1.json")
    parser.add_argument("--output-json", default="research/reports/family_aware_validation.json")
    parser.add_argument("--output-md", default="research/reports/family_aware_validation.md")
    args = parser.parse_args()

    candidates = load_candidates(Path(args.candidates))
    scored = load_scored(Path(args.scored))
    annotations = load_annotations(Path(args.annotations))
    report = build_family_aware_report(candidates, scored, annotations)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"Family-aware validation written to: {output_json}")
    print(f"Family-aware markdown written to: {output_md}")


if __name__ == "__main__":
    main()
