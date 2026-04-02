from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from research.scripts.train_shortlist_ranker_v1 import (
    BATCH_JSONL,
    MODEL_DIR,
    PAIRWISE_CSV,
    RANDOM_SEED,
    RANKER_KWARGS,
    TRAINING_CSV,
    build_feature_matrix,
    load_batch_tasks,
    load_pairwise_rows,
    load_payloads,
    matrix_for_groups,
    mean_ndcg,
    pairwise_accuracy,
    project_rank_groups,
    score_map_from_model,
)
from app.services.pipeline import ScoringPipeline


ABLATION_JSON = MODEL_DIR / "ablation_summary.json"
ABLATION_MD = MODEL_DIR / "ablation_report.md"


@dataclass(frozen=True)
class CandidateMeta:
    candidate_id: str
    split: str
    source_group: str
    final_recommendation: str
    final_hidden_potential_band: bool
    final_support_needed_band: bool
    final_authenticity_review_band: bool
    has_interview_text: bool
    text_length_bucket: str


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_training_meta() -> dict[str, CandidateMeta]:
    rows: dict[str, CandidateMeta] = {}
    with TRAINING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate_id = row["candidate_id"]
            rows[candidate_id] = CandidateMeta(
                candidate_id=candidate_id,
                split=row["split"],
                source_group=row["source_group"],
                final_recommendation=row["final_recommendation"],
                final_hidden_potential_band=parse_bool(row["final_hidden_potential_band"]),
                final_support_needed_band=parse_bool(row["final_support_needed_band"]),
                final_authenticity_review_band=parse_bool(row["final_authenticity_review_band"]),
                has_interview_text=parse_bool(row["has_interview_text"]),
                text_length_bucket=row["text_length_bucket"],
            )
    return rows


def pairwise_accuracy_filtered(
    pairwise_rows: list[dict[str, str]],
    candidate_meta: dict[str, CandidateMeta],
    score_map: dict[str, float],
    split: str,
    predicate: Callable[[CandidateMeta, CandidateMeta], bool],
) -> tuple[float | None, int]:
    total = 0
    correct = 0
    for row in pairwise_rows:
        left_id = row["candidate_id_left"]
        right_id = row["candidate_id_right"]
        left = candidate_meta[left_id]
        right = candidate_meta[right_id]
        if left.split != split or right.split != split:
            continue
        if not predicate(left, right):
            continue
        total += 1
        preferred = row["preferred_candidate_id"]
        predicted = left_id if score_map[left_id] >= score_map[right_id] else right_id
        if predicted == preferred:
            correct += 1
    if total == 0:
        return None, 0
    return correct / total, total


def pairwise_accuracy_by_preferred_slice(
    pairwise_rows: list[dict[str, str]],
    candidate_meta: dict[str, CandidateMeta],
    score_map: dict[str, float],
    split: str,
    selector: Callable[[CandidateMeta, CandidateMeta, CandidateMeta], str],
) -> dict[str, tuple[float | None, int]]:
    buckets: dict[str, list[bool]] = {}
    for row in pairwise_rows:
        left_id = row["candidate_id_left"]
        right_id = row["candidate_id_right"]
        left = candidate_meta[left_id]
        right = candidate_meta[right_id]
        if left.split != split or right.split != split:
            continue
        preferred = candidate_meta[row["preferred_candidate_id"]]
        key = selector(left, right, preferred)
        predicted = left_id if score_map[left_id] >= score_map[right_id] else right_id
        buckets.setdefault(key, []).append(predicted == preferred.candidate_id)

    result: dict[str, tuple[float | None, int]] = {}
    for key, values in buckets.items():
        result[key] = ((sum(values) / len(values)) if values else None, len(values))
    return result


def fit_model(
    feature_names: list[str],
    feature_map_by_candidate: dict[str, dict[str, float]],
    train_groups: list[Any],
    validation_groups: list[Any],
) -> lgb.LGBMRanker:
    x_train, y_train, train_group_sizes = matrix_for_groups(train_groups, feature_names, feature_map_by_candidate)
    x_val, y_val, val_group_sizes = matrix_for_groups(validation_groups, feature_names, feature_map_by_candidate)

    model = lgb.LGBMRanker(**RANKER_KWARGS)
    model.fit(
        x_train,
        y_train,
        group=train_group_sizes,
        eval_set=[(x_val, y_val)],
        eval_group=[val_group_sizes],
        eval_at=[3, 5],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(period=0)],
    )
    return model


def summarize_variant(
    variant_name: str,
    score_map: dict[str, float],
    candidate_meta: dict[str, CandidateMeta],
    pairwise_rows: list[dict[str, str]],
    validation_groups: list[Any],
    test_groups_min2: list[Any],
    test_groups_min3: list[Any],
) -> dict[str, Any]:
    val_pair, val_pair_count = pairwise_accuracy(pairwise_rows, candidate_meta, score_map, "validation")
    test_pair, test_pair_count = pairwise_accuracy(pairwise_rows, candidate_meta, score_map, "test")
    test_ndcg2 = mean_ndcg(test_groups_min2, score_map, k=2, min_group_size=2)
    test_ndcg3 = mean_ndcg(test_groups_min3, score_map, k=3, min_group_size=3)
    val_ndcg3 = mean_ndcg(validation_groups, score_map, k=3, min_group_size=3)

    source_metrics = pairwise_accuracy_by_preferred_slice(
        pairwise_rows,
        candidate_meta,
        score_map,
        "test",
        lambda _left, _right, preferred: preferred.source_group,
    )
    recommendation_metrics = pairwise_accuracy_by_preferred_slice(
        pairwise_rows,
        candidate_meta,
        score_map,
        "test",
        lambda _left, _right, preferred: preferred.final_recommendation,
    )
    support_metrics = pairwise_accuracy_by_preferred_slice(
        pairwise_rows,
        candidate_meta,
        score_map,
        "test",
        lambda _left, _right, preferred: str(preferred.final_support_needed_band).lower(),
    )
    hidden_metrics = pairwise_accuracy_by_preferred_slice(
        pairwise_rows,
        candidate_meta,
        score_map,
        "test",
        lambda _left, _right, preferred: str(preferred.final_hidden_potential_band).lower(),
    )
    interview_metrics = pairwise_accuracy_by_preferred_slice(
        pairwise_rows,
        candidate_meta,
        score_map,
        "test",
        lambda _left, _right, preferred: str(preferred.has_interview_text).lower(),
    )
    length_metrics = pairwise_accuracy_by_preferred_slice(
        pairwise_rows,
        candidate_meta,
        score_map,
        "test",
        lambda _left, _right, preferred: preferred.text_length_bucket,
    )

    def unpack(metric_map: dict[str, tuple[float | None, int]], key: str) -> dict[str, Any]:
        accuracy, count = metric_map.get(key, (None, 0))
        return {"accuracy": accuracy, "count": count}

    slice_metrics: dict[str, Any] = {
        "test_source_messy_batch_v5": unpack(source_metrics, "messy_batch_v5"),
        "test_source_messy_batch_v5_extension": unpack(source_metrics, "messy_batch_v5_extension"),
        "test_source_seed_pack": unpack(source_metrics, "seed_pack"),
        "test_recommendation_standard_review": unpack(recommendation_metrics, "standard_review"),
        "test_support_needed_true": unpack(support_metrics, "true"),
        "test_hidden_potential_false": unpack(hidden_metrics, "false"),
        "test_has_interview_false": unpack(interview_metrics, "false"),
        "test_text_length_medium": unpack(length_metrics, "medium"),
    }

    return {
        "variant_name": variant_name,
        "validation_pairwise_accuracy": val_pair,
        "validation_pairwise_count": val_pair_count,
        "validation_ndcg_at_3": val_ndcg3,
        "test_pairwise_accuracy": test_pair,
        "test_pairwise_count": test_pair_count,
        "test_ndcg_at_2": test_ndcg2,
        "test_ndcg_at_3": test_ndcg3,
        "slice_metrics": slice_metrics,
    }


def ranking_key(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    def safe(metric: float | None) -> float:
        return -1.0 if metric is None else float(metric)

    return (
        safe(summary["validation_pairwise_accuracy"]),
        safe(summary["test_pairwise_accuracy"]),
        safe(summary["slice_metrics"]["test_source_messy_batch_v5"]["accuracy"]),
        safe(summary["slice_metrics"]["test_source_seed_pack"]["accuracy"]),
        safe(summary["slice_metrics"]["test_recommendation_standard_review"]["accuracy"]),
    )


def conservative_candidates(
    baseline_summary: dict[str, Any],
    learned_variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_test = baseline_summary["test_pairwise_accuracy"] or -1.0
    baseline_val = baseline_summary["validation_pairwise_accuracy"] or -1.0
    baseline_slices = baseline_summary["slice_metrics"]

    accepted: list[dict[str, Any]] = []
    for item in learned_variants:
        if (item["validation_pairwise_accuracy"] or -1.0) <= baseline_val:
            continue
        if (item["test_pairwise_accuracy"] or -1.0) < baseline_test:
            continue
        required_slice_keys = [
            "test_source_messy_batch_v5",
            "test_source_messy_batch_v5_extension",
            "test_source_seed_pack",
            "test_recommendation_standard_review",
            "test_support_needed_true",
            "test_has_interview_false",
        ]
        okay = True
        for key in required_slice_keys:
            baseline_accuracy = baseline_slices[key]["accuracy"]
            variant_accuracy = item["slice_metrics"][key]["accuracy"]
            if baseline_accuracy is None:
                continue
            if variant_accuracy is None or variant_accuracy < baseline_accuracy:
                okay = False
                break
        if okay:
            accepted.append(item)
    return accepted


def write_markdown(
    variant_summaries: list[dict[str, Any]],
    recommended_by_rank_key: str | None,
    recommended_conservative: str | None,
) -> None:
    lines = [
        "# Shortlist Ranker V1 Ablation",
        "",
        f"- recommended_by_rank_key: `{recommended_by_rank_key}`",
        f"- recommended_conservative: `{recommended_conservative}`",
        "",
        "| Variant | Val Pair | Test Pair | Test NDCG@3 | V5 Test | V5 Ext Test | Seed Test | Std Review Test | Support=True Test | No Interview Test |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in variant_summaries:
        slice_metrics = item["slice_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    item["variant_name"],
                    f"{(item['validation_pairwise_accuracy'] or 0):.3f}",
                    f"{(item['test_pairwise_accuracy'] or 0):.3f}",
                    f"{(item['test_ndcg_at_3'] or 0):.3f}",
                    f"{(slice_metrics['test_source_messy_batch_v5']['accuracy'] or 0):.3f}",
                    f"{(slice_metrics['test_source_messy_batch_v5_extension']['accuracy'] or 0):.3f}",
                    f"{(slice_metrics['test_source_seed_pack']['accuracy'] or 0):.3f}",
                    f"{(slice_metrics['test_recommendation_standard_review']['accuracy'] or 0):.3f}",
                    f"{(slice_metrics['test_support_needed_true']['accuracy'] or 0):.3f}",
                    f"{(slice_metrics['test_has_interview_false']['accuracy'] or 0):.3f}",
                ]
            )
            + " |"
        )
    ABLATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = ScoringPipeline()
    payloads = load_payloads()
    candidate_meta = load_training_meta()
    batch_tasks = load_batch_tasks()
    pairwise_rows = load_pairwise_rows()

    all_feature_names, feature_map_by_candidate, baseline_score_map = build_feature_matrix(pipeline, payloads)
    projected_groups_min2 = project_rank_groups(batch_tasks, candidate_meta, min_group_size=2)
    projected_groups_min3 = project_rank_groups(batch_tasks, candidate_meta, min_group_size=3)

    train_groups = [group for group in projected_groups_min3 if group.split == "train"]
    validation_groups = [group for group in projected_groups_min3 if group.split == "validation"]
    test_groups_min2 = [group for group in projected_groups_min2 if group.split == "test"]
    test_groups_min3 = [group for group in projected_groups_min3 if group.split == "test"]

    risky_axes = ["leadership_axis", "trust_axis", "potential_axis", "community_values_axis"]
    supportive_flags = ["hidden_potential_score", "support_needed_score", "authenticity_risk_neg"]
    base_semantic = ["trajectory_score", "confidence_score", "merit_score", "evidence_coverage_score", "shortlist_priority_score"]

    variants: dict[str, list[str]] = {
        "all_features": all_feature_names,
        "drop_authenticity": [name for name in all_feature_names if name != "authenticity_risk_neg"],
        "drop_support": [name for name in all_feature_names if name != "support_needed_score"],
        "drop_hidden_support_auth": [name for name in all_feature_names if name not in supportive_flags],
        "drop_trust_auth": [name for name in all_feature_names if name not in {"trust_axis", "authenticity_risk_neg"}],
        "drop_leadership_trust": [name for name in all_feature_names if name not in {"leadership_axis", "trust_axis"}],
        "drop_axes": [name for name in all_feature_names if name not in risky_axes],
        "drop_axes_plus_auth": [name for name in all_feature_names if name not in set(risky_axes + ["authenticity_risk_neg"])],
        "base_plus_supportive": [name for name in all_feature_names if name in base_semantic + supportive_flags],
        "scores_only_light": [name for name in all_feature_names if name in ["trajectory_score", "confidence_score", "hidden_potential_score", "support_needed_score", "authenticity_risk_neg", "merit_score"]],
    }

    variant_summaries: list[dict[str, Any]] = []

    baseline_summary = summarize_variant(
        variant_name="offline_baseline",
        score_map=baseline_score_map,
        candidate_meta=candidate_meta,
        pairwise_rows=pairwise_rows,
        validation_groups=validation_groups,
        test_groups_min2=test_groups_min2,
        test_groups_min3=test_groups_min3,
    )
    variant_summaries.append(baseline_summary)

    candidate_ids = sorted(payloads)
    learned_score_maps: dict[str, dict[str, float]] = {}

    for variant_name, feature_names in variants.items():
        model = fit_model(feature_names, feature_map_by_candidate, train_groups, validation_groups)
        learned_score_map = score_map_from_model(candidate_ids, feature_names, feature_map_by_candidate, model)
        learned_score_maps[variant_name] = learned_score_map
        variant_summaries.append(
            summarize_variant(
                variant_name=variant_name,
                score_map=learned_score_map,
                candidate_meta=candidate_meta,
                pairwise_rows=pairwise_rows,
                validation_groups=validation_groups,
                test_groups_min2=test_groups_min2,
                test_groups_min3=test_groups_min3,
            )
        )

    blend_specs = {
        "blend_all_features_0.35": ("all_features", 0.35),
        "blend_all_features_0.50": ("all_features", 0.50),
        "blend_drop_trust_auth_0.50": ("drop_trust_auth", 0.50),
        "blend_drop_axes_0.50": ("drop_axes", 0.50),
    }
    for variant_name, (base_variant, alpha) in blend_specs.items():
        learned_score_map = learned_score_maps[base_variant]
        blended_score_map = {
            candidate_id: (alpha * learned_score_map[candidate_id]) + ((1.0 - alpha) * baseline_score_map[candidate_id])
            for candidate_id in candidate_ids
        }
        variant_summaries.append(
            summarize_variant(
                variant_name=variant_name,
                score_map=blended_score_map,
                candidate_meta=candidate_meta,
                pairwise_rows=pairwise_rows,
                validation_groups=validation_groups,
                test_groups_min2=test_groups_min2,
                test_groups_min3=test_groups_min3,
            )
        )

    learned_variants = [item for item in variant_summaries if item["variant_name"] != "offline_baseline"]
    learned_variants.sort(key=ranking_key, reverse=True)
    conservative_variants = conservative_candidates(baseline_summary, learned_variants)

    summary = {
        "row_count": len(candidate_meta),
        "variant_count": len(variant_summaries),
        "variants": variant_summaries,
        "recommended_variant_by_rank_key": learned_variants[0]["variant_name"] if learned_variants else None,
        "recommended_variant_conservative": conservative_variants[0]["variant_name"] if conservative_variants else None,
        "ranking_priority": [
            "validation_pairwise_accuracy",
            "test_pairwise_accuracy",
            "test_source_messy_batch_v5",
            "test_source_seed_pack",
            "test_recommendation_standard_review",
        ],
    }

    with ABLATION_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    sorted_for_report = [baseline_summary] + learned_variants
    write_markdown(
        sorted_for_report,
        summary["recommended_variant_by_rank_key"],
        summary["recommended_variant_conservative"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
