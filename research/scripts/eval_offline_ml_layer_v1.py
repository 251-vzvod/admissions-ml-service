from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from research.scripts.offline_ml_common import OFFLINE_LAYER_DIR


METRICS_JSON = OFFLINE_LAYER_DIR / "metrics_summary.json"
SLICE_JSON = OFFLINE_LAYER_DIR / "slice_eval_summary.json"
PROMOTION_JSON = OFFLINE_LAYER_DIR / "promotion_summary.json"


def _safe_delta(selected: float | None, baseline: float | None) -> float | None:
    if selected is None or baseline is None:
        return None
    return float(selected) - float(baseline)


def _slice_regressions(slice_payload: dict[str, dict], *, metric_name: str, max_drop: float) -> list[dict[str, float | str]]:
    regressions: list[dict[str, float | str]] = []
    baseline_payload = slice_payload.get("baseline", {}).get("test", {})
    selected_payload = slice_payload.get("selected_model", {}).get("test", {})
    for slice_name, baseline_metrics in baseline_payload.items():
        selected_metrics = selected_payload.get(slice_name)
        if not selected_metrics:
            continue
        baseline_metric = baseline_metrics.get(metric_name)
        selected_metric = selected_metrics.get(metric_name)
        delta = _safe_delta(selected_metric, baseline_metric)
        if delta is not None and delta < -max_drop:
            regressions.append(
                {
                    "slice_name": slice_name,
                    "baseline_metric": float(baseline_metric),
                    "selected_metric": float(selected_metric),
                    "delta": float(delta),
                }
            )
    return regressions


def main() -> None:
    if not METRICS_JSON.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_JSON}")
    if not SLICE_JSON.exists():
        raise FileNotFoundError(f"Missing slice summary file: {SLICE_JSON}")

    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    slice_summary = json.loads(SLICE_JSON.read_text(encoding="utf-8"))

    priority = metrics["priority_model_v1"]
    priority_pass = (
        priority["selected_model"]["validation"]["spearman"] > priority["baseline"]["validation"]["spearman"]
        and priority["selected_model"]["test"]["spearman"] > priority["baseline"]["test"]["spearman"]
    )

    routing_promotion: dict[str, dict[str, object]] = {}
    for target_name, payload in metrics["routing_models_v1"].items():
        routing_promotion[target_name] = {
            "promotion_pass": (
                payload["selected_tuned"]["validation"]["f1"] > payload["baseline"]["validation"]["f1"]
                and payload["selected_tuned"]["test"]["f1"] > payload["baseline"]["test"]["f1"]
            ),
            "slice_regressions_gt_3pp": _slice_regressions(
                slice_summary.get(target_name, {}),
                metric_name="f1",
                max_drop=0.03,
            ),
        }

    pairwise = metrics["pairwise_ranker_v2"]
    pairwise_pass = (
        (pairwise["selected_model"]["validation_pairwise_accuracy"] or 0.0)
        > (pairwise["baseline"]["validation_pairwise_accuracy"] or 0.0)
        and (pairwise["selected_model"]["test_pairwise_accuracy"] or 0.0)
        > (pairwise["baseline"]["test_pairwise_accuracy"] or 0.0)
    )

    promotion_summary = {
        "priority_model_v1": {
            "promotion_pass": priority_pass,
            "slice_regressions_gt_3pp": _slice_regressions(
                slice_summary.get("priority_model_v1", {}),
                metric_name="spearman",
                max_drop=0.03,
            ),
        },
        "routing_models_v1": routing_promotion,
        "pairwise_ranker_v2": {
            "promotion_pass": pairwise_pass,
        },
    }

    PROMOTION_JSON.write_text(
        json.dumps(promotion_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(promotion_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
