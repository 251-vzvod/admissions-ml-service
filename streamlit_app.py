"""Streamlit demo UI for candidate scoring service."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.config import CONFIG
from app.services.pipeline import ScoringPipeline


def _default_candidate() -> dict[str, Any]:
    return {
        "candidate_id": "demo_001",
        "structured_data": {
            "education": {
                "english_proficiency": {"type": "ielts", "score": 7.0},
                "school_certificate": {"type": "unt", "score": 112},
            }
        },
        "text_inputs": {
            "motivation_letter_text": (
                "I started a student product club and organized weekly sessions. "
                "In four months we built two prototypes and improved participation by 35%."
            ),
            "motivation_questions": [
                {
                    "question": "Describe your initiative.",
                    "answer": "I launched the club myself, recruited peers, and documented outcomes.",
                }
            ],
            "interview_text": "I can explain actions, timeline, challenges, and measurable outcomes.",
        },
        "behavioral_signals": {
            "completion_rate": 1.0,
            "returned_to_edit": True,
            "skipped_optional_questions": 0,
            "meaningful_answers_count": 1,
            "scenario_depth": 0.8,
        },
    }


def _apply_runtime_mode(mode: str, provider: str, model: str, fallback: bool) -> None:
    CONFIG.llm.enable_llm = mode == "llm"
    CONFIG.llm.provider = provider
    CONFIG.llm.model = model
    CONFIG.llm.fallback_to_baseline = fallback


def _render_result(result: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Merit", result["merit_score"])
    c2.metric("Confidence", result["confidence_score"])
    c3.metric("Authenticity Risk", result["authenticity_risk"])

    st.write("### Recommendation")
    st.write(result["recommendation"])
    st.write("Flags:", result.get("review_flags", []))

    st.write("### Extraction Metadata")
    st.json(
        {
            "extraction_mode": result.get("extraction_mode"),
            "extractor_version": result.get("extractor_version"),
            "llm_metadata": result.get("llm_metadata"),
        }
    )

    st.write("### Merit Breakdown")
    st.dataframe(pd.DataFrame([result.get("merit_breakdown", {})]))

    st.write("### Feature Snapshot")
    st.dataframe(pd.DataFrame([result.get("feature_snapshot", {})]))

    st.write("### Explanation")
    st.json(
        {
            "top_strengths": result.get("top_strengths", []),
            "main_gaps": result.get("main_gaps", []),
            "uncertainties": result.get("uncertainties", []),
            "evidence_spans": result.get("evidence_spans", []),
            "summary": result.get("explanation", {}).get("summary"),
        }
    )


def _batch_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    recommendations = Counter(item["recommendation"] for item in results)
    extraction_modes = Counter(item.get("extraction_mode", "baseline") for item in results)
    avg_merit = round(sum(item["merit_score"] for item in results) / max(len(results), 1), 2)
    avg_confidence = round(sum(item["confidence_score"] for item in results) / max(len(results), 1), 2)
    avg_risk = round(sum(item["authenticity_risk"] for item in results) / max(len(results), 1), 2)
    return {
        "count": len(results),
        "avg_merit": avg_merit,
        "avg_confidence": avg_confidence,
        "avg_authenticity_risk": avg_risk,
        "recommendation_distribution": dict(recommendations),
        "extraction_mode_distribution": dict(extraction_modes),
    }


def main() -> None:
    st.set_page_config(page_title="inVision U Scoring Demo", layout="wide")
    st.title("inVision U Candidate Scoring Demo")
    st.caption("LLM is used as extractor only. Final scoring and routing remain deterministic.")

    with st.sidebar:
        st.header("Runtime Mode")
        mode = st.radio("Extraction mode", options=["baseline", "llm"], index=0)
        provider = st.text_input("LLM provider", value=CONFIG.llm.provider)
        model = st.text_input("LLM model", value=CONFIG.llm.model)
        fallback = st.checkbox("Fallback to baseline on LLM failure", value=CONFIG.llm.fallback_to_baseline)
        _apply_runtime_mode(mode=mode, provider=provider, model=model, fallback=fallback)
        st.info(
            "Recommendation is a workflow routing label, not final admission decision. "
            "Confidence score measures reliability, not candidate quality."
        )

    pipeline = ScoringPipeline()
    tab_single, tab_batch, tab_dataset = st.tabs(["Single Score", "Batch File", "Dataset Demo"])

    with tab_single:
        st.subheader("Single Candidate Scoring")
        default_json = json.dumps(_default_candidate(), ensure_ascii=False, indent=2)
        raw_input = st.text_area("Candidate JSON", value=default_json, height=360)
        if st.button("Run single scoring", type="primary"):
            try:
                payload = json.loads(raw_input)
                result = pipeline.score_candidate(payload).model_dump()
                _render_result(result)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to score candidate: {exc}")

    with tab_batch:
        st.subheader("Batch Scoring From JSON")
        uploaded = st.file_uploader("Upload candidates JSON", type=["json"])
        if st.button("Run batch scoring"):
            try:
                if uploaded is not None:
                    payload = json.loads(uploaded.getvalue().decode("utf-8"))
                else:
                    data_path = Path("data/candidates.json")
                    payload = json.loads(data_path.read_text(encoding="utf-8"))

                candidates = payload.get("candidates", [])
                results = [pipeline.score_candidate(item).model_dump() for item in candidates]
                st.success(f"Scored {len(results)} candidates")
                st.json(_batch_summary(results))
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "candidate_id": r["candidate_id"],
                                "merit_score": r["merit_score"],
                                "confidence_score": r["confidence_score"],
                                "authenticity_risk": r["authenticity_risk"],
                                "recommendation": r["recommendation"],
                                "extraction_mode": r.get("extraction_mode"),
                            }
                            for r in results
                        ]
                    )
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Batch scoring failed: {exc}")

    with tab_dataset:
        st.subheader("Quick Demo On data/candidates.json")
        n = st.slider("Number of candidates", min_value=1, max_value=20, value=5)
        if st.button("Run dataset demo"):
            try:
                payload = json.loads(Path("data/candidates.json").read_text(encoding="utf-8"))
                candidates = payload.get("candidates", [])[:n]
                results = [pipeline.score_candidate(item).model_dump() for item in candidates]
                st.json(_batch_summary(results))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Dataset demo failed: {exc}")


if __name__ == "__main__":
    main()
