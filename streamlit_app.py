"""Streamlit demo for inVision U scoring service."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import random
from typing import Any

import httpx
import streamlit as st


def default_payload() -> dict[str, Any]:
    return {
        "candidate_id": "cand_demo_001",
        "consent": True,
        "structured_data": {
            "education": {
                "english_proficiency": {"type": "ielts", "score": 6.5},
                "school_certificate": {"type": "unt", "score": 112},
            },
            "application_materials": {
                "documents": ["cv.pdf"],
                "attachments": [],
                "portfolio_links": ["https://example.com/project"],
                "video_presentation_link": "https://example.com/video",
            },
        },
        "behavioral_signals": {
            "completion_rate": 0.9,
            "returned_to_edit": False,
            "skipped_optional_questions": 0,
            "meaningful_answers_count": 2,
            "scenario_depth": 0.75,
        },
        "text_inputs": {
            "motivation_letter_text": (
                "I led a peer tutoring initiative, tracked weekly progress, "
                "and improved attendance by 30 percent over one semester."
            ),
            "motivation_questions": [
                {
                    "question": "Why this program?",
                    "answer": "I want stronger tools for measurable education impact projects.",
                }
            ],
            "interview_text": "I coordinated a small team and documented outcomes every week.",
        },
    }


def synthetic_payload() -> dict[str, Any]:
    suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return {
        "candidate_id": f"cand_synth_{suffix}",
        "consent": True,
        "structured_data": {
            "education": {
                "english_proficiency": {"type": "ielts", "score": 7.0},
                "school_certificate": {"type": "unt", "score": 118},
            },
            "application_materials": {
                "documents": ["cv.pdf", "motivation.pdf"],
                "attachments": ["portfolio.pdf"],
                "portfolio_links": ["https://example.com/portfolio", "https://github.com/example"],
                "video_presentation_link": "https://example.com/presentation-video",
            },
        },
        "behavioral_signals": {
            "completion_rate": 0.95,
            "returned_to_edit": False,
            "skipped_optional_questions": 1,
            "meaningful_answers_count": 3,
            "scenario_depth": 0.82,
        },
        "text_inputs": {
            "motivation_letter_text": (
                "Over two years I built a school mentorship program for 60 students, "
                "set up monthly checkpoints, and raised project completion rates by 22 percent."
            ),
            "motivation_questions": [
                {
                    "question": "Why this program?",
                    "answer": "I need a stronger evidence framework and mentorship to scale my initiatives.",
                },
                {
                    "question": "Describe a challenge you solved",
                    "answer": (
                        "Our first cohort had low retention. I introduced role ownership and weekly retrospectives, "
                        "and retention improved from 54 to 83 percent in three months."
                    ),
                },
            ],
            "interview_text": (
                "I coordinated a 7-person volunteer team, established tracking dashboards, "
                "and reported outcomes to two partner schools."
            ),
            "video_interview_transcript_text": (
                "In the interview I explained how we split responsibilities, verified attendance, "
                "and handled dropout risk with targeted outreach."
            ),
        },
    }


def load_random_candidate(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"

    pool = raw.get("candidates", raw)
    if not isinstance(pool, list) or not pool:
        return None, "No candidates found in file"
    picked = random.choice(pool)
    if not isinstance(picked, dict):
        return None, "Candidate entry is not an object"
    return picked, None


def post_json(base_url: str, endpoint: str, payload: dict[str, Any], timeout_seconds: int) -> tuple[bool, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        response = httpx.post(url, json=payload, timeout=timeout_seconds)
        data = response.json()
        if response.status_code >= 400:
            return False, {"status_code": response.status_code, "error": data}
        return True, data
    except httpx.HTTPError as exc:
        return False, {"error": str(exc)}
    except json.JSONDecodeError:
        return False, {"error": "Response is not valid JSON"}


def get_json(base_url: str, endpoint: str, timeout_seconds: int) -> tuple[bool, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        response = httpx.get(url, timeout=timeout_seconds)
        data = response.json()
        if response.status_code >= 400:
            return False, {"status_code": response.status_code, "error": data}
        return True, data
    except httpx.HTTPError as exc:
        return False, {"error": str(exc)}
    except json.JSONDecodeError:
        return False, {"error": "Response is not valid JSON"}


def render_score_result(result: dict[str, Any]) -> None:
    st.subheader("Decision")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Merit", result.get("merit_score", "n/a"))
    col2.metric("Confidence", result.get("confidence_score", "n/a"))
    col3.metric("Authenticity risk", result.get("authenticity_risk", "n/a"))
    col4.metric("Recommendation", str(result.get("recommendation", "n/a")))

    flags = result.get("review_flags", [])
    if isinstance(flags, list) and flags:
        st.write("Review flags:")
        st.write(" | ".join(str(flag) for flag in flags))

    with st.expander("Merit breakdown"):
        st.json(result.get("merit_breakdown", {}), expanded=False)

    with st.expander("Top strengths"):
        for item in result.get("top_strengths", []):
            st.write(f"- {item}")

    with st.expander("Main gaps"):
        for item in result.get("main_gaps", []):
            st.write(f"- {item}")

    with st.expander("Uncertainties"):
        for item in result.get("uncertainties", []):
            st.write(f"- {item}")

    with st.expander("Raw response JSON"):
        st.json(result, expanded=False)


def ensure_payload_state() -> None:
    if "payload_json" not in st.session_state:
        st.session_state.payload_json = json.dumps(default_payload(), ensure_ascii=False, indent=2)


def main() -> None:
    st.set_page_config(page_title="inVision U Streamlit Demo", layout="wide")
    ensure_payload_state()

    st.title("inVision U Scoring Demo")
    st.caption("Interactive demo client for /score and /debug/score-trace endpoints")

    with st.sidebar:
        st.header("Connection")
        base_url = st.text_input("Service base URL", value=os.getenv("SCORING_API_BASE_URL", "http://127.0.0.1:8000"))
        timeout_seconds = int(st.number_input("Timeout (seconds)", min_value=5, max_value=300, value=90, step=5))

        if st.button("Check /health", use_container_width=True):
            ok, data = get_json(base_url, "/health", timeout_seconds)
            if ok:
                st.success("Service is reachable")
                st.json(data, expanded=False)
            else:
                st.error("Health check failed")
                st.json(data, expanded=False)

        if st.button("Show /config/scoring", use_container_width=True):
            ok, data = get_json(base_url, "/config/scoring", timeout_seconds)
            if ok:
                st.json(data, expanded=False)
            else:
                st.error("Config fetch failed")
                st.json(data, expanded=False)

    st.subheader("Candidate payload")
    c1, c2, c3 = st.columns(3)
    if c1.button("Load default sample", use_container_width=True):
        st.session_state.payload_json = json.dumps(default_payload(), ensure_ascii=False, indent=2)
    if c2.button("Load random from data/candidates.json", use_container_width=True):
        picked, err = load_random_candidate(Path("data/candidates.json"))
        if err:
            st.warning(err)
        else:
            st.session_state.payload_json = json.dumps(picked, ensure_ascii=False, indent=2)
    if c3.button("Generate synthetic candidate", use_container_width=True):
        st.session_state.payload_json = json.dumps(synthetic_payload(), ensure_ascii=False, indent=2)

    st.text_area("Editable JSON", key="payload_json", height=360)

    a1, a2, a3 = st.columns(3)
    run_score = a1.button("Run /score", type="primary", use_container_width=True)
    run_trace = a2.button("Run /debug/score-trace", use_container_width=True)
    run_llm_extract = a3.button("Run /debug/llm-extract", use_container_width=True)

    if run_score or run_trace or run_llm_extract:
        try:
            payload = json.loads(st.session_state.payload_json)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON payload: {exc}")
            return

        endpoint = "/score"
        if run_trace:
            endpoint = "/debug/score-trace"
        if run_llm_extract:
            endpoint = "/debug/llm-extract"

        with st.spinner(f"Calling {endpoint}..."):
            ok, data = post_json(base_url, endpoint, payload, timeout_seconds)

        if not ok:
            st.error(f"Request to {endpoint} failed")
            st.json(data, expanded=False)
            return

        if endpoint == "/score":
            render_score_result(data)
        else:
            st.subheader(f"Result: {endpoint}")
            st.json(data, expanded=False)


if __name__ == "__main__":
    main()
