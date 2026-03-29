# inVision U Scoring Service MVP

Explainable FastAPI service for primary candidate screening support.

This is a decision-support tool for committee workflow, not an autonomous admission engine.

## Core Behavior

- Final numeric scoring is deterministic and rule-based.
- LLM is optional and used only for explainability text (claims linked to evidence).
- Sensitive/socio-economic fields are excluded from merit scoring.
- Human-in-the-loop is required for final decisions.

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run tests:

```bash
pytest -q
```

## API Endpoints

Required:

- `GET /health`
- `GET /config/scoring`
- `POST /score`
- `POST /score/batch`

Useful debug endpoints:

- `POST /debug/features`
- `POST /debug/explanation`
- `POST /debug/llm-extract` (LLM explainability output only)
- `POST /debug/score-trace` (deterministic formulas and factor contributions)

Additional utility endpoint:

- `POST /score/file`

## `POST /score`

Purpose:

- Score one candidate profile.

Minimum input:

- `candidate_id`
- At least one non-empty text source in `text_inputs`.

Supported `text_inputs` fields:

- `motivation_letter_text`
- `motivation_questions[]` (`question` + `answer`)
- `interview_text`
- `video_interview_transcript_text` (optional)
- `video_presentation_transcript_text` (optional)

Example request:

```json
{
  "candidate_id": "cand_001",
  "structured_data": {
    "education": {
      "english_proficiency": {"type": "ielts", "score": 7.0},
      "school_certificate": {"type": "unt", "score": 110}
    }
  },
  "text_inputs": {
    "motivation_letter_text": "I organized a student project and improved attendance by 30%.",
    "motivation_questions": [
      {
        "question": "Why this program?",
        "answer": "I want to build social impact projects with measurable outcomes."
      }
    ],
    "video_presentation_transcript_text": "In my video I described a tutoring initiative and weekly tracking."
  }
}
```

Core response fields:

- `candidate_id`, `scoring_run_id`, `scoring_version`
- `eligibility_status`, `eligibility_reasons`
- `merit_score`, `confidence_score`, `authenticity_risk`
- `recommendation`, `review_flags`
- `merit_breakdown`, `feature_snapshot`
- `top_strengths`, `main_gaps`, `uncertainties`, `evidence_spans`, `explanation`

Trace fields:

- `extraction_mode`: `deterministic_scoring`
- `extractor_version`
- `llm_metadata` (nullable)

## Scoring Model (Short)

### Deterministic numeric path

- Structured features: rule-based extraction from structured data and process signals.
- Text features: rule-based extraction from all available text sources (letter, Q/A, interview, transcripts).
- Authenticity risk: deterministic heuristic from genericness/evidence/consistency signals.
- Final scores: deterministic formulas and fixed weights.

### Explainability path

- LLM receives candidate text plus deterministic signals.
- LLM returns only explainability artifacts in claim -> evidence format.
- If LLM fails, deterministic fallback explanations are still returned.

## Eligibility Statuses

- `invalid`: missing id, no usable text, or consent false
- `incomplete_application`: too little text for reliable scoring
- `conditionally_eligible`: scoreable but missingness/formal checks trigger review
- `eligible`: key content present

Formal material reasons (when configured):

- `missing_required_materials_documents`
- `missing_required_materials_portfolio`
- `missing_required_materials_video`

## Recommendation Labels

- `review_priority`
- `standard_review`
- `manual_review_required`
- `insufficient_evidence`
- `incomplete_application`
- `invalid`

These are routing labels for committee workflow, not admission decisions.

## Contract Constants

Source of truth for API decision labels:

- [app/schemas/decision.py](app/schemas/decision.py)

Canonical `recommendation` values:

- `invalid`
- `incomplete_application`
- `insufficient_evidence`
- `review_priority`
- `manual_review_required`
- `standard_review`

Canonical `review_flags` values:

- `eligibility_gate`
- `low_confidence`
- `insufficient_evidence`
- `low_evidence_density`
- `moderate_authenticity_risk`
- `high_authenticity_risk`
- `contradiction_risk`
- `possible_contradiction`
- `polished_but_empty_pattern`
- `high_polished_but_empty`
- `high_genericness`
- `cross_section_mismatch`
- `section_mismatch`
- `missing_required_materials`

Note: clients should treat unknown future flags as non-breaking and render them as generic/neutral badges.

## Score Trace (Auditing)

Use `POST /debug/score-trace` to get:

- Explicit formulas for each axis
- Per-feature values, weights, and contributions
- Penalty components
- Raw outputs and 0..100 display scores

This endpoint is the main anti-black-box audit surface.

## Configuration

Primary env vars:

- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `LLM_TEMPERATURE`
- `LLM_MAX_RETRIES`
- `LLM_RETRY_BACKOFF_SECONDS`
- `LLM_RETRY_JITTER_SECONDS`
- `LLM_FALLBACK_TO_BASELINE`
- `LLM_BASE_URL`
- `LLM_API_KEY`

Example:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_TIMEOUT_SECONDS=20
LLM_TEMPERATURE=0
LLM_MAX_RETRIES=1
LLM_RETRY_BACKOFF_SECONDS=0.6
LLM_RETRY_JITTER_SECONDS=0.2
LLM_FALLBACK_TO_BASELINE=true
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key_here
```

## Evaluation Scripts

Quick diagnostics:

```bash
python scripts/score_candidates.py --input data/candidates.json
```

Offline evaluation pack:

```bash
python scripts/evaluation_pack.py --input data/candidates.json --output-dir data/evaluation_pack
```

Main artifacts:

- `data/scored_candidates.json`
- `data/diagnostics_report.json`
- `data/evaluation_pack/evaluation_report.json`

## Project Map

- `app/api/routes.py`: HTTP endpoints
- `app/schemas/`: request/response contracts
- `app/services/`: pipeline, scoring, explainability, privacy
- `app/config.py`: thresholds, weights, versions, env config
- `scripts/`: offline scoring/evaluation utilities
- `tests/`: unit and API tests

## Limits

- Heuristic text features are approximate by design.
- Authenticity risk is a review-risk signal, not proof of cheating/AI use.
- Recommendation is operational routing only.
