# Backend Integration One-Pager

## Base URL

- Local: `http://127.0.0.1:8000`

## Production Endpoints

1. `GET /health`
2. `GET /config/scoring`
3. `POST /score`
4. `POST /score/batch`

Use debug routes only for diagnostics:

- `POST /debug/features`
- `POST /debug/explanation`
- `POST /debug/llm-extract`

## 1) `POST /score`

Purpose:

- Score one candidate application and return explainable decision-support payload.

Request body:

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
    "motivation_letter_text": "...",
    "motivation_questions": [{"question": "...", "answer": "..."}],
    "interview_text": "..."
  },
  "behavioral_signals": {
    "completion_rate": 1.0,
    "returned_to_edit": false,
    "skipped_optional_questions": 0,
    "meaningful_answers_count": 1,
    "scenario_depth": 0.7
  },
  "metadata": {
    "source": "web_form",
    "submitted_at": "2026-03-29T12:00:00Z",
    "scoring_version": "v1"
  }
}
```

Core response fields (stable contract):

- `candidate_id`
- `scoring_run_id`
- `scoring_version`
- `prompt_version`
- `eligibility_status`
- `eligibility_reasons`
- `merit_score` (0..100)
- `confidence_score` (0..100)
- `authenticity_risk` (0..100)
- `recommendation`
- `review_flags`
- `merit_breakdown`
- `feature_snapshot`
- `top_strengths`
- `main_gaps`
- `uncertainties`
- `evidence_spans`
- `explanation`

Current trace fields (also returned):

- `extraction_mode` (`hybrid`)
- `extractor_version`
- `llm_metadata` (nullable)

## 2) `POST /score/batch`

Purpose:

- Score multiple candidates in one request.

Request body:

```json
{
  "candidates": [
    {"candidate_id": "cand_001", "text_inputs": {"motivation_letter_text": "..."}},
    {"candidate_id": "cand_002", "text_inputs": {"motivation_letter_text": "..."}}
  ]
}
```

Response fields:

- `scoring_run_id`
- `scoring_version`
- `count`
- `results` (array of `POST /score` responses)

## 3) `GET /config/scoring`

Purpose:

- Read active runtime scoring config and extraction settings.

Returns:

- `scoring_version`
- `prompt_version`
- `excluded_fields`
- `weights`
- `thresholds`
- `extraction_strategy`
- `llm_provider`
- `llm_model`
- `llm_fallback_to_deterministic_extractor_on_failure`

## 4) `GET /health`

Purpose:

- Liveness probe.

Returns:

```json
{"status": "ok", "service": "invision-u-scoring-mvp", "scoring_version": "v1.0.0"}
```

## Integration Notes

- The service is a decision-support tool, not an autonomous admission decision engine.
- LLM is used only for feature extraction/explanation assistance.
- Final scores and recommendation are deterministic and computed internally.
- Sensitive/demographic/socio-economic fields are excluded from merit scoring.
