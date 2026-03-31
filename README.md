# inVision U Scoring Service

FastAPI service for primary candidate screening support in the inVision U admissions workflow.

This service is a decision-support tool. It does not make autonomous admission decisions.

## What The Service Does

The API accepts one or more candidate profiles and returns:

- `merit_score`: overall strength of the candidate profile
- `confidence_score`: how reliable the assessment is, given the available evidence
- `authenticity_risk`: review-risk signal, not proof of cheating
- `recommendation`: routing label for committee workflow
- short explanations, evidence spans, and follow-up guidance for the committee

The scoring logic is transparent:

- numeric scoring is deterministic
- sensitive and socio-economic fields are excluded from merit scoring
- LLM is optional and used only for explainability output
- final decision always stays with a human committee

## Public API

The service exposes only three routes:

- `GET /health`
- `POST /score`
- `POST /score/batch`

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open API docs:

- `http://127.0.0.1:8000/docs`

## Environment Configuration

The main config file is:

- `.env`

The template is:

- `.env.example`

### Default Local Mode

Right now the project is configured to work locally with `Ollama`.

Minimal local setup:

```env
ENABLE_LLM=true
LLM_PROVIDER=openai-compatible
LLM_MODEL=llama3.1:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_TIMEOUT_SECONDS=120
```

If you do not want to use any LLM at all:

```env
ENABLE_LLM=false
```

In that mode:

- `POST /score`
- `POST /score/batch`

still work normally, but the service skips LLM explainability and returns deterministic scoring only.

### Switching Back To OpenAI Later

When credits are available again, only these fields need to change:

```env
ENABLE_LLM=true
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_openai_api_key
```

No application code changes are required.

## Request Shape

Minimum useful payload:

```json
{
  "candidate_id": "cand_001",
  "text_inputs": {
    "motivation_letter_text": "I started a small student initiative and kept improving it after the first version failed."
  },
  "consent": true
}
```

Supported fields:

- `candidate_id`
- `structured_data`
- `text_inputs`
- `behavioral_signals`
- `metadata`
- `consent`

Important `text_inputs` fields:

- `motivation_letter_text`
- `motivation_questions[]`
- `interview_text`
- `video_interview_transcript_text`
- `video_presentation_transcript_text`

## `POST /score`

Use this route to score one candidate.

Example request:

```json
{
  "candidate_id": "cand_001",
  "structured_data": {
    "education": {
      "english_proficiency": {
        "type": "ielts",
        "score": 7.0
      },
      "school_certificate": {
        "type": "unt",
        "score": 110
      }
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
  },
  "consent": true
}
```

Main response fields:

- `candidate_id`
- `scoring_run_id`
- `scoring_version`
- `eligibility_status`
- `eligibility_reasons`
- `merit_score`
- `confidence_score`
- `authenticity_risk`
- `recommendation`
- `review_flags`
- `merit_breakdown`
- `feature_snapshot`
- `semantic_rubric_scores`
- `top_strengths`
- `main_gaps`
- `uncertainties`
- `authenticity_review_reasons`
- `ai_detector`
- `committee_cohorts`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`
- `evidence_spans`
- `explanation`

Example response shape:

```json
{
  "candidate_id": "cand_001",
  "scoring_run_id": "run_20260329_120000_abcd1234",
  "scoring_version": "v1.2.0",
  "prompt_version": null,
  "extraction_mode": "deterministic_scoring",
  "extractor_version": "llm-extractor-v1",
  "llm_metadata": null,
  "eligibility_status": "eligible",
  "eligibility_reasons": [],
  "merit_score": 25,
  "confidence_score": 42,
  "authenticity_risk": 40,
  "recommendation": "standard_review",
  "review_flags": [
    "low_confidence",
    "low_evidence_density"
  ],
  "semantic_rubric_scores": {
    "leadership_potential": 51,
    "growth_trajectory": 48,
    "motivation_authenticity": 54,
    "authenticity_groundedness": 46,
    "hidden_potential": 49
  },
  "committee_cohorts": [
    "Promising but needs support"
  ],
  "what_to_verify_manually": [
    "Ask for one concrete example with actions, obstacles, and measurable outcome."
  ],
  "suggested_follow_up_question": "What is one example from your application that best shows how you create value for other people, not just for yourself?"
}
```

## `POST /score/batch`

Use this route to score multiple candidates in one request.

Example request:

```json
{
  "candidates": [
    {
      "candidate_id": "cand_001",
      "text_inputs": {
        "motivation_letter_text": "I started a tutoring group for younger students."
      },
      "consent": true
    },
    {
      "candidate_id": "cand_002",
      "text_inputs": {
        "motivation_letter_text": "I want to study because I need more opportunities."
      },
      "consent": true
    }
  ]
}
```

Response shape:

- `scoring_run_id`
- `scoring_version`
- `count`
- `results[]`

## How To Read The Scores

- `merit_score`: candidate promise and strength signals
- `confidence_score`: reliability of the current assessment
- `authenticity_risk`: review-risk signal based on groundedness, consistency, and evidence density

These scores are not admission decisions.

`recommendation` is only a workflow routing label:

- `review_priority`
- `standard_review`
- `manual_review_required`
- `insufficient_evidence`
- `incomplete_application`
- `invalid`

## AI Detector

The service supports an optional auxiliary AI-generated text detector:

- `desklib/ai-text-detector-v1.01`

It is used conservatively:

- English-only by default
- weak signal only
- never changes `merit_score`
- never auto-rejects a candidate
- only contributes to authenticity review guidance

Optional install:

```bash
python -m pip install -r requirements-ai-detector.txt
```

Enable in `.env`:

```env
AI_DETECTOR_ENABLED=true
AI_DETECTOR_MODEL=desklib/ai-text-detector-v1.01
AI_DETECTOR_ENGLISH_ONLY=true
AI_DETECTOR_MIN_WORDS=60
AI_DETECTOR_ELEVATED_PROBABILITY_THRESHOLD=0.80
```

## Project Structure

- `app/api/routes.py`: public HTTP routes
- `app/schemas/`: request and response contracts
- `app/services/`: scoring pipeline, explainability, privacy, authenticity, semantic features
- `app/config.py`: environment and scoring config
- `tests/`: API and scoring tests

## Important Limits

- The service is a committee-support tool, not an autonomous selector.
- `authenticity_risk` is a review signal, not proof of AI use or cheating.
- LLM output is optional and used only for explanations.
- Missing modalities reduce confidence of assessment, not candidate worth by default.

## Deployment

Example `Procfile`:

```txt
web: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For Railway or similar platforms:

1. Push the repo.
2. Set environment variables from `.env.example`.
3. Start the service with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

After deployment, the API docs are available at:

- `/docs`
