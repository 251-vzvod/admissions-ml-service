# inVision U Scoring Service

FastAPI service for primary candidate screening support in the inVision U admissions workflow.

This service is a decision-support tool. It does not make autonomous admission decisions.

## Feature Dictionary

To understand what the service outputs actually mean, start here:

- [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md)

This file explains:

- which scores are committee-facing
- which signals are secondary reviewer details
- which raw features stay internal
- why some correlated signals are intentionally aggregated before exposure

For the current fairness posture, targeted mitigations, and remaining risks:

- [docs/fairness_note.md](docs/fairness_note.md)

For the current runtime architecture and shortlist direction:

- [docs/ML_SERVICE_V2_ARCHITECTURE.md](docs/ML_SERVICE_V2_ARCHITECTURE.md)

## What The Service Does

The API accepts one or more candidate profiles and returns:

- `merit_score`: overall strength of the candidate profile
- `confidence_score`: how reliable the assessment is, given the available evidence
- `authenticity_risk`: review-risk signal, not proof of cheating
- `recommendation`: routing label for committee workflow
- `hidden_potential_score`, `support_needed_score`, `shortlist_priority_score`
- `trajectory_score`, `evidence_coverage_score`
- compact `evidence_highlights`
- section-to-section consistency-aware authenticity review
- short explanations and follow-up guidance for the committee

The scoring logic is transparent:

- numeric scoring is deterministic
- sensitive and socio-economic fields are excluded from merit scoring
- LLM is optional and used only for bounded reviewer outputs and explainability
- documents, portfolio links, and video presence are not direct merit boosts
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

If you want the heavier sentence-transformer semantic backend locally:

```bash
python -m pip install -r requirements-semantic.txt
```

For local test runs:

```bash
python -m pip install -r requirements-dev.txt
```

Open API docs:

- `http://127.0.0.1:8000/docs`

## Environment Configuration

The main config file is:

- `.env`

The template is:

- `.env.example`

### Default Local Mode

Right now the project is deterministic-first by default.

If you want local LLM reviewer assistance, the project is configured to work with `Ollama`.

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

## Semantic Backend

The base install is now deploy-safe and lightweight.

Default mode:

- `SEMANTIC_BACKEND=hash`

Optional heavier mode:

- install `requirements-semantic.txt`
- set `SEMANTIC_BACKEND=sentence-transformer`
- set `SEMANTIC_MODEL=sentence-transformers/all-MiniLM-L6-v2`

Runtime posture:

- applicant text is assumed to be English
- `hash` is the safe default for cloud builds with tight image limits
- `sentence-transformer` is an opt-in stronger NLP mode when your deploy target can afford the extra weight

## Request Shape

Canonical payload:

```json
{
  "candidate_id": "cand_001",
  "profile": {
    "academics": {
      "english_proficiency": {
        "type": "ielts",
        "score": 7.0
      }
    },
    "narratives": {
      "motivation_letter_text": "I started a small student initiative and kept improving it after the first version failed."
    },
    "process_signals": {
      "completion_rate": 1.0
    }
  },
  "consent": true
}
```

Canonical top-level fields:

- `candidate_id`
- `profile`
- `consent`

Canonical `profile` sections:

- `academics`
- `materials`
- `narratives`
- `process_signals`
- `metadata`

Important `profile.narratives` fields:

- `motivation_letter_text`
- `motivation_questions[]`
- `interview_text`
- `video_interview_transcript_text`
- `video_presentation_transcript_text`

Legacy payloads with `structured_data`, `text_inputs`, `behavioral_signals`, and top-level `metadata` are still accepted and normalized into the canonical profile contract.

## `POST /score`

Use this route to score one candidate.

Example request:

```json
{
  "candidate_id": "cand_001",
  "profile": {
    "academics": {
      "english_proficiency": {
        "type": "ielts",
        "score": 7.0
      },
      "school_certificate": {
        "type": "unt",
        "score": 110
      }
    },
    "narratives": {
      "motivation_letter_text": "I organized a student project and improved attendance by 30%.",
      "motivation_questions": [
        {
          "question": "Why this program?",
          "answer": "I want to build social impact projects with measurable outcomes."
        }
      ],
      "video_presentation_transcript_text": "In my video I described a tutoring initiative and weekly tracking."
    }
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
- `hidden_potential_score`
- `support_needed_score`
- `shortlist_priority_score`
- `evidence_coverage_score`
- `trajectory_score`
- `committee_cohorts`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`
- `evidence_highlights`
- `top_strengths`
- `main_gaps`
- `explanation`

The merit core now also includes a community/value dimension aligned with InVisionU's mission.

Example response shape:

```json
{
  "candidate_id": "cand_001",
  "scoring_run_id": "run_20260329_120000_abcd1234",
  "scoring_version": "v1.2.0",
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
  "hidden_potential_score": 58,
  "support_needed_score": 41,
  "shortlist_priority_score": 63,
  "evidence_coverage_score": 46,
  "trajectory_score": 61,
  "evidence_highlights": [
    {
      "claim": "Candidate demonstrates growth through challenge, adaptation, and reflection.",
      "support_level": "moderate",
      "source": "motivation_letter_text",
      "snippet": "I started a tutoring group, changed the format after the first month failed, and tracked what improved.",
      "support_score": 58,
      "rationale": "Supported by challenge-response and reflection signals across sections."
    }
  ],
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
      "profile": {
        "narratives": {
          "motivation_letter_text": "I started a tutoring group for younger students."
        }
      },
      "consent": true
    },
    {
      "candidate_id": "cand_002",
      "profile": {
        "narratives": {
          "motivation_letter_text": "I want to study because I need more opportunities."
        }
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
- `ranked_candidate_ids`
- `shortlist_candidate_ids`
- `hidden_potential_candidate_ids`
- `support_needed_candidate_ids`
- `authenticity_review_candidate_ids`
- `results[]`

Batch ranking note:

- `ranked_candidate_ids` is now produced by an offline-trained shortlist ranker artifact
- runtime ranking uses aggregated axes and committee-facing scores, not hand-written pairwise heuristics
- ranker training and evaluation belong to the offline research layer, not to request-time code

## How To Read The Scores

- `merit_score`: candidate promise and strength signals
- `confidence_score`: reliability of the current assessment
- `authenticity_risk`: review-risk signal based on groundedness, consistency, and evidence density
- `evidence_highlights`: strongest evidence-grounded claims surfaced for committee reading
- consistency is now computed not only from generic mismatch, but also from:
  - claim overlap across sections
  - role consistency across sections
  - time and change narrative consistency

These scores are not admission decisions.

`recommendation` is only a workflow routing label:

- `review_priority`
- `standard_review`
- `manual_review_required`
- `insufficient_evidence`
- `incomplete_application`
- `invalid`

## Feature Layers

The service intentionally separates signals into three layers:

1. `Product-facing outputs`
2. `Aggregated scoring signals`
3. `Internal raw features`

Only the first two belong in the public API by default.

Public API principle:

- committee-facing outputs stay compact
- raw internal diagnostics stay inside the scoring layer
- correlated engineering signals are aggregated before exposure

For the current runtime architecture and the exact boundary between heuristics, NLP, and LLM:

- [docs/ML_SERVICE_V2_ARCHITECTURE.md](docs/ML_SERVICE_V2_ARCHITECTURE.md)

## AI Detector

The service supports an optional auxiliary AI-generated text detector:

- `desklib/ai-text-detector-v1.01`

It is used conservatively:

- English-only by default
- weak signal only
- never changes `merit_score`
- never auto-rejects a candidate
- only contributes to authenticity review guidance

This detector is only one weak signal.
The main authenticity layer still relies primarily on:

- groundedness
- evidence density
- section-to-section consistency
- contradiction and mismatch signals

Optional install for local experiments only:

```bash
python -m pip install -r requirements-ai-detector.txt
```

Recommended deployment rule:

- keep `AI_DETECTOR_ENABLED=false` in cloud deployments unless you explicitly want the heavier detector runtime
- do not install detector dependencies on Railway by default

Enable in `.env`:

```env
AI_DETECTOR_ENABLED=true
AI_DETECTOR_MODEL=desklib/ai-text-detector-v1.01
AI_DETECTOR_ENGLISH_ONLY=true
AI_DETECTOR_MIN_WORDS=60
AI_DETECTOR_ELEVATED_PROBABILITY_THRESHOLD=0.80
```

## Optional LLM Explainability

When `ENABLE_LLM=true`, the service can ask the configured LLM for bounded explainability only.

This LLM layer does **not** set final numeric scores or workflow routing.
It is limited to:

- bounded rubric-style reviewer notes
- reviewer-facing strengths and gaps
- one committee follow-up question

Deterministic backend scoring still owns:

- `merit_score`
- `confidence_score`
- `authenticity_risk`
- `recommendation`
- shortlist ordering

Offline calibration and adjudication prompts belong to the `research/calibration/` layer, not to the live `/score` runtime path.

## Project Structure

- `app/api/routes.py`: public HTTP routes
- `app/schemas/`: request and response contracts
- `app/services/`: request-time scoring pipeline, explainability, privacy, authenticity, semantic features
- `app/assets/`: runtime model artifacts such as the offline shortlist ranker weights
- `app/config.py`: environment and scoring config
- `research/`: offline evaluation, annotation analysis, and ranker training utilities
- `tests/`: API and scoring tests

## Important Limits

- The service is a committee-support tool, not an autonomous selector.
- `authenticity_risk` is a review signal, not proof of AI use or cheating.
- LLM output is optional and used only for explanations.
- Missing modalities reduce confidence of assessment, not candidate worth by default.
- Raw internal features are intentionally not exposed in the main response.

## Deployment

Example `Procfile`:

```txt
web: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For Railway or similar platforms:

1. Push the repo.
2. Set environment variables from `.env.example`.
3. Start the service with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

Recommended Railway posture:

- keep `SEMANTIC_BACKEND=hash`
- do not install `requirements-semantic.txt`
- keep `AI_DETECTOR_ENABLED=false`

After deployment, the API docs are available at:

- `/docs`
