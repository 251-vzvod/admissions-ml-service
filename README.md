# inVision U Scoring Service

FastAPI service for primary candidate screening support in the inVision U admissions workflow.

This is a decision-support tool. It does not make autonomous admission decisions.

## Docs

- [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md): meaning of public scores and signals
- [guides/backend_to_ml_score_request.example.json](guides/backend_to_ml_score_request.example.json): concrete backend request example
- [guides/ml_score_response_public.example.json](guides/ml_score_response_public.example.json): concrete public score response example

## Public API

Routes:

- `GET /health`
- `POST /score`
- `POST /score/batch`
- `POST /rank`

`/rank` behavior:

- without `top_k`: returns the full batch in ranked order
- with `top_k`, for example `POST /rank?top_k=10`: still ranks the full batch, but only returns the first `N` ranked candidates
- shortlist outputs remain batch-level shortlist signals, not just "top N"
- `top N` and `shortlist` are related but not identical:
  - `top N` = first `N` candidates by batch rank
  - `shortlist_candidate_ids` = candidates that pass shortlist policy logic across the full batch

Main public outputs:

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
- `trajectory_score`
- `evidence_coverage_score`
- `committee_cohorts`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`
- `evidence_highlights`
- `top_strengths`
- `main_gaps`
- `explanation`

Ranking outputs:

- `ranked_candidate_ids`
- `ranked_candidates`
- `shortlist_candidate_ids`
- `hidden_potential_candidate_ids`
- `support_needed_candidate_ids`
- `authenticity_review_candidate_ids`
- `ranker_metadata`

Runtime posture:

- deterministic scoring owns final routing and ranking
- sensitive and socio-economic fields are excluded from merit scoring
- LLM is optional and used only for explainability
- documents, portfolio links, and video presence are not direct merit boosts
- `/score/batch` is the full batch scoring endpoint
- `/rank` is the ranking endpoint
- optional review-routing sidecar can run in shadow mode without changing public recommendation logic

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Optional local installs:

- tests: `python -m pip install -r requirements-dev.txt`
- heavier semantic mode: `python -m pip install -r requirements-semantic.txt`
- AI detector experiments: `python -m pip install -r requirements-ai-detector.txt`

OpenAPI docs:

- `http://127.0.0.1:8000/docs`

## Minimal Config

Base deploy-safe defaults:

```env
ENABLE_LLM=false
SEMANTIC_BACKEND=hash
AI_DETECTOR_ENABLED=false
ENABLE_REVIEW_ROUTING_SIDECAR=false
```

Optional remote LLM mode:

```env
ENABLE_LLM=true
LLM_PROVIDER=openai-compatible
LLM_MODEL=glm-4.6:cloud
LLM_BASE_URL=https://ollama.com/v1
LLM_API_KEY=your_ollama_api_key_here
```

Optional local Ollama mode:

```env
ENABLE_LLM=true
LLM_PROVIDER=openai-compatible
LLM_MODEL=llama3.1:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
```

Optional heavier semantic mode:

```env
SEMANTIC_BACKEND=sentence-transformer
SEMANTIC_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Optional review-routing shadow mode:

```env
ENABLE_REVIEW_ROUTING_SIDECAR=true
REVIEW_ROUTING_SIDECAR_ARTIFACT=review_routing_sidecar_v1
```

Important:

- this sidecar is shadow-only by default
- it does not override deterministic `recommendation`
- it is meant for internal comparison and routing research

## Request Shape

Public payload:

```json
{
  "candidate_id": "cand_001",
  "structured_data": {
    "education": {
      "english_proficiency": {
        "type": "ielts",
        "score": 7.0
      }
    }
  },
  "text_inputs": {
    "motivation_letter_text": "I started a small student initiative and kept improving it after the first version failed."
  }
}
```

Public top-level fields:

- `candidate_id`
- `structured_data`
- `text_inputs`
- `behavioral_signals`
- `metadata`

`structured_data` sections:

- `education`

`text_inputs` sections:

- `motivation_letter_text`
- `motivation_questions`
- `interview_text`

The public request contract is intentionally minimal and only includes the fields above.

Canonical `profile` payloads are still accepted for backward compatibility and normalized into the same internal runtime shape.

## Ranking Endpoints

### `POST /score/batch`

Use this when you need:

- full scored responses for each candidate
- the same per-candidate score payloads as `/score`, but for many candidates at once
- input-order preservation for downstream systems that already manage their own ordering

Example request:

```json
{
  "candidates": [
    {
      "candidate_id": "cand_batch_001",
      "text_inputs": {
        "motivation_letter_text": "I started a tutoring group and improved the format after attendance dropped.",
        "interview_text": "I can explain what failed first and what I changed."
      }
    },
    {
      "candidate_id": "cand_batch_002",
      "text_inputs": {
        "motivation_letter_text": "I want a stronger university environment for growth and peer learning."
      }
    }
  ]
}
```

Example response shape:

```json
{
  "scoring_run_id": "score_run_x",
  "scoring_version": "v1.4.0",
  "count": 2,
  "results": [
    {
      "candidate_id": "cand_batch_001",
      "merit_score": 67,
      "confidence_score": 61,
      "authenticity_risk": 28,
      "recommendation": "review_priority"
    },
    {
      "candidate_id": "cand_batch_002",
      "merit_score": 49,
      "confidence_score": 44,
      "authenticity_risk": 31,
      "recommendation": "standard_review"
    }
  ]
}
```

### `POST /rank`

Use this when you need:

- compact batch ranking without full per-candidate explanation payloads
- ranked candidate IDs
- lightweight ranked candidate summaries
- shortlist-related batch slices

Default behavior:

- returns all candidates in ranked order

Optional query parameter:

- `top_k`

Example:

```txt
POST /rank?top_k=10
```

Important:

- the service still scores and ranks the full batch first
- `top_k` only truncates the returned ranked list
- this avoids changing the underlying order depending on response size

Example request:

```json
{
  "candidates": [
    {
      "candidate_id": "cand_rank_001",
      "text_inputs": {
        "motivation_letter_text": "I run a small tutoring group, track attendance, and change the plan when it does not work.",
        "interview_text": "I can explain what improved after feedback."
      }
    },
    {
      "candidate_id": "cand_rank_002",
      "text_inputs": {
        "motivation_letter_text": "I want to join a strong community and develop leadership and impact."
      }
    }
  ]
}
```

Example response shape:

```json
{
  "scoring_run_id": "score_run_x",
  "scoring_version": "v1.4.0",
  "count": 2,
  "returned_count": 2,
  "ranked_candidate_ids": ["cand_rank_001", "cand_rank_002"],
  "ranked_candidates": [
    {
      "candidate_id": "cand_rank_001",
      "rank_position": 1,
      "recommendation": "review_priority",
      "merit_score": 68,
      "confidence_score": 63,
      "authenticity_risk": 24,
      "shortlist_priority_score": 72,
      "is_shortlist_candidate": true
    },
    {
      "candidate_id": "cand_rank_002",
      "rank_position": 2,
      "recommendation": "standard_review",
      "merit_score": 47,
      "confidence_score": 42,
      "authenticity_risk": 33,
      "shortlist_priority_score": 46,
      "is_shortlist_candidate": false
    }
  ],
  "shortlist_candidate_ids": ["cand_rank_001"],
  "hidden_potential_candidate_ids": ["cand_rank_001"],
  "support_needed_candidate_ids": [],
  "authenticity_review_candidate_ids": [],
  "ranker_metadata": {
    "version": "offline-shortlist-ranker-v1",
    "feature_count": 11,
    "top_k_applied": null,
    "full_ranked_count": 2
  }
}
```

## Score Semantics

`recommendation` is a workflow routing label, not an admission decision:

- `review_priority`
- `standard_review`
- `manual_review_required`
- `insufficient_evidence`
- `incomplete_application`
- `invalid`

Important limits:

- `authenticity_risk` is a review signal, not proof of cheating or AI use
- missing modalities reduce confidence, not candidate worth by default
- raw internal features are intentionally hidden from the public response

## Deployment

Example start command:

```txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Recommended Railway posture:

- keep `SEMANTIC_BACKEND=hash`
- do not install `requirements-semantic.txt`
- keep `AI_DETECTOR_ENABLED=false`
- if enabling LLM in deploy, use an OpenAI-compatible remote endpoint such as Ollama Cloud:
  - `ENABLE_LLM=true`
  - `LLM_PROVIDER=openai-compatible`
  - `LLM_MODEL=glm-4.6:cloud`
  - `LLM_BASE_URL=https://ollama.com/v1`
  - `LLM_API_KEY=<your_ollama_api_key>`

## Project Structure

- `app/api/routes.py`: HTTP routes
- `app/schemas/`: request and response contracts
- `app/services/`: scoring, explainability, authenticity, privacy, semantic features
- `app/assets/`: runtime artifacts such as shortlist ranker weights
- `app/config.py`: environment and scoring config
- `guides/`: JSON request/response examples only
- `.local/`: ignored local workspace for research, scripts, and archive docs
- `tests/`: API and scoring tests
