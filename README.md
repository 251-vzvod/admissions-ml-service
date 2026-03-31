# inVision U Scoring Service

FastAPI service for primary candidate screening support in the inVision U admissions workflow.

This is a decision-support tool. It does not make autonomous admission decisions.

## Core Docs

- [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md): meaning of public scores and signals
- [guides/ML_SERVICE_V2_ARCHITECTURE.md](guides/ML_SERVICE_V2_ARCHITECTURE.md): runtime architecture and signal boundaries
- [guides/fairness_note.md](guides/fairness_note.md): fairness posture and known limits
- [guides/BACKEND_TO_ML_CONTRACT.md](guides/BACKEND_TO_ML_CONTRACT.md): backend request/response contract for ML integration
- [guides/FRONTEND_ML_RESPONSE_CONTRACT.md](guides/FRONTEND_ML_RESPONSE_CONTRACT.md): frontend-facing public ML response contract
- [guides/INTEGRATION_SYNC_SUMMARY.md](guides/INTEGRATION_SYNC_SUMMARY.md): short sync note for frontend/backend teams

## Public API

Routes:

- `GET /health`
- `POST /score`
- `POST /score/batch`

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

Runtime posture:

- deterministic scoring owns final routing and ranking
- sensitive and socio-economic fields are excluded from merit scoring
- LLM is optional and used only for explainability
- documents, portfolio links, and video presence are not direct merit boosts

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
```

Optional local LLM mode:

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

Legacy payloads with `structured_data`, `text_inputs`, `behavioral_signals`, and top-level `metadata` are still accepted and normalized into the canonical profile contract.

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

## Project Structure

- `app/api/routes.py`: HTTP routes
- `app/schemas/`: request and response contracts
- `app/services/`: scoring, explainability, authenticity, privacy, semantic features
- `app/assets/`: runtime artifacts such as shortlist ranker weights
- `app/config.py`: environment and scoring config
- `guides/`: lightweight tracked project documentation
- `.local/`: ignored local workspace for research, scripts, and archive docs
- `tests/`: API and scoring tests
