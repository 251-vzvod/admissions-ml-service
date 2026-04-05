# inVision U Scoring Service

FastAPI service for primary candidate screening support in the inVision U admissions workflow.

This is a decision-support tool. It does not make autonomous admission decisions.

## Current Snapshot

| Area | Current state |
| --- | --- |
| Product role | Explainable decision-support service for inVision U admissions |
| Public API | `GET /health`, `POST /score`, `POST /score/batch`, `POST /rank` |
| Frozen operational GT | `training_dataset_v3` |
| Labeled candidate rows | `323` |
| Pairwise supervision rows | `1512` |
| Batch shortlist tasks | `54` |
| Runtime ranking posture | Static offline ranker artifact remains the safe runtime default |
| Learned shortlist status | Offline-best aggregate-feature ranker exists and outperforms baseline on key metrics |
| Review-routing status | Separate sidecar exists in shadow mode only |
| Decision policy | Final routing remains deterministic and human-in-the-loop |

## Docs

- [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md): meaning of public scores and signals
- [guides/backend_to_ml_score_request.example.json](guides/backend_to_ml_score_request.example.json): concrete backend request example
- [guides/ml_score_response_public.example.json](guides/ml_score_response_public.example.json): concrete public score response example

## Public API

Routes:

| Route | Purpose |
| --- | --- |
| `GET /health` | Health check |
| `POST /score` | Full score response for one candidate |
| `POST /score/batch` | Full score responses for many candidates, preserving input order |
| `POST /rank` | Batch ranking with shortlist-oriented outputs and optional `top_k` |

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

Core production dependencies now already include:

- `huggingface_hub`
- `langdetect`

OpenAPI docs:

- `http://127.0.0.1:8000/docs`

## Minimal Config

Base deploy-safe defaults:

```env
ENABLE_LLM=false
SEMANTIC_BACKEND=tfidf-char-ngram
AI_DETECTOR_ENABLED=true
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

Recommended lightweight production semantic mode:

```env
SEMANTIC_BACKEND=tfidf-char-ngram
```

Why this is the current production recommendation:

- materially stronger lexical-semantic matching than plain `hash`
- still uses `scikit-learn`, which is already in the main deploy
- avoids `torch` and `sentence-transformers` in production
- keeps Railway-friendly image size and startup time

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

- keep `SEMANTIC_BACKEND=tfidf-char-ngram`
- do not install heavy transformer dependencies in the main service
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

## Research Snapshot

This repository now includes a full offline experimentation layer used to compare the deterministic baseline against learned ranking and routing improvements.

### Current Offline Data Snapshot

Source artifact: `data/ml_workbench/exports/training_dataset_v3_manifest.json`

| Item | Value |
| --- | --- |
| Canonical dataset | `training_dataset_v3` |
| Candidate rows | `323` |
| Train split | `227` |
| Validation split | `50` |
| Test split | `46` |
| Pairwise rows | `1512` |
| Batch shortlist tasks | `54` |

Current source coverage:

| Source | Rows |
| --- | ---: |
| `seed_pack` | 18 |
| `synthetic_batch_v1` | 48 |
| `contrastive_batch_v2` | 24 |
| `translated_batch_v3` | 17 |
| `messy_batch_v4` | 40 |
| `messy_batch_v5` | 60 |
| `messy_batch_v5_extension` | 20 |
| `ordinary_batch_v6` | 24 |
| `gap_fill_batch_v7` | 72 |

### Main Experiment Results: Shortlist Ranker

Source artifact: `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/metrics_summary.json`

Current selected offline-best variant:

- feature variant: `drop_support`
- learned blend alpha: `0.4`
- feature count: `11`

| Metric | Baseline | Learned | Delta |
| --- | ---: | ---: | ---: |
| Validation NDCG@3 | `0.8725` | `0.8946` | `+0.0221` |
| Validation NDCG@5 | `0.9271` | `0.9361` | `+0.0090` |
| Validation pairwise accuracy | `0.6512` | `0.7209` | `+0.0698` |
| Validation hidden-potential recall@3 | `0.8333` | `0.8333` | `+0.0000` |
| Test NDCG@2 | `0.9332` | `0.9289` | `-0.0043` |
| Test NDCG@3 | `0.9299` | `0.9384` | `+0.0084` |
| Test pairwise accuracy | `0.7556` | `0.7556` | `+0.0000` |

Interpretation:

- the learned shortlist ranker improves validation ranking quality over the static baseline
- test-side quality improves on NDCG@3 and holds pairwise flat
- this is enough to show measurable `Baseline & improvements` in a demo or presentation
- the runtime service still keeps the safer static artifact as the default deployment posture

### Main Experiment Results: Review-Routing Sidecar

Source artifact: `data/ml_workbench/exports/models/manual_review_probe_v2_training_dataset_v3/metrics_summary.json`

Current selected offline-best routing sidecar:

- target: `nonstandard_route`
- meaning: `manual_review_required OR insufficient_evidence`
- model: `random_forest_balanced`
- tuned validation threshold: `0.4`

Comparison below uses the simple `authenticity_risk_only` baseline versus the selected sidecar.

| Metric | Baseline | Sidecar | Delta |
| --- | ---: | ---: | ---: |
| Validation average precision | `0.3114` | `0.5950` | `+0.2836` |
| Validation ROC AUC | `0.6000` | `0.7300` | `+0.1300` |
| Test average precision | `0.2049` | `0.6091` | `+0.4042` |
| Test ROC AUC | `0.5897` | `0.8791` | `+0.2894` |

Interpretation:

- routing for non-standard cases is learnable and materially better than a simple authenticity-only baseline
- this sidecar remains shadow-only by design because routing is higher-risk than shortlist ordering
- it is used for internal comparison and future calibration work, not as an autonomous decision layer

### Caveats

| Topic | Current truth |
| --- | --- |
| Ground truth quality | Time-boxed operational GT, not institutional gold |
| Supervision mix | Real reviewed rows plus synthetic and translated coverage |
| Runtime decision ownership | Deterministic recommendation logic remains authoritative |
| ML role | Ranking and routing support, not autonomous admission decisions |

### Deployment Experiment Notes

The team also tested a separate `sentence-transformers` service for heavier semantic features.

Key result:

- this was not selected for production deployment because the Railway image exceeded the free-tier build limit
- current operational choice is therefore `SEMANTIC_BACKEND=tfidf-char-ngram`
- this gives a better quality/simplicity tradeoff than plain `hash` while staying deploy-safe

If you need the full experiment narrative, see:

- `research/roadmap.md`
- `research/baseline_log.md`
- `research/eval_protocol.md`

## GT Results: `SEMANTIC_BACKEND=tfidf-char-ngram` (2026-04-05)

Run configuration:

- `SEMANTIC_BACKEND=tfidf-char-ngram`
- `ENABLE_LLM=false`
- scripts:
  - `research/scripts/train_shortlist_ranker_v1.py`
  - `research/scripts/train_manual_review_probe_v2.py`

Artifacts used for metrics:

- `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/metrics_summary.json`
- `data/ml_workbench/exports/models/manual_review_probe_v2_training_dataset_v3/metrics_summary.json`

### Shortlist Ranker (baseline vs learned)

| Metric | Baseline | Learned | Delta |
| --- | ---: | ---: | ---: |
| Validation NDCG@3 | `0.8794` | `0.8946` | `+0.0152` |
| Validation NDCG@5 | `0.9218` | `0.9361` | `+0.0143` |
| Validation pairwise accuracy | `0.6512` | `0.6977` | `+0.0465` |
| Test NDCG@3 | `0.9252` | `0.9433` | `+0.0181` |
| Test pairwise accuracy | `0.7556` | `0.7778` | `+0.0222` |

### Review-Routing Sidecar Snapshot

Selected sidecar from this tfidf run:

- target: `review_risk_or_insufficient`
- model: `random_forest_balanced`
- tuned threshold: `0.35`

Comparison (baseline `authenticity_risk_only` vs selected sidecar default-threshold metrics):

| Metric | Baseline | Sidecar | Delta |
| --- | ---: | ---: | ---: |
| Validation average precision | `0.3234` | `0.5088` | `+0.1854` |
| Validation ROC AUC | `0.5793` | `0.8089` | `+0.2296` |
| Test average precision | `0.2041` | `0.4830` | `+0.2790` |
| Test ROC AUC | `0.5897` | `0.7729` | `+0.1832` |

Why `tfidf-char-ngram` for production (instead of `hash`, without `sentence-transformers`):

- stronger lexical matching than plain hashing while keeping deterministic behavior
- relies on `scikit-learn` already present in the main service dependencies
- avoids heavy `torch` / `sentence-transformers` production footprint
- keeps deploy image and startup profile suitable for lightweight Railway-style hosting
