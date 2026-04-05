# Slides Plan For The inVision U Scoring Service

## Purpose

This file is a presentation blueprint for explaining the service end-to-end:

- product logic
- deterministic scoring backbone
- NLP and LLM layers
- AI detection
- offline ML experiments
- privacy / fairness posture
- production deployment tradeoffs

It is written to help build a deck that is understandable to judges, product reviewers, and non-ML stakeholders.

## Recommended Deck Length

Best target:

- `10-12 core slides`
- `2-4 appendix slides`

If time is tight, keep only the first `10` slides.

## Slide 1. Problem And Goal

### Slide title

`Decision Support For inVision U Admissions`

### What to say

- Admissions teams do not just need ranking.
- They need a system that surfaces hidden potential, explains itself, and stays human-in-the-loop.
- The service is not an autonomous admissions model.
- It is a committee-support system for triage, prioritization, and reviewer guidance.

### Visual

- One sentence problem statement
- One sentence solution statement
- Optional flow:
  - `candidate application -> explainable scoring -> reviewer guidance -> committee decision`

### Backing sources

- `README.md`
- `research/roadmap.md`

## Slide 2. Product Principles

### Slide title

`What We Optimized For`

### What to say

- Explainability over black-box novelty
- Human-in-the-loop over autonomous decisions
- Trajectory and hidden potential over polish
- Merit-safe evaluation over socio-economic profiling
- Deployability on lightweight infra

### Visual

- 5 short product principles as tiles

### Backing sources

- `research/roadmap.md`
- `FEATURE_DICTIONARY.md`

## Slide 3. Public API And Committee Outputs

### Slide title

`What The Service Actually Returns`

### What to say

The service exposes:

- `GET /health`
- `POST /score`
- `POST /score/batch`
- `POST /rank`

The key committee-facing outputs are:

- `merit_score`
- `confidence_score`
- `authenticity_risk`
- `recommendation`
- `hidden_potential_score`
- `support_needed_score`
- `shortlist_priority_score`
- `trajectory_score`
- `committee_cohorts`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`

### Visual

- compact response card mockup
- 1 example candidate output

### Backing sources

- `README.md`
- `FEATURE_DICTIONARY.md`
- `guides/ml_score_response_public.example.json`

## Slide 4. Runtime Architecture

### Slide title

`How Scoring Works End To End`

### What to say

The runtime path is layered:

1. normalize request
2. remove protected/sensitive fields
3. preprocess text inputs
4. extract structured features
5. extract text features
6. add semantic rubric features
7. run AI detector
8. estimate authenticity risk
9. compute scores
10. map recommendation
11. build committee guidance
12. optionally improve explanation with LLM

### Visual

- one pipeline diagram with 12 boxes

### Backing sources

- `app/services/pipeline.py`
- `app/services/preprocessing.py`
- `app/services/privacy.py`
- `app/services/scoring.py`
- `app/services/recommendation.py`

## Slide 5. Deterministic Scoring Backbone

### Slide title

`Deterministic Core: Why The System Is Defensible`

### What to say

The authoritative layer is deterministic:

- eligibility stays deterministic
- privacy projection stays deterministic
- scoring and routing stay deterministic
- LLM does not own `recommendation`
- LLM does not own `merit_score`

The main scoring axes include:

- potential
- motivation
- leadership / agency
- community values
- experience / skills
- trust / completeness

### Visual

- a merit breakdown wheel or bar chart

### Backing sources

- `app/services/scoring.py`
- `app/services/eligibility.py`
- `app/services/recommendation.py`
- `research/roadmap.md`

## Slide 6. NLP Layer

### Slide title

`NLP Layer: From Raw Text To Rubric Signals`

### What to say

The system does not rely on raw keyword counting only.

It uses:

- text preprocessing
- heuristic text features
- semantic rubric matching
- prototype-based alignment to:
  - leadership potential
  - growth trajectory
  - motivation authenticity
  - authenticity groundedness
  - community orientation

Current production semantic mode:

- `SEMANTIC_BACKEND=tfidf-char-ngram`

Why:

- better than plain hash
- much lighter than transformer inference
- deploy-safe for Railway-style hosting

### Visual

- source texts on the left
- rubric dimensions on the right
- arrows showing semantic matching

### Backing sources

- `app/services/text_features.py`
- `app/services/semantic_rubrics.py`
- `README.md`

## Slide 7. Explainability Layer

### Slide title

`Explainability: Deterministic Facts + LLM Narrative`

### What to say

The LLM layer is used for explanation, not for final decision ownership.

It helps generate:

- `explanation.summary`
- `top_strengths`
- `main_gaps`
- `committee_cohorts`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`

This gives committee-readable outputs while the underlying score semantics stay controlled.

### Visual

- split view:
  - left: deterministic signals
  - right: committee-friendly narrative

### Backing sources

- `app/services/llm_extractor.py`
- `app/services/llm_committee_writer.py`
- `app/services/llm_prompts.py`
- `app/services/committee_guidance.py`

## Slide 8. Authenticity And AI Detection

### Slide title

`Authenticity Review, Not Auto-Rejection`

### What to say

The system separates:

- groundedness risk
- cross-source mismatch
- contradiction risk
- auxiliary AI-likeness

Important message:

- `authenticity_risk` is not proof of cheating
- AI detector output is a review signal only
- the system uses source-level AI probabilities, not only one global number

### Visual

- authenticity components stack
- small example of `text_ai_probabilities`

### Backing sources

- `app/services/authenticity.py`
- `app/services/ai_detector.py`
- `FEATURE_DICTIONARY.md`

## Slide 9. Privacy And Bias Controls

### Slide title

`How We Reduce Bias`

### What to say

Protected and socio-economic fields are excluded from merit scoring.

Examples already removed:

- income
- social background
- gender / sex
- citizenship
- ethnicity / race
- religion
- direct PII

Also important:

- LLM is not allowed to become the decision maker
- public score semantics remain stable
- deterministic fallback always exists

Be honest about remaining risk:

- proxy bias through polish, fluency, and resource-heavy self-presentation can still exist
- that is why hidden-potential and support-needed logic matters

### Visual

- shield / guardrail slide
- 2 columns:
  - excluded attributes
  - remaining proxy-bias risks

### Backing sources

- `app/services/privacy.py`
- `app/config.py`
- `research/roadmap.md`

## Slide 10. Data And Ground Truth

### Slide title

`What We Trained And Evaluated On`

### What to say

The current operational GT is frozen in `training_dataset_v3`.

Current snapshot:

- `323` labeled candidate rows
- `227` train
- `50` validation
- `46` test
- `1512` pairwise rows
- `54` shortlist batch tasks

Coverage is intentionally broad:

- seed pack
- synthetic
- contrastive
- translated
- messy
- ordinary
- gap-fill

Be explicit:

- this is operational GT for the sprint
- not institutional gold-standard committee ground truth yet

### Visual

- dataset counts table
- source distribution bar chart

### Backing sources

- `data/ml_workbench/exports/training_dataset_v3_manifest.json`
- `data/ml_workbench/labels/bootstrap_label_artifacts_summary.md`
- `data/ml_workbench/labels/generated_batches_refresh_summary.md`

## Slide 11. ML Experiments And Results

### Slide title

`Offline ML Results`

### What to say

There are two distinct stories:

1. successful offline improvements
2. features not promoted to runtime yet

Strong offline results:

- `priority_model_v1` improved:
  - validation Spearman `0.5322 -> 0.8443`
  - test Spearman `0.6586 -> 0.8634`
- routing models improved:
  - shortlist band test F1 `0.1333 -> 0.4167`
  - hidden potential test F1 `0.2667 -> 0.6154`
  - support needed test F1 `0.0000 -> 0.7317`
  - authenticity review test F1 `0.0000 -> 0.6667`

Important nuance:

- `pairwise_ranker_v2` did not pass promotion
- we did not force a weak model into runtime

### Visual

- before/after metric table
- one green badge:
  - `priority_model_v1: promotion pass`
- one red badge:
  - `pairwise_ranker_v2: not promoted`

### Backing sources

- `data/ml_workbench/exports/models/offline_ml_layer_v1_training_dataset_v3/training_report.md`
- `data/ml_workbench/exports/models/offline_ml_layer_v1_training_dataset_v3/promotion_summary.json`
- `data/ml_workbench/exports/models/offline_ml_layer_v1_training_dataset_v3/metrics_summary.json`

## Slide 12. Final Production Architecture

### Slide title

`What Is Actually Running In Production`

### What to say

Production uses the lightweight, defensible stack:

- FastAPI service
- deterministic scoring
- `tfidf-char-ngram` semantic backend
- optional remote LLM for explanation
- Hugging Face Inference API for AI detection
- static offline ranker artifact as safe runtime default

Why not heavy transformer inference in prod:

- image size / deployment constraints
- latency
- operational simplicity

### Visual

- stack logos:
  - Python
  - FastAPI
  - Pydantic
  - scikit-learn
  - LightGBM
  - Hugging Face
  - Railway
  - optional Ollama / OpenAI-compatible

### Backing sources

- `README.md`
- `requirements.txt`
- `app/services/semantic_rubrics.py`
- `app/services/ai_detector.py`

## Slide 13. Why This Service Is Useful

### Slide title

`Value To The Admissions Team`

### What to say

The service helps the committee:

- triage faster
- surface hidden potential
- separate low-evidence from low-merit
- detect authenticity-review cases without auto-rejecting
- ask better follow-up questions
- keep explanations consistent across reviewers

### Visual

- 5 benefit cards

### Backing sources

- `FEATURE_DICTIONARY.md`
- `app/services/committee_guidance.py`
- `app/services/claim_evidence.py`

## Slide 14. Limitations And Next Steps

### Slide title

`What We Know Is Still Incomplete`

### What to say

Be explicit and credible:

- GT is operational, not final institutional gold
- proxy bias through polish and fluency still needs monitoring
- learned offline models are promising, but not all are runtime-ready
- heavy embedding models were explored but not chosen for current production constraints

Next steps:

- collect stronger multi-review committee labels
- expand slice-based fairness evaluation
- promote only the offline models that continue to pass validation and test
- keep the deterministic fallback path

### Visual

- 2 columns:
  - current limitations
  - next iteration

### Backing sources

- `research/roadmap.md`
- `research/eval_protocol.md`
- `README.md`

## Appendix Suggestions

Use appendix slides only if you have time.

### Appendix A

`Score Dictionary`

Use:

- `FEATURE_DICTIONARY.md`

### Appendix B

`Example Candidate Output`

Use one real or synthetic response and annotate:

- recommendation
- why surfaced
- verify manually
- AI probabilities

### Appendix C

`Offline ML Layer Promotion Logic`

Use:

- `data/ml_workbench/exports/models/offline_ml_layer_v1_training_dataset_v3/promotion_summary.json`

### Appendix D

`Data Generation And Coverage`

Use:

- `data/ml_workbench/labels/generated_batches_refresh_summary.md`
- `data/ml_workbench/labels/bootstrap_label_artifacts_summary.md`

## Best 10-Slide Version

If you want the shortest strong version, keep these:

1. Problem And Goal
2. Product Principles
3. Public API And Outputs
4. Runtime Architecture
5. Deterministic Scoring Backbone
6. NLP + LLM Explainability
7. Authenticity + AI Detection
8. Data And Ground Truth
9. Offline ML Results
10. Final Production Architecture + Limitations

## Presentation Advice

- Do not oversell the system as an autonomous selector.
- Repeatedly say `decision support`, `human-in-the-loop`, and `hidden potential`.
- Use exact metrics only where you have artifacts.
- Show one slide with concrete numbers, not metrics on every slide.
- Keep one clear message:
  - `we combined deterministic guardrails, explainable NLP, optional LLM reasoning, and offline ML evaluation into a deployable admissions support service`
