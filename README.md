# inVision U Scoring Service MVP

Production-like prototype of an explainable, deterministic FastAPI scoring service for primary candidate review support.

Important: this service is a decision support tool for committee workflow. It is not an autonomous admission engine.

## What This Service Does

The service ingests structured data and text signals from candidate applications, then produces four different layers of outputs:

1. Candidate-level features (atomic signals in 0.0..1.0)
2. Candidate-level aggregated scores (0..100): `merit_score`, `confidence_score`, `authenticity_risk`
3. Workflow recommendation labels: `review_priority`, `standard_review`, `manual_review_required`, `insufficient_evidence`, `incomplete_application`, `invalid`
4. Explainability payload: strengths, gaps, uncertainties, evidence snippets, and scoring notes

## Architecture

Pipeline stages:

1. Request validation (Pydantic)
2. Privacy / merit-safe projection
3. Preprocessing and normalization
4. Eligibility layer
5. Structured feature extraction
6. Text rubric extraction (heuristic baseline or LLM-assisted extractor)
7. Authenticity risk estimation
8. Feature construction
9. Scoring engine
10. Recommendation mapping
11. Explanation generation

### Baseline vs LLM-Assisted Mode

- `baseline` mode:
  - heuristic text extractor
  - no external provider required
- `llm` mode:
  - LLM extracts structured rubric features only
  - final scores and recommendation are still deterministic and internal

The LLM is an extractor/helper, not the final decision-maker.

Code structure:

- `app/config.py`: thresholds, weights, normalization assumptions, excluded sensitive fields
- `app/schemas`: API input/output contracts
- `app/services`: core pipeline modules
- `app/api/routes.py`: HTTP endpoints
- `scripts/score_candidates.py`: batch scoring and system diagnostics without labels

## Required Endpoints

- `GET /health`
- `GET /config/scoring`
- `POST /score`
- `POST /score/batch`

Additional endpoints included:

- `POST /score/file`
- `POST /debug/features`
- `POST /debug/explanation`
- `POST /debug/llm-extract`

`POST /score` includes additional trace fields:

- `extraction_mode`: `baseline` | `llm`
- `extractor_version`
- `llm_metadata` (nullable)

## Scales And Normalization

The service uses three levels of scale:

1. Feature level: 0.0..1.0
2. Aggregated raw scores: 0.0..1.0
3. Display/API scores: integer 0..100

Utility functions:

- `clamp01`
- `safe_div`
- `to_display_score` where `displayed_score = round(raw * 100)`
- `weighted_average_normalized`

## Product Scores Versus Evaluation Metrics (Critical Distinction)

### Layer A: Candidate Scoring Outputs (Product Scores)

These are operational candidate-level outputs used in committee workflow:

- `merit_score`
- `confidence_score`
- `authenticity_risk`
- `recommendation`

#### `merit_score`

- Measures: strength of potential-related evidence (growth, initiative, leadership agency, motivation fit)
- Does NOT measure: probability of admission; "overall worth" of a person
- Why it exists: prioritization/ranking support for committee review queue
- Inputs: potential and motivation signals, with limited academic influence
- Human interpretation: higher score means stronger observed potential evidence, not guaranteed admission
- Misuse to avoid: treating it as auto admit/reject threshold

#### `confidence_score`

- Measures: reliability of this scoring assessment
- Does NOT measure: candidate quality
- Why it exists: communicate uncertainty and avoid overconfident automation
- Inputs: specificity, evidence count, consistency, completeness, contradiction/low-evidence penalties, soft authenticity penalty
- Human interpretation: low confidence means "needs manual review" more often than "weak candidate"
- Misuse to avoid: demoting candidates solely due to low confidence

#### `authenticity_risk`

- Measures: review-risk / inauthenticity suspicion uncertainty signal
- Does NOT measure: cheating proof or chatbot usage proof
- Why it exists: route cases for manual scrutiny, not final verdict
- Inputs: genericness, evidence density, mismatch signals, contradiction flag
- Human interpretation: elevated risk asks for review; it does not invalidate merit automatically
- Misuse to avoid: direct rejection or hard suppression of merit

#### `recommendation`

- Measures: workflow routing label for committee operations
- Does NOT measure: final admission outcome
- Why it exists: reduce manual triage load with transparent categories
- Inputs: eligibility status + merit/confidence/risk profile
- Human interpretation: operational queue guidance only
- Misuse to avoid: using recommendation as final decision

## Why the LLM Is an Extractor, Not the Judge

LLM usage in this service is intentionally bounded.

The LLM is allowed to:

- read candidate text sections
- extract structured rubric features in `0.0..1.0`
- suggest strengths/gaps/uncertainties and evidence spans

The LLM is not allowed to:

- produce final admission decision
- directly set final recommendation as source of truth
- replace eligibility, privacy filtering, or deterministic scoring formulas

Internal deterministic code computes `merit_score`, `confidence_score`, `authenticity_risk`, and `recommendation` in all modes.

### Layer B: System Evaluation Metrics (Validation Metrics)

These evaluate the scoring system quality itself, not candidates:

- coverage
- missingness rate
- score variance and distribution checks
- recommendation mix
- stability under perturbation tests
- future rank correlation with manual review labels
- future precision@k when labels appear
- future inter-rater agreement proxy
- future confidence calibration-like analysis
- authenticity flag-rate analysis

Without labels, do not invent supervised metrics. Use sanity/stability diagnostics instead.

## Scoring Logic

### Eligibility Status

- `invalid`: missing candidate id, no usable content, or consent false
- `incomplete_application`: too little text for meaningful scoring
- `conditionally_eligible`: minimally scoreable but high missingness
- `eligible`: key sections present

Important: incomplete application is not equivalent to low merit.

### Merit Breakdown Axes (0..100)

- `potential`: growth trajectory, resilience, initiative, program fit
- `motivation`: motivation clarity, program fit, evidence richness
- `leadership_agency`: initiative, leadership impact, linked examples, project mentions
- `experience_skills`: achievements, project mentions, limited academic readiness signals
- `trust_completeness`: completeness, consistency, evidence count minus contradiction/low-evidence penalties

Merit weights are intentionally non-uniform:

- potential: 0.30
- motivation: 0.25
- leadership_agency: 0.20
- experience_skills: 0.15
- trust_completeness: 0.10

Reason: product goal emphasizes potential, growth, and agency over pure formal academic polish.

### Confidence Score

Weighted from:

- specificity score
- evidence count
- consistency score
- completeness score

Minus penalties:

- contradiction
- low evidence
- elevated authenticity risk (soft penalty)

### Authenticity Risk

Raised by:

- polished-but-empty pattern
- high genericness
- low evidence density
- section mismatch
- contradiction risk

Reduced by:

- grounded personal episodes
- consistent narrative across sections
- rich concrete evidence

## Fairness And Privacy

Merit-safe projection explicitly excludes sensitive and socio-economic fields from scoring.

### Excluded Fields

Configured in `app/config.py` (`EXCLUDED_FIELDS`), including:

- names
- personal identifiers (IIN/id number)
- addresses and phones
- social links
- family details
- income/social background
- gender/sex
- citizenship
- race/ethnicity/religion

### Excluded Reasoning

These fields can introduce unfair bias or socio-economic proxies and are not used as merit signals.

## LLM Configuration

Environment flags:

- `ENABLE_LLM`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `LLM_TEMPERATURE`
- `LLM_MAX_RETRIES`
- `LLM_FALLBACK_TO_BASELINE`
- `LLM_BASE_URL`
- `LLM_API_KEY`

Example `.env`:

```env
ENABLE_LLM=true
LLM_PROVIDER=mock
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=20
LLM_TEMPERATURE=0
LLM_MAX_RETRIES=1
LLM_FALLBACK_TO_BASELINE=true
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key_here
```

Fallback behavior:

- if LLM call/parsing fails and fallback is enabled, pipeline uses baseline heuristic extraction
- final scoring and recommendation remain deterministic in both paths

## How To Run

### Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

### Start API

```bash
uvicorn app.main:app --reload
```

### Run Tests

```bash
pytest -q
```

## API Examples

### Single Candidate

```bash
curl -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
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
        {"question": "Why this program?", "answer": "I want to scale social projects with evidence."}
      ],
      "interview_text": "I can explain steps, timeline, and measurable outcomes."
    }
  }'
```

### Batch Scoring

```bash
curl -X POST "http://127.0.0.1:8000/score/batch" \
  -H "Content-Type: application/json" \
  -d @data/candidates.json
```

### File-Based Scoring

```bash
curl -X POST "http://127.0.0.1:8000/score/file?file_path=data/candidates.json"
```

## Example Response Shape

```json
{
  "candidate_id": "cand_001",
  "scoring_run_id": "run_20260329120000_ab12cd34",
  "scoring_version": "v1.0.0",
  "prompt_version": null,
  "extraction_mode": "baseline",
  "extractor_version": "heuristic-extractor-v1",
  "llm_metadata": null,
  "eligibility_status": "eligible",
  "eligibility_reasons": [],
  "merit_score": 74,
  "confidence_score": 61,
  "authenticity_risk": 43,
  "recommendation": "manual_review_required",
  "review_flags": ["low_evidence_density", "moderate_authenticity_risk"],
  "merit_breakdown": {
    "potential": 81,
    "motivation": 84,
    "leadership_agency": 68,
    "experience_skills": 55,
    "trust_completeness": 62
  },
  "feature_snapshot": {
    "motivation_clarity": 0.84,
    "initiative": 0.71,
    "evidence_count": 0.57,
    "consistency_score": 0.74,
    "genericness_score": 0.41,
    "contradiction_flag": false
  },
  "top_strengths": [],
  "main_gaps": [],
  "uncertainties": [],
  "evidence_spans": [],
  "explanation": {
    "summary": "...",
    "scoring_notes": {
      "potential": "...",
      "motivation": "...",
      "confidence": "...",
      "authenticity_risk": "...",
      "recommendation": "..."
    }
  }
}
```

## How To Evaluate MVP Without Ground Truth Labels

Use `scripts/score_candidates.py` to produce:

1. Score distribution report
2. Recommendation distribution
3. Confidence vs completeness sanity check
4. Authenticity risk vs evidence density sanity check
5. Edge-case and perturbation checks:
   - remove examples -> confidence should drop
   - add concrete outcomes -> specificity should rise
   - make text generic -> authenticity risk should rise
6. Missingness/coverage diagnostics
7. No-sensitive-fields-used audit
8. Framework placeholders for future label-based evaluation
9. Extraction success rate / fallback rate / parsing validity rate
10. Baseline-vs-LLM score and recommendation distribution shifts

Run:

```bash
python scripts/score_candidates.py --input data/candidates.json
```

Outputs:

- `data/scored_candidates.json`
- `data/diagnostics_report.json`

## Extension Points Toward LLM Extractor

This repo already supports optional LLM-assisted extraction while keeping deterministic scoring.

Future extension points:

- richer provider adapters in `app/services/llm_client.py`
- prompt/version governance in `app/services/llm_prompts.py`
- stronger parser/contract validation in `app/services/llm_parser.py`
- per-field extractor calibration against reviewer feedback when labels appear

## Limitations

- Heuristic rubric extraction is approximate and not scientific truth
- Contradiction detection is coarse and intentionally conservative
- Domain-specific language variability may reduce recall
- No database/history layer in MVP

## Common Misinterpretations to Avoid

Wrong: "Merit score is admission probability."
Correct: Merit score is evidence-based potential prioritization support.

Wrong: "Low confidence means weak candidate."
Correct: Low confidence means the system has weak evidence reliability and needs manual review.

Wrong: "High authenticity risk proves cheating/AI usage."
Correct: High authenticity risk signals uncertainty and review need, not proof.

Wrong: "Recommendation is final decision."
Correct: Recommendation is workflow routing for human committee.

## Decision Support Notice

This service supports committee decision-making and triage only.
It must operate with human-in-the-loop review and must not be used as autonomous admit/reject automation.
