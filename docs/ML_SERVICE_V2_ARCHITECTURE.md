# ML Service V2 Architecture

This document defines the runtime architecture for the ML service after the refactor toward a cleaner InVisionU-aligned scorer.

## Main Rule

The service is not an essay scorer.

It is a transparent screening-support engine that tries to surface:

- leadership through action
- growth through challenge and adaptation
- community and values orientation
- grounded motivation
- promising candidates who may need support

## Runtime Layers

### 1. Heuristics / Rules

These are deterministic and must stay transparent:

- consent and eligibility gate
- privacy / sensitive-field exclusion
- text normalization and source segmentation
- missingness / source coverage checks
- contradiction and cross-section consistency checks
- final score aggregation
- workflow recommendation and committee routing

Heuristics should also own:

- explainable penalties
- confidence calibration
- audit traces

They must not pretend to be semantic understanding.

### 2. NLP Models

These should handle bounded semantic tasks, not final authority:

- chunk-level semantic matching to rubric dimensions
- claim-to-evidence retrieval
- offline-trained ranking support for shortlist ordering
- static ranker artifacts loaded at runtime after offline validation

Target semantic dimensions:

- leadership_potential
- growth_trajectory
- motivation_authenticity
- authenticity_groundedness
- community_orientation

Important:

- document / portfolio / video presence is not merit
- NLP should operate on candidate content, not on resource proxies

### 3. LLM

LLM is optional and must stay outside final numeric scoring.

LLM may be used for:

- reviewer card generation
- bounded rubric commentary
- follow-up interview question
- evidence-grounded summary for committee members
- structured extraction from long transcripts when needed

LLM must not be used for:

- final merit score
- final shortlist authority
- final authenticity verdict
- hidden demographic inference

## Canonical Input

Runtime input is one candidate record:

- `candidate_id`
- `profile`
- `consent`

Canonical `profile` sections:

- `academics`
- `materials`
- `narratives`
- `process_signals`
- `metadata`

The important information-bearing modalities are:

- `profile.narratives.motivation_letter_text`
- `profile.narratives.motivation_questions`
- `profile.narratives.interview_text`
- `profile.narratives.video_interview_transcript_text`
- `profile.narratives.video_presentation_transcript_text`

Legacy payloads with `structured_data`, `text_inputs`, `behavioral_signals`, and top-level `metadata` are normalized into this canonical profile contract before scoring.

Current runtime assumption:

- applicants submit their materials in English
- runtime scoring is optimized for English content
- multilingual support is not a current product goal

Structured data should stay secondary and narrow:

- academic readiness
- application completeness
- support-planning context

It should not dominate merit.

## Canonical Runtime Outputs

The service should keep a small set of committee-facing outputs:

- `merit_score`
- `confidence_score`
- `authenticity_risk`
- `recommendation`
- `hidden_potential_score`
- `support_needed_score`
- `shortlist_priority_score`
- `trajectory_score`
- `evidence_coverage_score`
- evidence-grounded claims and reviewer guidance

Internal feature clutter should stay internal.

## Routing Policy

Recommendation, shortlist surfacing, hidden-potential surfacing, and support-needed surfacing should not be separate ad hoc rule sets.

They should all derive from one shared policy layer:

- `priority_band`
- `shortlist_band`
- `hidden_potential_band`
- `support_needed_band`
- `authenticity_review_band`
- `insufficient_evidence_band`

Derived scores help explain the policy, but the policy owns the final routing behavior.

Policy thresholds should live in centralized config, not as scattered magic numbers across routing modules.

## Scoring Philosophy

The merit core should be driven by a small number of interpretable axes:

- `potential`
- `motivation`
- `leadership_agency`
- `community_values`
- `experience_skills`
- `trust_completeness`

These axes are then aggregated into:

- merit
- confidence
- authenticity review risk

## Explicit Non-Goals

The runtime service should not reward:

- polished language by itself
- prestigious packaging
- more attachments as a direct merit gain
- portfolio/video presence as a direct merit gain
- abstract ambition without grounded contribution

## Default Operating Mode

Default runtime mode should be:

- deterministic scoring first
- lightweight semantic retrieval enabled through the hash backend by default
- sentence-transformer semantic mode available as an opt-in heavier deployment path
- LLM disabled unless explicitly enabled

This keeps the service reproducible, testable, and deployable.
