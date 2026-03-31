# Feature Dictionary

## Purpose

This document explains what each major feature or score means, why it exists, and whether it belongs on the committee-facing API surface.

The system uses three layers of signals:

1. `Product-facing outputs`
2. `Aggregated scoring signals`
3. `Internal raw features`

This separation is intentional. Not every internal feature should be shown to users, and some raw features are correlated enough that exposing them directly would create noise.

## Layer 1. Product-Facing Outputs

These are the fields that matter most for the committee, the frontend, and the hackathon demo.

### `merit_score`

- Meaning: overall candidate strength signal
- Used for: general candidate assessment
- Shown to committee: yes
- Notes: should never be interpreted as an autonomous admission decision

### `confidence_score`

- Meaning: how reliable the current assessment is, given the available evidence
- Used for: deciding how much trust to place in the score
- Shown to committee: yes
- Notes: low confidence does not mean low potential

### `authenticity_risk`

- Meaning: manual review risk based on groundedness, consistency, and evidence density
- Used for: authenticity review routing
- Shown to committee: yes
- Notes: not proof of cheating or AI use

### `recommendation`

- Meaning: workflow routing label
- Used for: committee triage
- Shown to committee: yes
- Notes: operational label only

### `hidden_potential_score`

- Meaning: whether the underlying growth and leadership signal appears stronger than the self-presentation quality
- Used for: surfacing underestimated candidates
- Shown to committee: yes
- Notes: one of the main differentiators of the system

### `support_needed_score`

- Meaning: whether a promising candidate may need onboarding, academic, or language support
- Used for: support-aware review
- Shown to committee: yes
- Notes: should reframe some borderline cases as promising, not weak

### `shortlist_priority_score`

- Meaning: shortlist-oriented review priority
- Used for: ranking and manual review order
- Shown to committee: yes
- Notes: more useful than raw merit alone in shortlist workflows

### `trajectory_score`

- Meaning: quality of the candidate's growth path
- Used for: growth-aware selection
- Shown to committee: yes
- Notes: directly aligned with the brief's "trajectory of growth"

### `evidence_coverage_score`

- Meaning: how well key claims are supported across the application
- Used for: separating promising but grounded from polished but thin applications
- Shown to committee: yes

### `committee_cohorts`

- Meaning: high-level case type
- Used for: reviewer workflow and demo clarity
- Shown to committee: yes

### `why_candidate_surfaced`

- Meaning: concise reasons why the candidate was surfaced
- Used for: reviewer orientation
- Shown to committee: yes

### `what_to_verify_manually`

- Meaning: next manual checks the reviewer should make
- Used for: human-in-the-loop workflow
- Shown to committee: yes

### `suggested_follow_up_question`

- Meaning: one useful next interview question
- Used for: committee workflow
- Shown to committee: yes

### `top_strengths`, `main_gaps`, `uncertainties`, `evidence_spans`, `explanation`

- Meaning: explanation layer
- Used for: explainability and reviewer support
- Shown to committee: yes

## Layer 2. Aggregated Scoring Signals

These are useful and interpretable, but secondary.

They can be shown in reviewer details or an expanded UI, but they should not dominate the main product surface.

### `merit_breakdown`

- Meaning: split of overall merit into core axes
- Axes:
  - `potential`
  - `motivation`
  - `leadership_agency`
  - `experience_skills`
  - `trust_completeness`
- Used for: explaining where the score comes from
- Shown to committee: optional expanded view

### `semantic_rubric_scores`

- Meaning: semantic alignment with rubric dimensions
- Dimensions:
  - `leadership_potential`
  - `growth_trajectory`
  - `motivation_authenticity`
  - `authenticity_groundedness`
  - `hidden_potential`
- Used for: rubric-aware reasoning
- Shown to committee: optional expanded view

### `authenticity_review_reasons`

- Meaning: concrete reasons why authenticity review was raised
- Used for: transparency of review routing
- Shown to committee: optional but useful

### `ai_detector`

- Meaning: auxiliary detector payload
- Used for: authenticity support only
- Shown to committee: optional and carefully worded
- Notes: should never be treated as proof

### `llm_metadata`

- Meaning: runtime metadata about the optional explanation call
- Used for: ops and debugging
- Shown to committee: usually no
- Notes: acceptable in API for engineering consumers, but not part of the reviewer-facing story

## Layer 3. Internal Raw Features

These are useful for model design, calibration, error analysis, and diagnostics.

They are not good primary API fields.

### Evidence cluster

- `evidence_count`
- `evidence_richness`
- `specificity_score`
- `linked_examples_count`
- `evidence_count_estimate`

Meaning:

- they describe related aspects of groundedness and support

Risk:

- highly correlated
- easy to double-count if exposed or weighted carelessly

Recommendation:

- keep internal
- surface only aggregated `evidence_coverage_score`

### Trajectory cluster

- `growth_trajectory`
- `resilience`
- `trajectory_challenge_score`
- `trajectory_adaptation_score`
- `trajectory_reflection_score`
- `trajectory_outcome_score`

Meaning:

- they all describe growth path and adaptation

Risk:

- partially correlated
- too many similar numbers are confusing in UI

Recommendation:

- keep internal
- surface aggregated `trajectory_score`

### Presentation and authenticity cluster

- `genericness_score`
- `polished_but_empty_score`
- `cross_section_mismatch_score`
- `consistency_score`
- `contradiction_flag`

Meaning:

- these help identify weakly grounded or suspiciously polished applications

Risk:

- useful internally
- too raw and easy to misread externally

Recommendation:

- keep internal
- surface `authenticity_risk` and `authenticity_review_reasons`

### Material support cluster

- `docs_count_score`
- `portfolio_links_score`
- `has_video_presentation`
- `material_support_score`

Meaning:

- these represent supporting material presence, not candidate worth

Risk:

- dangerous if overinterpreted as potential

Recommendation:

- keep internal
- do not show as primary committee fields
- do not let them dominate merit

### Other raw internal features

- `program_fit`
- `motivation_clarity`
- `initiative`
- `leadership_impact`
- `completeness_score`
- `question_coverage_score`
- `behavioral_completion_score`

Recommendation:

- internal only
- useful for formulas and audits
- not necessary as public API fields

## Intentional API Simplification

The public API should avoid exposing raw feature dumps by default.

That is why `feature_snapshot` is not part of the main committee-facing response anymore.

The same rule applies to internal prompt and extractor versioning: they are useful for engineering diagnostics, but not part of the committee-facing product contract.

## Correlation Note

Yes, the system contains correlated features.

This is acceptable as long as:

- correlated raw features stay internal
- public API surfaces aggregated signals
- scoring weights are monitored to avoid accidental double-counting

The design goal is:

- raw correlated features for transparent engineering
- aggregated rubric signals for scoring
- compact product outputs for committee workflow
