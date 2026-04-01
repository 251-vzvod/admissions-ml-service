# Data Collection Spec

## Purpose

This document turns the ML roadmap into a concrete data collection plan.

Use it when:

- defining what data to save from applications
- designing reviewer annotation workflows
- preparing offline training datasets
- deciding what labels are needed for ranking, calibration, and semantic improvements

This file is intentionally operational.
The strategy lives in [roadmap.md](./roadmap.md).
First-round reviewer instructions live in [annotation_guide_v1.md](./annotation_guide_v1.md).

## Current Active Scope

For the current v1 data program, active scope is English-only.

Current practical rule:

- use only candidates where `content_profile.language_profile == "english"` from the existing internal seed pack
- treat `mixed` as a future evaluation slice
- keep `russian` out of the first ML training loop
- sanitize the seed pack down to the frozen ML-service public input contract before annotation or model work

Reason:

- current heuristics, semantic prototypes, and auxiliary text logic are most stable on English
- this reduces language noise while we build the first reliable label program
- multilingual expansion should happen after the first English-only baseline is stable
- removing non-contract metadata also reduces leakage risk from synthetic hints such as `scenario_meta.archetype`

## Core Principle

The bottleneck is not more raw text.
The bottleneck is better ground truth.

For this roadmap, we need four layers of data:

1. raw candidate applications
2. human review labels
3. pairwise and batch shortlist judgments
4. calibration outcomes for review-risk and confidence

## Data Packages We Need

### 1. `applications.jsonl`

Purpose:

- canonical offline dataset of candidate applications
- source of raw inputs for feature extraction, NLP, and replay

Format:

- one candidate per JSON line

Required fields:

- `candidate_id`
- `text_inputs`
- `structured_data`
- `behavioral_signals`
- `metadata`

Recommended additional fields for offline analysis only:

- `language_profile`
- `text_length_bucket`
- `has_interview_text`
- `has_transcript`
- `application_completeness_bucket`
- `source_dataset`

Important:

- keep protected or sensitive attributes out of merit features
- if such fields are stored for fairness audit, keep them separated from training features

### 2. `human_labels_individual.csv`

Purpose:

- store independent first-pass reviewer judgments
- capture disagreement before adjudication

Row granularity:

- one reviewer x one candidate

Required columns:

- `annotation_id`
- `candidate_id`
- `reviewer_id`
- `review_round`
- `reviewed_at_utc`
- `recommendation`
- `committee_priority`
- `shortlist_band`
- `hidden_potential_band`
- `support_needed_band`
- `authenticity_review_band`
- `reviewer_confidence`
- `notes`

Recommended columns:

- `evidence_required`
- `language_profile`
- `text_length_bucket`
- `has_interview_text`
- `has_transcript`

Column semantics:

- `recommendation`: one of the public routing labels
- `committee_priority`: integer scale, recommended `1-5`
- `shortlist_band`: boolean
- `hidden_potential_band`: boolean
- `support_needed_band`: boolean
- `authenticity_review_band`: boolean
- `reviewer_confidence`: integer scale, recommended `1-5`
- `notes`: short rationale, not a full essay

### 3. `human_labels_adjudicated.csv`

Purpose:

- final ground-truth table after reviewer disagreement is resolved
- main supervised label source for ranking and calibration

Row granularity:

- one final label row per candidate

Required columns:

- `candidate_id`
- `adjudication_status`
- `adjudicator_id`
- `adjudicated_at_utc`
- `final_recommendation`
- `final_committee_priority`
- `final_shortlist_band`
- `final_hidden_potential_band`
- `final_support_needed_band`
- `final_authenticity_review_band`
- `final_notes`

Recommended columns:

- `reviewer_count`
- `disagreement_flag`
- `disagreement_summary`
- `evidence_span_count`

Column semantics:

- `adjudication_status`: recommended values `agreed`, `resolved_after_disagreement`, `single_reviewer_only`
- `final_committee_priority`: integer scale, recommended `1-5`

### 4. `pairwise_labels.csv`

Purpose:

- train and evaluate ranking models more directly than with scalar labels alone

Row granularity:

- one comparison between candidate A and candidate B

Required columns:

- `pair_id`
- `batch_id`
- `reviewer_id`
- `reviewed_at_utc`
- `candidate_id_left`
- `candidate_id_right`
- `preferred_candidate_id`
- `preference_strength`
- `reason_primary`
- `reason_notes`

Recommended columns:

- `task_type`
- `hidden_potential_preference`
- `authenticity_review_preference`
- `support_needed_preference`

Column semantics:

- `preferred_candidate_id`: must equal either left or right candidate id
- `preference_strength`: recommended `1-3`
- `reason_primary`: recommended controlled values such as `trajectory`, `evidence`, `leadership`, `community`, `authenticity_review`, `support_needed`
- `task_type`: recommended values `read_first`, `shortlist_first`, `hidden_potential`, `manual_review_priority`

### 5. `batch_shortlist_tasks.jsonl`

Purpose:

- capture realistic committee shortlist behavior in context
- evaluate rankers in the same form the product is actually used

Format:

- one shortlist task per JSON line

Required fields:

- `batch_id`
- `task_created_at_utc`
- `reviewer_ids`
- `candidate_ids`
- `selected_shortlist_candidate_ids`
- `ranked_candidate_ids`
- `hidden_potential_candidate_ids`
- `support_needed_candidate_ids`
- `authenticity_review_candidate_ids`
- `notes`

Recommended fields:

- `batch_size`
- `language_mix`
- `task_version`
- `adjudication_status`

Important:

- this dataset is especially valuable for shortlist rankers
- batch and pairwise labels are often more useful than one scalar score

### 6. `evidence_spans.jsonl`

Purpose:

- connect reviewer judgment to exact supporting text
- support explainability evaluation, evidence retrieval, and claim grounding

Format:

- one evidence span per JSON line

Required fields:

- `evidence_id`
- `candidate_id`
- `reviewer_id`
- `label_source`
- `field_name`
- `span_text`
- `span_start`
- `span_end`
- `claim_type`
- `supports_label`

Recommended fields:

- `strength`
- `notes`

Column semantics:

- `label_source`: recommended values `individual_review`, `adjudicated_review`
- `field_name`: recommended values such as `motivation_letter_text`, `motivation_questions`, `interview_text`
- `claim_type`: recommended values `trajectory`, `initiative`, `leadership`, `community`, `authenticity_concern`, `support_needed`
- `supports_label`: boolean

### 7. `calibration_outcomes.csv`

Purpose:

- train and calibrate `authenticity_risk` and `confidence_score`

Row granularity:

- one reviewed candidate outcome

Required columns:

- `candidate_id`
- `review_cycle`
- `manual_review_required`
- `manual_review_trigger_reason`
- `claims_verified`
- `claims_partially_verified`
- `claims_not_verified`
- `material_inconsistency_found`
- `simple_but_sincere_flag`
- `polished_but_thin_flag`
- `final_assessment_reliability`
- `calibration_notes`

Recommended columns:

- `reviewer_id`
- `adjudicator_id`
- `resolved_at_utc`

Column semantics:

- `manual_review_required`: boolean
- `claims_verified`: boolean
- `claims_partially_verified`: boolean
- `claims_not_verified`: boolean
- `material_inconsistency_found`: boolean
- `simple_but_sincere_flag`: boolean
- `polished_but_thin_flag`: boolean
- `final_assessment_reliability`: recommended scale `1-5`

## Minimum Label Program

To start meaningful ML work without flying blind:

- `200-300` fully labeled candidates
- `2` independent reviewers per candidate
- `500-1500` pairwise comparisons
- `50-100` batch shortlist tasks

Better target for a stronger first learned ranker:

- `700+` adjudicated candidates
- stable pairwise and batch labels
- one held-out split untouched by iteration

## Labeling Rules

- reviewers should not see model scores while labeling
- the rubric must stay fixed during one annotation round
- every important verdict should be backed by evidence text
- disagreement should be preserved first, then resolved
- pairwise and batch tasks should be sampled from realistic committee mixes
- protected attributes must not be used as merit features

## Recommended Controlled Vocabularies

### `recommendation`

- `review_priority`
- `standard_review`
- `manual_review_required`
- `insufficient_evidence`
- `incomplete_application`
- `invalid`

### `reason_primary`

- `trajectory`
- `evidence`
- `leadership`
- `community`
- `motivation`
- `authenticity_review`
- `support_needed`
- `completeness`

### `claim_type`

- `trajectory`
- `initiative`
- `leadership`
- `community`
- `motivation`
- `authenticity_concern`
- `support_needed`

## Which Data Supports Which Phase

### Roadmap Phase 1

Required:

- `applications.jsonl`
- `human_labels_individual.csv`
- `human_labels_adjudicated.csv`
- `pairwise_labels.csv`
- `batch_shortlist_tasks.jsonl`

### Roadmap Phase 2

Required:

- `applications.jsonl`
- `evidence_spans.jsonl`
- language and text-quality slices for offline evaluation

### Roadmap Phase 3

Required:

- `human_labels_adjudicated.csv`
- `pairwise_labels.csv`
- `batch_shortlist_tasks.jsonl`

### Roadmap Phase 4

Required:

- `calibration_outcomes.csv`
- `human_labels_adjudicated.csv`

## What Not To Mistake For Ground Truth

These are useful, but they are not enough by themselves:

- synthetic candidate packs
- model-produced scores
- LLM judgments without human review
- final admit / reject outcome without context

Important:

- `admit / reject` is usually too entangled with capacity, budget, interview outcomes, and external constraints
- for this product, shortlist and review-routing labels are better first targets than final admission outcome

## Recommended Build Order

- [ ] Create empty templates for all seven datasets.
- [ ] Freeze scales and controlled vocabularies.
- [ ] Run a pilot annotation round on `25-50` candidates.
- [ ] Measure disagreement and revise rubric only once after the pilot.
- [ ] Start collecting adjudicated labels.
- [ ] Add pairwise and batch tasks once the single-candidate rubric is stable.
- [ ] Begin model training only after the first adjudicated dataset is coherent.
