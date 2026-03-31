# Dataset Schema V2

This document defines the recommended file layout for a realistic InVisionU screening benchmark.

## Core Files

### `candidates.csv`

One row per candidate package.

Columns:

- `candidate_id`
- `admission_cycle`
- `program_track`
- `application_language`
- `application_complete`
- `has_video`
- `has_interview`
- `has_roleplay`
- `consent_for_ml_review`

### `candidate_structured.csv`

One row per candidate.

Columns:

- `candidate_id`
- `education_stage`
- `english_level_reported`
- `academic_record_summary`
- `activities_summary`
- `projects_summary`
- `responsibilities_summary`
- `constraints_summary`

### `candidate_texts.jsonl`

One JSON object per candidate.

Fields:

- `candidate_id`
- `motivation_letter_text`
- `motivation_questions`
- `video_presentation_transcript_text`
- `interview_text`
- `roleplay_notes_text`
- `additional_materials_text`

### `annotations_individual.csv`

One row per annotator per candidate.

Columns:

- `candidate_id`
- `annotator_id`
- `annotator_role`
- `leadership_through_action`
- `growth_trajectory`
- `community_orientation`
- `project_based_readiness`
- `motivation_groundedness`
- `evidence_strength`
- `shortlist_priority`
- `hidden_potential_flag`
- `needs_support_flag`
- `authenticity_review_needed`
- `label_confidence`
- `annotation_notes`

### `annotations_adjudicated.csv`

One row per candidate after adjudication.

Columns:

- `candidate_id`
- `adjudicator_id`
- `final_leadership_through_action`
- `final_growth_trajectory`
- `final_community_orientation`
- `final_project_based_readiness`
- `final_motivation_groundedness`
- `final_evidence_strength`
- `final_shortlist_priority`
- `final_hidden_potential_flag`
- `final_needs_support_flag`
- `final_authenticity_review_needed`
- `final_label_confidence`
- `adjudication_notes`

### `annotation_evidence.jsonl`

One JSON object per evidence span.

Fields:

- `candidate_id`
- `annotator_id`
- `dimension`
- `source_field`
- `snippet`
- `why_it_matters`

## Ranking-Oriented Files

### `pairwise_labels.csv`

Columns:

- `pair_id`
- `left_candidate_id`
- `right_candidate_id`
- `annotator_id`
- `preferred_candidate_id`
- `preference_strength`
- `reason`

### `batch_shortlist_tasks.jsonl`

One object per shortlist task.

Fields:

- `task_id`
- `annotator_id`
- `candidate_ids`
- `selected_top_k`
- `selected_hidden_potential_ids`
- `selected_support_needed_ids`
- `selected_authenticity_review_ids`
- `notes`

## Optional Outcome File

### `student_outcomes.csv`

Use only after enrollment outcomes exist.

Columns:

- `candidate_id`
- `admitted`
- `retained_after_1_term`
- `project_participation_level`
- `mentor_rating`
- `teamwork_rating`
- `initiative_after_admission`
- `support_needed_after_admission`
- `support_response_quality`

## Fairness-Audit File

### `fairness_audit.csv`

Keep this separate from model inputs.

Columns:

- `candidate_id`
- `program_track`
- `language_slice`
- `region_slice`
- `school_resource_slice`
- `support_context_slice`

These fields are for auditing only, not for merit scoring.
