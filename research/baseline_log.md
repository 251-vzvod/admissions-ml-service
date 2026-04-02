# Baseline Log

## Purpose

This file records the current baseline state of the scoring system and the changes made after each roadmap iteration.

Use it for:

- remembering the current technical baseline
- comparing before / after states across iterations
- keeping the project direction stable when context is lost

Strategy lives in [roadmap.md](./roadmap.md).
Data requirements live in [data_collection_spec.md](./data_collection_spec.md).

## How To Update This File

For every meaningful roadmap iteration, append one new section with:

- what changed
- what stayed deterministic
- what data or labels were used
- what metrics improved or regressed
- what risks remain

Do not overwrite the baseline snapshot below.
New work should be logged after it.

## Iteration 0: Current Baseline Snapshot

### Snapshot Metadata

- captured_on: `2026-04-01`
- branch: `main`
- base_commit: `089f21b`
- worktree_state: `dirty`

Current local uncommitted changes at capture time:

- `app/config.py`
- `app/services/pipeline.py`
- `app/services/shortlist.py`
- `research/` additions and docs

Important:

- this snapshot reflects the current working tree, not only the last commit

### Baseline Architecture

Current system shape:

- deterministic request normalization
- deterministic eligibility logic
- deterministic privacy projection
- heuristic + rule-based text and structured feature extraction
- lightweight semantic rubric layer
- deterministic score aggregation
- static offline shortlist ranker artifact
- optional LLM explainability layer

Interpretation:

- the service is already hybrid
- but the learned part is still shallow and mostly artifact-based, not label-rich and not deeply supervised

### What Is Deterministic Right Now

- public input contract and normalization
- eligibility routing
- privacy-safe projection
- main feature extraction pipeline
- merit / confidence / authenticity aggregation logic
- claim-to-evidence backbone
- fallback behavior when optional components are unavailable

### What Is Learned Or Semi-Learned Right Now

- semantic rubric matching uses a lightweight semantic backend
- shortlist ordering uses an offline ranker artifact
- optional LLM explainability can enrich committee-facing explanation text

Important limitation:

- these components are not yet driven by a strong real-world adjudicated label program

### Current Runtime / Config Snapshot

Workspace config values at capture time:

- `scoring_version = v1.4.0`
- `semantic_backend = hash`
- `semantic_model = sentence-transformers/all-MiniLM-L6-v2`
- `llm_enabled = True`
- `llm_fallback_to_baseline = True`
- `ai_detector_enabled = False`

Notes:

- current workspace config is env-dependent and may differ from deploy posture
- baseline scoring below was captured with `enable_llm_explainability=False` to isolate deterministic scoring behavior

### Current Public Contract Snapshot

Public top-level request fields:

- `candidate_id`
- `structured_data`
- `text_inputs`
- `behavioral_signals`
- `metadata`

Current public request content shape:

- `structured_data.education`
- `text_inputs.motivation_letter_text`
- `text_inputs.motivation_questions`
- `text_inputs.interview_text`

### Detailed Candidate Scoring Logic

This section is the technical reference for how a single candidate is scored in the current baseline.

#### 1. End-to-End Pipeline Order

Current `ScoringPipeline` flow:

1. normalize candidate payload into the internal runtime shape
2. remove excluded sensitive fields from merit scoring
3. preprocess and normalize text inputs into a text bundle
4. run deterministic eligibility gate
5. if status is `invalid` or `incomplete_application`, early-exit with zero scores
6. extract structured features
7. extract text heuristic features
8. extract semantic rubric features
9. optionally run auxiliary AI-detector
10. estimate `authenticity_risk_raw`
11. compute `merit_raw`, `confidence_raw`, and merit breakdown
12. build shortlist-oriented derived scores
13. build routing policy snapshot
14. map everything into `recommendation`
15. build claim evidence, committee guidance, and explanation payload
16. optionally enrich explanation with LLM output

Important:

- `conditionally_eligible` candidates still continue through scoring
- only `invalid` and `incomplete_application` trigger the early zero-score path

#### 2. Eligibility Gate Logic

Current deterministic rules:

- if `consent is False` -> `invalid`
- if `candidate_id` is missing or blank -> `invalid`
- if `word_count == 0` or `non_empty_text_sources == 0` -> `invalid`
- if `word_count < 40` -> add reason `text_too_short_for_reliable_scoring`
- if `non_empty_sources < 1` -> add reason `insufficient_text_sources`
- optional document / portfolio / video requirements can add missing-materials reasons, but defaults are conservative
- if `word_count < 20` -> `incomplete_application`
- if `logical_groups_present < 2` -> add reason `insufficient_multi_source_evidence`
- if `logical_groups_missing >= 2` or any reasons exist -> `conditionally_eligible`
- else -> `eligible`

Interpretation:

- eligibility is a formal completeness / admissibility gate
- it is not a merit estimate

#### 3. Current Feature Families

Structured features currently include:

- `english_score_normalized`
- `certificate_score_normalized`
- `text_completeness_score`
- `question_coverage_score`
- `behavioral_completion_score`
- `returned_to_edit_flag`
- `skipped_optional_questions_penalty`
- `docs_count_score`
- `portfolio_links_score`
- `has_video_presentation`
- `evidence_count_estimate`
- `linked_examples_count`
- `achievement_mentions_count`
- `project_mentions_count`
- `verbosity_density_score`

Text heuristic features currently include:

- `motivation_clarity`
- `initiative`
- `leadership_impact`
- `growth_trajectory`
- `resilience`
- `program_fit`
- `evidence_richness`
- `community_value_orientation`
- `specificity_score`
- `evidence_count`
- `trajectory_challenge_score`
- `trajectory_adaptation_score`
- `trajectory_reflection_score`
- `trajectory_outcome_score`
- `section_claim_overlap_score`
- `section_role_consistency_score`
- `section_time_consistency_score`
- `consistency_score`
- `completeness_score`
- `genericness_score`
- `contradiction_flag`
- `low_evidence_flag`
- `polished_but_empty_score`
- `cross_section_mismatch_score`

Semantic rubric features currently include:

- `semantic_leadership_potential`
- `semantic_growth_trajectory`
- `semantic_motivation_authenticity`
- `semantic_authenticity_groundedness`
- `semantic_community_orientation`
- `semantic_hidden_potential`

Current semantic layer behavior:

- semantic dimensions are matched against positive and negative rubric prototypes
- each dimension score is based on the top semantic chunk matches
- current workspace backend is `hash`
- stronger semantic backends can be swapped in later without changing the public API

#### 4. Authenticity Risk Logic

`authenticity_risk` is a review-risk / uncertainty signal, not proof of cheating.

Current raw formula:

```text
fairness_discount =
  0 if evidence_count < 0.25 or consistency < 0.45
  else clamp01(
    0.55 * section_claim_overlap_score +
    0.25 * section_role_consistency_score +
    0.20 * section_time_consistency_score
  ) * 0.16

risk =
  genericness_score * 0.35 * (1 - fairness_discount)
  + (1 - evidence_count) * 0.25
  + (1 - consistency_score) * 0.18
  + polished_but_empty_score * 0.12 * (1 - fairness_discount * 0.85)
  + cross_section_mismatch_score * 0.10
  + optional section-overlap penalties
  + 0.12 if long_but_thin polished-empty pattern
  + 0.15 if contradiction_flag
  + optional AI-detector penalty
  - 0.12 if evidence_count > 0.65 and consistency_score > 0.65

authenticity_risk_raw = clamp01(risk)
authenticity_risk = round(authenticity_risk_raw * 100)
```

Current flag behavior:

- `MODERATE_AUTHENTICITY_RISK` if `0.45 <= raw < 0.70`
- `HIGH_AUTHENTICITY_RISK` if `raw >= 0.70`
- additional flags fire for polished-but-empty, genericness, mismatch, contradiction, low evidence, and optional AI-detector signal

Interpretation:

- the system currently penalizes generic, weakly grounded, cross-section inconsistent applications
- a fairness discount reduces over-penalization when evidence and consistency are reasonably strong

#### 5. Merit Breakdown Logic

All core scores are computed on normalized raw values in `[0, 1]`, then converted with:

```text
display_score = round(clamp01(raw) * 100)
```

Current axis formulas:

```text
potential =
  wavg(
    growth_trajectory 0.22,
    resilience 0.18,
    initiative 0.14,
    leadership_impact 0.10,
    evidence_richness 0.08,
    program_fit 0.06,
    semantic_growth_trajectory 0.12,
    semantic_leadership_potential 0.08,
    semantic_hidden_potential 0.12
  )

motivation =
  wavg(
    motivation_clarity 0.20,
    program_fit 0.12,
    evidence_richness 0.16,
    specificity_score 0.12,
    consistency_score 0.10,
    semantic_motivation_authenticity 0.18,
    semantic_authenticity_groundedness 0.12
  )

leadership_agency =
  wavg(
    initiative 0.20,
    leadership_impact 0.18,
    evidence_richness 0.14,
    evidence_count 0.10,
    project_mentions_count 0.10,
    trajectory_outcome_score 0.08,
    community_value_orientation 0.06,
    semantic_leadership_potential 0.14,
    semantic_hidden_potential 0.10
  )

community_values =
  wavg(
    community_value_orientation 0.34,
    program_fit 0.14,
    motivation_clarity 0.10,
    leadership_impact 0.10,
    initiative 0.10,
    evidence_richness 0.12,
    semantic_community_orientation 0.10
  )

experience_skills =
  wavg(
    specificity_score 0.24,
    evidence_count 0.20,
    evidence_richness 0.18,
    project_mentions_count 0.14,
    achievement_mentions_count 0.08,
    trajectory_outcome_score 0.08,
    trajectory_adaptation_score 0.08
  )
```

Current trust / completeness base:

```text
trust_base =
  wavg(
    completeness_score 0.24,
    consistency_score 0.24,
    specificity_score 0.10,
    evidence_count 0.16,
    behavioral_completion_score 0.10,
    source_support_score 0.08,
    text_completeness_score 0.08,
    semantic_authenticity_groundedness 0.08
  )
```

Current trust penalties:

```text
trust_penalty =
  contradiction_penalty
  + low_evidence_penalty
  + genericness_penalty
  + polished_empty_penalty
  + mismatch_penalty
  + unsupported_narrative_penalty

trust_completeness = clamp01(trust_base - trust_penalty)
```

Current merit breakdown weights from config:

- `potential = 0.22`
- `motivation = 0.15`
- `leadership_agency = 0.18`
- `community_values = 0.18`
- `experience_skills = 0.15`
- `trust_completeness = 0.12`

So:

```text
merit_base_raw =
  wavg(
    potential 0.22,
    motivation 0.15,
    leadership_agency 0.18,
    community_values 0.18,
    experience_skills 0.15,
    trust_completeness 0.12
  )
```

#### 6. Committee Calibration Signal

Current baseline does not use `merit_base_raw` alone.
It blends it with a committee-oriented calibration signal.

Current committee signal base:

```text
committee_base =
  wavg(
    specificity_score 0.13,
    consistency_score 0.12,
    completeness_score 0.08,
    evidence_richness 0.12,
    evidence_count 0.09,
    practical_action_score 0.16,
    growth_trajectory 0.08,
    resilience 0.06,
    leadership_impact 0.05,
    community_value_orientation 0.06,
    semantic_authenticity_groundedness 0.02,
    semantic_growth_trajectory 0.02,
    semantic_community_orientation 0.05,
    program_fit 0.06
  )
```

Current practical action score:

```text
practical_action_score =
  wavg(
    initiative 0.24,
    leadership_impact 0.18,
    evidence_count 0.18,
    project_mentions_count 0.14,
    trajectory_adaptation_score 0.14,
    trajectory_outcome_score 0.12
  )
```

Current committee adjustments:

- hidden-signal bonus rewards cases where semantic hidden potential is strong but writing specificity is modest
- low-evidence penalty fires more strongly when support is weak
- contradiction, genericness, polished-but-empty, mismatch, unsupported narrative, and authenticity penalties reduce the signal
- a language fairness discount softens some genericness / polish penalties when coherence and cross-section support are decent

Final current formula:

```text
committee_signal_raw = clamp01(committee_base + hidden_bonus - total_penalty)
merit_raw = clamp01(0.56 * merit_base_raw + 0.44 * committee_signal_raw)
merit_score = round(merit_raw * 100)
```

Interpretation:

- the baseline already tries to behave more like a shortlist engine than a pure essay-quality scorer
- this is one of the most important non-trivial parts of the current system

#### 7. Confidence Logic

Current confidence base:

```text
source_support_score =
  wavg(
    text_completeness_score 0.42,
    question_coverage_score 0.24,
    behavioral_completion_score 0.22,
    consistency_score 0.12
  )

confidence_base =
  wavg(
    specificity_score 0.20,
    evidence_count 0.14,
    consistency_score 0.20,
    completeness_score 0.18,
    source_support_score 0.08,
    semantic_authenticity_groundedness 0.10
  )
```

Current confidence penalties:

```text
confidence_penalty =
  0.15 if contradiction_flag
  + (0.10 if low_evidence_flag and source_support < 0.35 else 0.04 if low_evidence_flag else 0)
  + max(0, authenticity_risk_raw - 0.60) * 0.12

confidence_raw = clamp01(confidence_base - confidence_penalty)
confidence_score = round(confidence_raw * 100)
```

Interpretation:

- confidence is not candidate quality
- it is a reliability estimate for how much trust to place in the current assessment

#### 8. Derived Shortlist Scores

Current evidence coverage:

```text
evidence_coverage_raw =
  wavg(
    evidence_count 0.28,
    specificity_score 0.24,
    evidence_richness 0.20,
    consistency_score 0.16,
    completeness_score 0.12
  )
```

Current trajectory score:

```text
trajectory_raw =
  wavg(
    trajectory_challenge_score 0.18,
    trajectory_adaptation_score 0.24,
    trajectory_reflection_score 0.22,
    trajectory_outcome_score 0.08,
    growth_trajectory 0.10,
    resilience 0.08,
    semantic_growth_trajectory 0.07,
    semantic_motivation_authenticity 0.03
  )
```

Current support-needed score:

```text
support_gap = 1 - confidence_score / 100
adaptation_signal =
  wavg(
    growth_trajectory 0.22,
    resilience 0.20,
    trajectory_adaptation_score 0.20,
    trajectory_reflection_score 0.16,
    completeness_score 0.12,
    consistency_score 0.10
  )
promise_floor = merit_score / 100

support_needed_raw =
  clamp01(
    support_gap * 0.50 +
    adaptation_signal * 0.20 +
    promise_floor * 0.30
  )
```

Current hidden-potential score:

```text
underlying_signal =
  wavg(
    trajectory_signal 0.22,
    resilience 0.11,
    initiative 0.08,
    leadership_impact 0.06,
    semantic_hidden_potential 0.22,
    semantic_growth_trajectory 0.18,
    semantic_leadership_potential 0.13
  )

self_presentation =
  wavg(
    motivation_clarity 0.28,
    specificity_score 0.24,
    evidence_richness 0.18,
    evidence_count 0.14,
    completeness_score 0.08,
    semantic_motivation_authenticity 0.08
  )

hidden_potential_raw =
  clamp01(
    underlying_signal * 0.30
    + trajectory_signal * 0.14
    + credible_action_signal * 0.24
    + understatement_gap * 0.14
    + evidence_floor * 0.14
    + modest_presentation * 0.04
    + bounded_action_bonus
    - overstatement_risk * 0.08
    - low_evidence_penalty * 0.20
    - inflated_presentation_penalties
    - authenticity_penalty_if_extreme
  )
```

Interpretation:

- hidden potential is not just "weak writing but promising"
- it specifically tries to reward strong underlying growth and action signals that exceed self-presentation quality

Current shortlist priority:

```text
shortlist_priority_raw =
  wavg(
    merit_score / 100 0.38,
    trajectory_raw 0.18,
    hidden_potential_raw 0.18,
    evidence_coverage_raw 0.12,
    confidence_score / 100 0.08,
    review_priority_indicator 0.06
  )
  - max(0, authenticity_risk / 100 - 0.70) * 0.22

shortlist_priority_score = round(clamp01(shortlist_priority_raw) * 100)
```

#### 9. Policy Bands

Current policy thresholds:

- insufficient evidence band:
  - `confidence_score <= 31`
  - `evidence_coverage_score <= 27`
- authenticity review band:
  - `authenticity_risk >= 45`
- hidden potential band:
  - `hidden_potential_score >= 24`
  - `trajectory_score >= 22`
  - `evidence_coverage_score >= 24`
- support needed band:
  - `support_needed_score >= 55`
  - `merit_score >= 28`
- priority band:
  - `merit_score >= 56`
  - `confidence_score >= 42`
  - `authenticity_risk <= 44`
  - `evidence_coverage_score >= 36`
- shortlist band:
  - `shortlist_priority_score >= 40`
  - `merit_score >= 28`
  - `evidence_coverage_score >= 24`
  - `authenticity_risk <= 64`
  - or candidate is already in priority band
  - or candidate is already in hidden-potential band

#### 10. Recommendation Mapping

Current recommendation flow:

```text
if confidence_raw < 0.48:
  add LOW_CONFIDENCE flag

if eligibility_status == invalid:
  recommendation = invalid

elif eligibility_status == incomplete_application:
  recommendation = incomplete_application

elif policy.insufficient_evidence_band:
  add INSUFFICIENT_EVIDENCE flag
  recommendation = insufficient_evidence

elif confidence_raw < 0.28 and evidence_count < 0.35:
  add INSUFFICIENT_EVIDENCE flag
  recommendation = insufficient_evidence

elif policy.priority_band:
  recommendation = review_priority

elif merit_raw >= 0.52 and confidence_raw >= 0.48 and authenticity_risk_raw < 0.45:
  recommendation = review_priority

elif policy.authenticity_review_band:
  recommendation = manual_review_required

elif merit_raw >= 0.42:
  if confidence_raw < 0.48 or authenticity_risk_raw >= 0.58:
    recommendation = manual_review_required
  else:
    recommendation = standard_review

elif authenticity_risk_raw >= 0.72:
  recommendation = manual_review_required

else:
  recommendation = standard_review
```

Interpretation:

- recommendation is a workflow routing label
- it is not an admission decision

#### 11. Final Response Assembly

After scoring, the current baseline additionally builds:

- `merit_breakdown`
- `semantic_rubric_scores`
- `committee_cohorts`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`
- `supported_claims`
- `weakly_supported_claims`
- `evidence_highlights`
- `top_strengths`
- `main_gaps`
- `uncertainties`
- `explanation`

Current explanation path:

- deterministic explanation is always available
- optional LLM enrichment can add stronger claims, gaps, evidence spans, rubric assessment, and follow-up questions
- if LLM fails, the deterministic path remains the source of truth

Current early-exit behavior:

- `invalid` and `incomplete_application` return zero score fields
- these responses still include explanation, review flags, and committee-facing guidance for the gate condition

### Current Offline Ranker Snapshot

- ranker_artifact_version: `offline-shortlist-ranker-v1`
- ranker_feature_count: `11`

Interpretation:

- shortlist ordering already uses an offline-trained artifact
- but the ranker is still simple and depends on current aggregate scores rather than stronger supervised ranking signals

### Current Data Snapshot

From `data/candidates.json`:

- candidate_count: `52`
- english: `18`
- russian: `17`
- mixed: `17`

Interpretation:

- current sample pack is useful for development, smoke tests, and qualitative checks
- it is not enough to serve as the long-term source of truth for supervised ranking

### Current Active Data Scope

The current active seed set for roadmap work is English-only.

Working dataset:

- source: `data/candidates.json`
- processed subset: `data/ml_workbench/processed/english_candidates_api_input_v1.jsonl`
- batch payload: `data/ml_workbench/processed/english_candidates_api_input_v1_batch.json`
- manifest: `data/ml_workbench/processed/english_candidates_api_input_v1_manifest.json`
- selection rule: `content_profile.language_profile == "english"`

Sanitization rule:

- keep only the frozen public input contract fields accepted by the ML service
- remove `content_profile`
- remove `scenario_meta`
- remove extra nested fields such as `motivation_questions[].id`

Current counts:

- total source candidates: `52`
- active English subset: `18`
- excluded mixed: `17`
- excluded russian: `17`

Interpretation:

- English-only is the correct v1 training / annotation scope
- `mixed` should be preserved as a future evaluation slice
- `russian` should stay out of the first ML loop until the English baseline is stable

### Current Verification Snapshot

Full test suite at capture time:

- command: `python -m pytest -q`
- result: `42 passed in 32.54s`

Interpretation:

- current local baseline is functionally stable enough to use as a starting point
- this does not yet mean it is well-calibrated against real committee labels

### Reference Candidate Snapshot

Local deterministic scoring reference for `cand_001` from `data/candidates.json`:

- `eligibility_status = eligible`
- `merit_score = 64`
- `confidence_score = 60`
- `authenticity_risk = 44`
- `recommendation = review_priority`
- `hidden_potential_score = 57`
- `support_needed_score = 54`
- `shortlist_priority_score = 68`
- `trajectory_score = 74`
- `evidence_coverage_score = 79`

Committee cohorts:

- `High priority`
- `Hidden potential`
- `Trajectory-led candidate`
- `Community-oriented builder`
- `Authenticity review needed`

Interpretation:

- this is a good example of the current system behavior
- the model can surface growth and hidden-potential signals
- authenticity review routing still looks somewhat noisy on sincere but simple profiles

### Current Strengths

- strong deterministic guardrails
- clear committee-facing outputs
- explicit separation of merit, confidence, authenticity risk, and shortlist logic
- good product direction around hidden potential and trajectory
- stable enough local test baseline

### Current Weaknesses

- semantic layer is still heuristic-heavy and relatively shallow
- shortlist ranker is not yet driven by a robust human-labeled ranking dataset
- authenticity risk likely needs better calibration
- confidence still reflects heuristic support more than validated reliability
- current development dataset is too small and too synthetic for strong supervised learning

### Current Bottleneck

The main bottleneck is not feature invention.
The main bottleneck is high-quality labeled data.

Without adjudicated human labels, pairwise comparisons, and batch shortlist tasks:

- stronger models will mostly optimize against proxy assumptions
- ranking gains will be hard to trust
- calibration work will remain noisy

## Current Labeling And Research Progress Snapshot

Date:

- `2026-04-02`

### Current Reviewed Label Pool

Current individual-review file:

- `data/ml_workbench/labels/human_labels_individual_llm_v2.csv`

Current counts:

- total labeled rows: `107`
- seed English candidates: `18`
- synthetic batch v1: `48`
- contrastive batch v2: `24`
- translated batch v3: `17`

Interpretation:

- the project now has a materially larger annotation pool than the original seed set
- this pool is useful for offline prototyping, rubric stress-tests, and first ranking experiments
- it is still not equivalent to multi-review committee ground truth

### Current Bootstrap Supervision Artifacts

Bootstrap artifact files:

- `data/ml_workbench/labels/human_labels_adjudicated.csv`
- `data/ml_workbench/labels/pairwise_labels.csv`
- `data/ml_workbench/labels/batch_shortlist_tasks.jsonl`
- `data/ml_workbench/labels/bootstrap_label_artifacts_summary.md`

Current counts:

- adjudicated rows written: `107`
- pairwise rows written: `504`
- batch shortlist tasks written: `18`
- batch size: `8`

Important caveat:

- these are bootstrap weak-label artifacts derived from one reviewer stream
- they are useful for offline ranking setup and evaluation plumbing
- they are not a substitute for true multi-review adjudication or real committee pairwise judgments

### Current Synthetic Expansion Snapshot

Completed synthetic packs:

- `synthetic_batch_v1`: generated, sanitized, and annotated
- `contrastive_batch_v2`: generated, sanitized, and annotated

Current translation work:

- `research/scripts/generate_translated_batch_v3.py` is ready
- Batch V3 has been generated from `17` Russian-only candidates in `data/candidates.json`
- Batch V3 has now also been annotated and merged into the main individual-review pool
- translated outputs should be treated as reviewer input packs, not as automatic labels

Current translation batch outputs:

- `data/ml_workbench/raw/generated/translated_batch_v3/translated_batch_v3_api_input.jsonl`
- `data/ml_workbench/raw/generated/translated_batch_v3/translated_batch_v3_generation_manifest.jsonl`
- `data/ml_workbench/raw/generated/translated_batch_v3/translated_batch_v3_summary.json`
- `data/ml_workbench/processed/annotation_packs/translated_batch_v3/translated_batch_v3_annotation_pack.json`

Current Batch V3 verification:

- translated candidate count: `17`
- raw and sanitized packs match one-to-one
- reviewer pack contains no `metadata`
- `behavioral_signals` now contains only:
  - `completion_rate`
  - `returned_to_edit`
  - `skipped_optional_questions`
- source-id leakage check: clean
- English-only heuristic: clean
- translation runtime: `cuda`

Interpretation:

- v1 covered broad archetype and ambiguity diversity
- v2 added stronger contrastive supervision around evidence, polish, initiative, and manual-review ambiguity
- v3 is the right next expansion because it increases domain realism without inventing entirely new underlying profiles

### Current Best Next Step

The current offline training export is now ready.

Current export files:

- `data/ml_workbench/exports/training_dataset_v1.jsonl`
- `data/ml_workbench/exports/training_dataset_v1.csv`
- `data/ml_workbench/exports/training_dataset_v1_manifest.json`

Current split snapshot:

- row count: `107`
- train: `76`
- validation: `17`
- test: `14`

The next research step now should be:

1. train the first shortlist-ranker baseline on interpretable aggregate features only
2. evaluate it against the current static offline ranker artifact
3. use the frozen split for all before/after comparisons
4. only then decide whether another synthetic expansion is still needed

Interpretation:

- more random synthetic generation is no longer the highest-value move
- the project is now at the stage where better supervision structure matters more than raw candidate count

## Iteration 1: Shortlist Ranker V1 Baseline

- roadmap_phase: `Phase 1 -> Phase 3 bridge`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- added `research/eval_protocol.md`
- added `research/scripts/train_shortlist_ranker_v1.py`
- trained the first `LightGBM LGBMRanker` on deterministic aggregate runtime features only
- saved offline outputs under `data/ml_workbench/exports/models/shortlist_ranker_v1/`

What stayed deterministic:

- frozen public input contract
- deterministic scoring pipeline
- deterministic aggregate feature extraction
- no runtime ranking behavior was changed

Data used:

- `training_dataset_v1.csv`
- projected split-local groups from `batch_shortlist_tasks.jsonl`
- split-local pairs from `pairwise_labels.csv`

Metrics before:

- validation NDCG@3: `0.8212`
- validation NDCG@5: `0.8570`
- validation pairwise accuracy: `0.4839`
- validation hidden-potential recall@3: `0.8611`

Metrics after:

- validation NDCG@3: `0.8905`
- validation NDCG@5: `0.9140`
- validation pairwise accuracy: `0.6774`
- validation hidden-potential recall@3: `0.9167`

Qualitative impact:

- the first learned ranker already beats the static linear artifact on the current bootstrap validation view
- this supports the roadmap decision to learn shortlist ordering before moving to embeddings or rerankers

Risks / open questions:

- supervision is still bootstrap-derived rather than true committee adjudication
- validation groups are small
- test-side ranking evidence remains weak and mostly pairwise
- `best_iteration = 12` still suggests the current feature set is strong relative to dataset size

Rollback path:

- keep using `app/assets/offline_shortlist_ranker_v1.json`
- treat `shortlist_ranker_v1` as offline research only until a deployment artifact and stronger supervision are ready

## Iteration 2: Label Pool Expansion Through V6

- roadmap_phase: `Phase 1 supervision expansion`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- cleaned remaining boilerplate and placeholder residue in `messy_batch_v5` and `messy_batch_v5_extension`
- appended bootstrap annotations for all new reviewer packs:
  - `messy_batch_v4`
  - `messy_batch_v5`
  - `messy_batch_v5_extension`
  - `ordinary_batch_v6`
- rebuilt `human_labels_adjudicated.csv`
- rebuilt `pairwise_labels.csv`
- rebuilt `batch_shortlist_tasks.jsonl`
- built refreshed canonical exports:
  - `data/ml_workbench/exports/training_dataset_v2.jsonl`
  - `data/ml_workbench/exports/training_dataset_v2.csv`
  - `data/ml_workbench/exports/training_dataset_v2_manifest.json`

What stayed deterministic:

- frozen public input contract
- deterministic normalization and eligibility
- deterministic runtime scoring pipeline
- no runtime scoring or ranking behavior shipped to the API

Data used:

- existing `human_labels_individual_llm_v2.csv` base pool
- sanitized reviewer packs for `messy_batch_v4`, `messy_batch_v5`, `messy_batch_v5_extension`, `ordinary_batch_v6`
- bootstrap adjudication and ranking-task regeneration on the expanded pool

Pool after expansion:

- total labeled rows: `251`
- source counts:
  - `seed_pack`: `18`
  - `synthetic_batch_v1`: `48`
  - `contrastive_batch_v2`: `24`
  - `translated_batch_v3`: `17`
  - `messy_batch_v4`: `40`
  - `messy_batch_v5`: `60`
  - `messy_batch_v5_extension`: `20`
  - `ordinary_batch_v6`: `24`

Latest label distribution:

- `review_priority`: `68`
- `standard_review`: `164`
- `manual_review_required`: `6`
- `insufficient_evidence`: `13`
- `shortlist_band=true`: `75`
- `hidden_potential_band=true`: `107`
- `support_needed_band=true`: `120`
- `authenticity_review_band=true`: `6`

Refreshed bootstrap artifacts:

- adjudicated rows written: `251`
- batch shortlist tasks written: `36`
- pairwise rows written: `1008`

Current canonical dataset:

- `data/ml_workbench/exports/training_dataset_v2.jsonl`
- `data/ml_workbench/exports/training_dataset_v2.csv`
- `data/ml_workbench/exports/training_dataset_v2_manifest.json`

Current split snapshot:

- row count: `251`
- train: `177`
- validation: `39`
- test: `35`

Qualitative impact:

- the pool now contains a much larger mix of ordinary, messy, low-polish, and disagreement-prone applications
- `messy_batch_v5` and `messy_batch_v5_extension` are now clean of the earlier visible placeholders and admissions-boilerplate
- `training_dataset_v2` is large enough to support a more credible second round of offline ranking experiments

Risks / open questions:

- these labels are still bootstrap single-review annotations, not true committee adjudication
- the new batches increase realism coverage, but they also increase synthetic share of the pool
- pairwise and batch artifacts are still derived rather than committee-authored

Rollback path:

- keep `training_dataset_v1` and the earlier `107`-row snapshot for before/after comparison
- treat `training_dataset_v2` as the new offline working set, not as production ground truth

## Iteration 3: Targeted Gap-Fill Batch And Final Working Set

- roadmap_phase: `Phase 1 supervision expansion`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- added `gap_fill_batch_v7` as a targeted synthetic batch for underrepresented slices
- cleaned its public payloads into the strict frozen contract shape by removing extra nullable `behavioral_signals` keys
- appended bootstrap annotations for all `72` `syn_gap_v7_*` candidates
- rebuilt `human_labels_adjudicated.csv`
- rebuilt `pairwise_labels.csv`
- rebuilt `batch_shortlist_tasks.jsonl`
- built the current canonical exports:
  - `data/ml_workbench/exports/training_dataset_v3.jsonl`
  - `data/ml_workbench/exports/training_dataset_v3.csv`
  - `data/ml_workbench/exports/training_dataset_v3_manifest.json`

What stayed deterministic:

- frozen public input contract
- deterministic normalization and eligibility
- deterministic runtime scoring pipeline
- no runtime API behavior changed

Data used:

- the prior `251`-row reviewed pool
- raw + sanitized payloads from `gap_fill_batch_v7`
- targeted weak-label routing for gap slices:
  - authenticity / manual-review
  - insufficient-evidence-but-valid
  - no-interview across quality levels
  - translated-thinking English
  - support-needed but not hidden-star

Pool after expansion:

- total labeled rows: `323`
- source counts:
  - `seed_pack`: `18`
  - `synthetic_batch_v1`: `48`
  - `contrastive_batch_v2`: `24`
  - `translated_batch_v3`: `17`
  - `messy_batch_v4`: `40`
  - `messy_batch_v5`: `60`
  - `messy_batch_v5_extension`: `20`
  - `ordinary_batch_v6`: `24`
  - `gap_fill_batch_v7`: `72`

Latest label distribution:

- `review_priority`: `71`
- `standard_review`: `196`
- `manual_review_required`: `24`
- `insufficient_evidence`: `32`
- `shortlist_band=true`: `78`
- `hidden_potential_band=true`: `114`
- `support_needed_band=true`: `143`
- `authenticity_review_band=true`: `29`

Gap-fill batch `v7` contribution:

- `manual_review_required`: `16`
- `insufficient_evidence`: `16`
- `authenticity_review_band=true`: `18`
- `has_interview_text=false`: `34`

Refreshed bootstrap artifacts:

- adjudicated rows written: `323`
- batch shortlist tasks written: `48`
- pairwise rows written: `1344`

Current canonical dataset:

- `data/ml_workbench/exports/training_dataset_v3.jsonl`
- `data/ml_workbench/exports/training_dataset_v3.csv`
- `data/ml_workbench/exports/training_dataset_v3_manifest.json`

Current split snapshot:

- row count: `323`
- train: `227`
- validation: `50`
- test: `46`

Qualitative impact:

- the working set now has much better coverage of review-risk and thin-but-valid applications
- the pool is less interview-dependent than before
- the dataset now includes more modest support-needed candidates who are not framed as hidden stars
- the current label mix is still synthetic-heavy, but it is much less biased toward strong review-priority narratives than earlier versions

Risks / open questions:

- `gap_fill_batch_v7` is still synthetic and weakly supervised, even though it targets real supervision gaps
- manual-review and insufficient-evidence labels are stronger now, but still not committee-authored
- before any runtime deployment changes, the shortlist ranker should be retrained and stress-tested on `training_dataset_v3`

Rollback path:

- keep `training_dataset_v2` as the previous canonical export
- treat `training_dataset_v3` as the current offline working set, not as production ground truth

## Iteration 4: Ranker Refit On Training Dataset V3

- roadmap_phase: `Phase 1 -> Phase 3 bridge`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- updated `research/scripts/train_shortlist_ranker_v1.py` to consume `training_dataset_v3`
- trained a new offline ranker run and saved outputs under:
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/`

What stayed deterministic:

- frozen public input contract
- deterministic runtime feature extraction
- no embedding features added
- no runtime API behavior changed

Data used:

- `training_dataset_v3.csv`
- `pairwise_labels.csv`
- `batch_shortlist_tasks.jsonl`

Metrics before:

- validation NDCG@3: `0.9817`
- validation NDCG@5: `0.9817`
- validation pairwise accuracy: `0.6667`
- validation hidden-potential recall@3: `1.0000`

Metrics after:

- validation NDCG@3: `0.9908`
- validation NDCG@5: `0.9908`
- validation pairwise accuracy: `0.8095`
- validation hidden-potential recall@3: `1.0000`
- test NDCG@3: `0.9700`
- test pairwise accuracy: `0.7813`

Group counts:

- train groups with size `>= 3`: `48`
- validation groups with size `>= 3`: `3`
- test groups with size `>= 2`: `15`
- test groups with size `>= 3`: `7`

Qualitative impact:

- the learned ranker still improves over the static artifact on the enlarged pool
- the gain is now modest on NDCG because validation ranking groups are few and already near ceiling
- pairwise accuracy improved more clearly than NDCG on the new split

Risks / open questions:

- validation is still too small to support strong claims from NDCG alone
- hidden-potential recall is saturated on the current validation groups
- the next useful move is slice/error analysis, not immediate architecture churn

Rollback path:

- keep using the static runtime artifact
- keep the earlier `shortlist_ranker_v1` research outputs for comparison
- treat `shortlist_ranker_v1_training_dataset_v3` as the current offline reference run

## Iteration 5: Slice Eval On Training Dataset V3

- roadmap_phase: `Phase 3 validation`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- added `research/scripts/slice_eval_v1.py`
- generated:
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/slice_eval_report.md`
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/slice_eval_summary.json`

What stayed deterministic:

- frozen public input contract
- aggregate deterministic runtime features
- no embedding features
- no runtime API changes

Data used:

- `training_dataset_v3.csv`
- `pairwise_labels.csv`
- `batch_shortlist_tasks.jsonl`
- `candidate_predictions.csv`
- `metrics_summary.json`

Main findings:

- validation pairwise accuracy still improves from `0.6667` to `0.8095`
- test pairwise accuracy regresses from `0.8125` to `0.7813`
- validation NDCG remains near ceiling because validation only has `3` projected groups with size `>= 3`
- test-side weakness is concentrated rather than uniform

Slice findings:

- learned ranker improves validation on:
  - `messy_batch_v5`
  - `ordinary_batch_v6`
  - hidden-potential positives
  - support-needed positives
- learned ranker regresses on test pairwise for:
  - `messy_batch_v5`
  - `seed_pack`
- learned ranker improves test pairwise for:
  - `messy_batch_v4`
  - `messy_batch_v5_extension`

Interpretation:

- the project is not yet at the “ship the learned ranker” stage
- the model is learning useful signals, but current supervision still creates slice instability
- the next useful move is not immediate embeddings
- the next useful move is targeted error analysis and feature inspection around:
  - `messy_batch_v5`
  - `seed_pack`
  - standard-review negatives
  - no-interview cases

Recommended next step:

1. inspect the misranked validation/test pairwise rows from `slice_eval_summary.json`
2. compare baseline vs learned behavior on `messy_batch_v5` and `seed_pack`
3. run feature ablations or feature audits before moving to Phase 2 embeddings

Rollback path:

- keep using the static runtime artifact as the trusted reference
- treat slice-eval outputs as decision support for the next offline experiment

## Iteration 6: Deterministic Feature Ablation And Ranker Stabilization

- roadmap_phase: `Phase 3 validation`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- added `research/scripts/ablate_shortlist_ranker_v1.py`
- made `research/scripts/train_shortlist_ranker_v1.py` deterministic:
  - disabled stochastic bagging / feature subsampling
  - enabled deterministic LightGBM settings
- selected a stabilized aggregate-feature variant:
  - `feature_variant_name = drop_support`
  - excluded `support_needed_score` from the learned ranker feature set
- regenerated:
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/metrics_summary.json`
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/feature_importance.csv`
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/candidate_predictions.csv`
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/slice_eval_report.md`
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/slice_eval_summary.json`
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/ablation_summary.json`
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/ablation_report.md`

What stayed deterministic:

- frozen public input contract
- aggregate deterministic runtime features only
- no embedding features
- no runtime API changes
- offline ranker remains the fallback reference

Data used:

- `training_dataset_v3.csv`
- `pairwise_labels.csv`
- `batch_shortlist_tasks.jsonl`
- current payload pool up to `gap_fill_batch_v7`

Metrics before:

- baseline validation pairwise accuracy: `0.7143`
- baseline test pairwise accuracy: `0.8125`
- baseline test NDCG@3: `0.9739`

Metrics after:

- learned validation pairwise accuracy: `0.8095`
- learned test pairwise accuracy: `0.8438`
- learned test NDCG@3: `0.9862`
- validation shortlist recall@3: unchanged at `1.0000`

Qualitative impact:

- overall ranking quality improved without changing the frozen contract
- previous test-side regressions on `messy_batch_v5` and `seed_pack` were removed
- the current learned ranker is now better than the static artifact on overall validation and overall test pairwise accuracy
- the remaining visible weakness is concentrated in `messy_batch_v5_extension`

Risks / open questions:

- `messy_batch_v5_extension` still regresses on test pairwise from `0.8333` to `0.6667` over `6` rows
- validation still has only `3` projected groups with size `>= 3`, so NDCG is near ceiling
- manual-review positives remain small and noisy

Rollback path:

- keep `app/assets/offline_shortlist_ranker_v1.json` as the trusted static fallback
- if the learned ranker is not promoted later, treat this run as the current best offline research artifact
- use `ablation_summary.json` as the reference for future aggregate-feature variants

## Iteration 7: Targeted Error Analysis On Remaining Weak Slices

- roadmap_phase: `Phase 3 validation`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- added `research/scripts/error_analysis_v1.py`
- generated:
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/error_analysis_report.md`
  - `data/ml_workbench/exports/models/shortlist_ranker_v1_training_dataset_v3/error_analysis_summary.json`
- ran a temporary monotone LightGBM probe against the current shortlist feature set to stress-test the `messy_batch_v5_extension` slice

What stayed deterministic:

- frozen public input contract
- aggregate-feature shortlist model only
- no runtime API changes
- no embedding features

Data used:

- `training_dataset_v3.csv`
- `pairwise_labels.csv`
- `candidate_predictions.csv`
- current payload pool up to `gap_fill_batch_v7`

Main findings:

- `messy_batch_v5_extension` test regression is narrow:
  - `2` learned-wrong pairs out of `6`
  - both are trajectory-driven preferences
- manual-review routing is the more important weakness:
  - `4` validation/test manual-review pairs
  - baseline gets all `4`
  - learned shortlist ranker gets only `2`
- both learned-wrong manual-review pairs prefer candidates with:
  - `authenticity_review = true`
  - `support_needed = true`

Interpretation:

- the shortlist ranker is now mostly doing shortlist ordering better than the static artifact
- the remaining weakness is not generic ranking quality
- the remaining weakness is that manual-review / review-risk positives should not be forced into the shortlist ranker objective
- this strengthens the case for a separate Phase 4 manual-review / confidence sidecar

Monotone probe result:

- monotone constraints recovered `messy_batch_v5_extension` to baseline accuracy
- but did not improve the overall shortlist model enough to justify promotion
- therefore the monotone probe was not adopted as the main shortlist ranker

Recommended next step:

1. keep the current `drop_support` shortlist ranker as the best overall offline shortlist model
2. stop trying to make the shortlist ranker solve manual-review routing
3. train a dedicated Phase 4 probe for `manual_review_required`
4. collect a tiny targeted pairwise / review-risk pack around:
   - `messy_batch_v5_extension`
   - manual-review positives

Rollback path:

- keep using the current deterministic shortlist artifact as the best aggregate-feature offline run
- treat `error_analysis_summary.json` as guidance for the next phase rather than as a reason to rewrite the shortlist objective

## Iteration 8: Manual Review Probe V1

- roadmap_phase: `Phase 4 groundwork`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- added `research/scripts/train_manual_review_probe_v1.py`
- generated:
  - `data/ml_workbench/exports/models/manual_review_probe_v1_training_dataset_v3/metrics_summary.json`
  - `data/ml_workbench/exports/models/manual_review_probe_v1_training_dataset_v3/coefficients.csv`
  - `data/ml_workbench/exports/models/manual_review_probe_v1_training_dataset_v3/candidate_predictions.csv`
  - `data/ml_workbench/exports/models/manual_review_probe_v1_training_dataset_v3/probe_report.md`

What stayed deterministic:

- frozen public input contract
- aggregate deterministic features only
- no runtime API changes
- no shortlist model change

Data used:

- `training_dataset_v3.csv`
- current payload pool up to `gap_fill_batch_v7`

Target:

- `final_recommendation == manual_review_required`

Main findings:

- the target remains too small and too noisy for a stable routing model:
  - train positives: `16`
  - validation positives: `6`
  - test positives: `2`
- simple `authenticity_risk` baseline remains weak, especially on test
- logistic probe improves train discrimination but is unstable on validation:
  - validation AP drops below the simple baseline
  - test AUC improves, but positive support is too small for a reliable go/no-go decision

Interpretation:

- the right next move is more review-risk supervision, not more classifier complexity
- Phase 4 is now justified, but the current dataset still under-supports a reliable manual-review model

Recommended next step:

1. add targeted manual-review positives and near-miss negatives
2. collect more no-interview review-risk cases
3. revisit the manual-review probe after the targeted label pack is added

Rollback path:

- keep using the existing deterministic review-risk logic
- treat `manual_review_probe_v1_training_dataset_v3` as a diagnostic artifact only

## Iteration 9: Split-Aware Targeted Shortlist Supervision And Blended Ranker

- roadmap_phase: `Phase 3 validation`
- date: `2026-04-02`

What changed:

- updated `research/scripts/build_bootstrap_label_artifacts.py` to:
  - include `gap_fill_batch_v7` in generic source rotation
  - add `6` split-aware targeted shortlist tasks around:
    - `manual_review_required` positives
    - `messy_batch_v5_extension` trajectory disagreements
    - no-interview and authenticity-review edge cases
- refreshed bootstrap artifacts:
  - `batch_shortlist_tasks.jsonl` -> `54` tasks
  - `pairwise_labels.csv` -> `1512` rows
- updated `research/scripts/train_shortlist_ranker_v1.py`:
  - kept `feature_variant_name = drop_support`
  - selected `learned_blend_alpha = 0.4`
- regenerated:
  - `metrics_summary.json`
  - `candidate_predictions.csv`
  - `slice_eval_report.md`
  - `slice_eval_summary.json`
  - `error_analysis_report.md`
  - `error_analysis_summary.json`

What stayed deterministic:

- frozen public input contract
- deterministic runtime feature extraction
- no embedding features
- no runtime API behavior changes

Data used:

- `training_dataset_v3.csv`
- refreshed `pairwise_labels.csv`
- refreshed `batch_shortlist_tasks.jsonl`
- current payload pool up to `gap_fill_batch_v7`

Current selected shortlist variant:

- aggregate feature set: `drop_support`
- blend alpha: `0.4`

Metrics before:

- baseline validation pairwise accuracy: `0.6512`
- baseline validation NDCG@3: `0.8725`
- baseline test pairwise accuracy: `0.7556`
- baseline test NDCG@3: `0.9299`
- baseline test shortlist recall@3: `0.8000`

Metrics after:

- learned validation pairwise accuracy: `0.7209`
- learned validation NDCG@3: `0.8946`
- learned test pairwise accuracy: `0.7556`
- learned test NDCG@3: `0.9384`
- learned test shortlist recall@3: `1.0000`

Slice-level impact:

- `manual_review_required` validation/test pairwise accuracy improved from `0.6429` to `0.7143`
- `messy_batch_v5_extension` test pairwise accuracy improved from `0.4000` to `0.6000`
- `gap_fill_batch_v7` validation pairwise held flat at `0.7222`
- remaining regression is concentrated in `messy_batch_v5` test pairs (`0.8333` -> `0.6667`)

Qualitative impact:

- targeted shortlist supervision now hits the right weak slices instead of only broad synthetic coverage
- the selected blend is less aggressive than pure raw-model ranking and is better aligned with review-risk edge cases
- shortlist recall on test projected groups improved without touching the deterministic runtime pipeline

Risks / open questions:

- pairwise gains are still bootstrap-derived, not committee-authored
- `manual_review_required` cases are better represented, but still not supported by true multi-review adjudication
- `messy_batch_v5` remains the main stubborn shortlist slice
- `error_analysis_summary.json` now points to narrower problems than before, but not to a deploy-ready artifact

Rollback path:

- keep the static shortlist artifact as the trusted production fallback
- keep `drop_support` with `learned_blend_alpha = 0.4` as the current offline-best research variant
- if needed, revert to the plain baseline ordering by setting blend alpha back to `0.0` in research only

## Iteration 10: Time-Boxed Ground Truth Freeze

- roadmap_phase: `Phase 1 freeze for execution`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- accepted the current bootstrap supervision stack as the operational ground truth for the current project timebox
- froze the current working set around:
  - `training_dataset_v3`
  - `human_labels_adjudicated.csv`
  - `pairwise_labels.csv`
  - `batch_shortlist_tasks.jsonl`
- explicitly stopped waiting on additional human adjudication for this iteration

What stayed deterministic:

- frozen public input contract
- deterministic runtime scoring pipeline
- deterministic shortlist fallback path
- no runtime API behavior changed

Data accepted as operational GT:

- candidate pool: `323` rows in `training_dataset_v3`
- bootstrap adjudicated labels: `323`
- bootstrap pairwise rows: `1512`
- bootstrap shortlist tasks: `54`

Decision:

- for the current delivery window, these artifacts are the working ground truth for offline training, evaluation, and research iteration
- they are not equivalent to true committee-authored gold labels
- they are accepted anyway because time no longer allows a real human re-annotation pass

Qualitative impact:

- the project can now move forward without keeping Phase 1 artificially open
- ranker and calibration work should now optimize against the frozen dataset instead of waiting on better labels
- future human adjudication, if it ever happens, should be treated as a later correction layer, not a blocker for current work

Risks / open questions:

- current GT remains bootstrap-derived and synthetic-heavy
- manual-review and authenticity slices are still weaker than they would be under true committee adjudication
- any runtime promotion should still be described as shipping against a time-boxed operational GT, not against final institutional truth

Rollback path:

- keep `training_dataset_v2` and the earlier `training_dataset_v3` artifacts for before/after comparison
- if needed, reopen Phase 1 later and replace the operational GT with true human-reviewed labels

## Iteration 11: Review-Routing Probe V2 On Frozen GT

- roadmap_phase: `Phase 4 groundwork`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- added `research/scripts/train_manual_review_probe_v2.py`
- compared multiple routing targets instead of forcing the model to learn only strict `manual_review_required`
- compared:
  - `manual_review_required`
  - `nonstandard_route = manual_review_required or insufficient_evidence`
  - `review_risk_or_insufficient`
- compared model families:
  - balanced logistic regression
  - balanced random forest
- generated:
  - `data/ml_workbench/exports/models/manual_review_probe_v2_training_dataset_v3/metrics_summary.json`
  - `data/ml_workbench/exports/models/manual_review_probe_v2_training_dataset_v3/experiments.csv`
  - `data/ml_workbench/exports/models/manual_review_probe_v2_training_dataset_v3/candidate_predictions.csv`
  - `data/ml_workbench/exports/models/manual_review_probe_v2_training_dataset_v3/selected_model_features.csv`
  - `data/ml_workbench/exports/models/manual_review_probe_v2_training_dataset_v3/probe_report.md`

What stayed deterministic:

- frozen public input contract
- deterministic runtime scorer remains unchanged
- no shortlist-ranker runtime behavior changed
- no embedding features were added

Data used:

- `training_dataset_v3.csv`
- frozen operational GT around `training_dataset_v3`, `pairwise_labels.csv`, and `batch_shortlist_tasks.jsonl`

Selected probe:

- target: `nonstandard_route`
- meaning: `manual_review_required` or `insufficient_evidence`
- model: `random_forest_balanced`
- threshold (validation best F1): `0.45`

Selected metrics:

- validation AP: `0.6242`
- validation ROC AUC: `0.7225`
- test AP: `0.5287`
- test ROC AUC: `0.7985`
- tuned validation F1: `0.5714`
- tuned test F1: `0.5556`

Baseline comparison:

- simple authenticity-risk-only routing baseline remains much weaker:
  - validation AP: `0.3114`
  - test AP: `0.2049`

Interpretation:

- strict `manual_review_required` is still too sparse and unstable to be the only learnable routing target
- a broader non-standard routing sidecar is materially more learnable on the frozen GT
- this supports a layered system design:
  - deterministic scorer stays the backbone
  - shortlist ranker handles ordering
  - review-routing sidecar handles non-standard routing signals

Risks / open questions:

- the chosen target is broader than strict manual review and must be described honestly as routing support, not as a final authenticity decision
- random forest is currently an offline probe, not a promoted runtime artifact
- current GT is still bootstrap-derived and synthetic-heavy

Rollback path:

- keep using the existing deterministic recommendation routing
- treat probe v2 as the current best offline routing-sidecar candidate, not as an automatic deployment decision

## Iteration 12: Runtime Shadow Sidecar And Rank API Expansion

- roadmap_phase: `Phase 3/4 runtime preparation`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- added optional runtime review-routing sidecar service:
  - `app/services/review_routing_sidecar.py`
- integrated that sidecar into `ScoringPipeline` in shadow mode only
- the sidecar now runs alongside the deterministic scorer when enabled, but does not override `recommendation`
- exported a runtime artifact pair:
  - `app/assets/review_routing_sidecar_v1.joblib`
  - `app/assets/review_routing_sidecar_v1.json`
- expanded `/rank` so it can:
  - return full ranked batches
  - optionally truncate the returned list with `top_k`
  - return compact `ranked_candidates` summaries instead of only IDs
- updated public docs in `README.md`

What stayed deterministic:

- frozen public input contract
- deterministic recommendation mapping remains authoritative
- deterministic shortlist fallback remains intact
- shadow sidecar does not change final routing labels

Runtime behavior:

- `/score` still returns the same public contract
- `review_routing_shadow` is computed internally and kept out of the public JSON response
- `score_candidate_trace()` now includes shadow-sidecar diagnostics for internal debugging
- `/rank` still ranks the full batch first; `top_k` only truncates the returned ranked list

Qualitative impact:

- the system now has a safe path to compare deterministic routing vs ML routing hints without changing committee-visible behavior
- `/rank` is now more practical for UI and backend consumers that only need ordered candidates, not full scoring payloads
- shortlist logic and top-K retrieval are now clearly separated

Verification:

- `tests/test_rank_endpoint.py`
- `tests/test_score_batch.py`
- `tests/test_score_single.py`
- `tests/test_review_routing_sidecar.py`
- local result: `12 passed`

Risks / open questions:

- the sidecar runtime artifact depends on `scikit-learn`
- the shadow-sidecar artifact is still trained on time-boxed operational GT, not institutional gold labels
- no runtime override policy is enabled yet; this is observability-first integration

Rollback path:

- set `ENABLE_REVIEW_ROUTING_SIDECAR=false`
- ignore `review_routing_shadow` diagnostics
- continue using pure deterministic runtime routing and the existing `/rank` semantics without `top_k`

## Iteration 13: LLM Explainability Prompt V2

- roadmap_phase: `Phase 2 / committee explainability quality`
- date: `2026-04-02`
- owner: `Codex`

What changed:

- upgraded the runtime LLM explainability prompt to an evidence-first variant
- made the prompt explicitly inVision U and university-admissions aware
- added stronger constraints against generic praise and unsupported verdict language
- clarified that `deterministic_text_signals` are hints, not evidence
- tightened the `committee_follow_up_question` requirement so it targets one concrete uncertainty
- added prompt versioning:
  - `prompt_version = llm-explainability-v2-evidence-first`
- added prompt-layer tests:
  - `tests/test_llm_prompts.py`
  - extended `tests/test_llm_extractor.py`

What stayed deterministic:

- recommendation mapping remains deterministic
- LLM still does not assign routing labels or shortlist bands
- rubric fallback remains deterministic when the model output is weak or malformed
- no public score semantics changed

Data used:

- runtime candidate package only
- existing deterministic text signals as non-authoritative hints

Qualitative impact:

- committee-facing explanations are now more strongly biased toward evidence extraction instead of generic evaluation language
- prompt instructions now better reflect inVision U as a university context rather than a generic screening use case
- awkward English, translated-thinking phrasing, and modest self-presentation are less likely to be over-penalized by the explainability layer
- prompt metadata is now explicit in extractor output for debugging and experiment tracking

Verification:

- `tests/test_llm_extractor.py`
- `tests/test_llm_parser.py`
- `tests/test_llm_prompts.py`
- full local test run: `45 passed`

Risks / open questions:

- this improves explainability quality, not core ranking or routing metrics directly
- no offline committee-utility benchmark exists yet for compare-and-promote prompt variants
- the LLM path remains optional and provider-dependent

Rollback path:

- revert prompt templates in `app/services/llm_prompts.py`
- keep deterministic explanation path as the source of truth
- prompt versioning makes the change easy to isolate in future experiments

## Future Iteration Entry Template

Copy this block for every roadmap iteration.

### Iteration N: Title

- roadmap_phase:
- date:
- owner:

What changed:

- 

What stayed deterministic:

- 

Data used:

- 

Metrics before:

- 

Metrics after:

- 

Qualitative impact:

- 

Risks / open questions:

- 

Rollback path:

- 
