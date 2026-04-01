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
