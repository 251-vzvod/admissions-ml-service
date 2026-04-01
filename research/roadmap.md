# Admissions ML Roadmap

## Purpose

This file is the working source of truth for how the scoring system should evolve.
We will update this checklist as work lands.

Operational data collection and labeling details live in [data_collection_spec.md](./data_collection_spec.md).
Baseline snapshots and per-iteration change notes live in [baseline_log.md](./baseline_log.md).

The direction is explicit:

- do not replace the service with a black box
- keep the committee-facing system auditable and safe
- use ML where it materially improves semantic understanding, ranking quality, and calibration

## Non-Negotiables

- [x] Public input schema is frozen. ML work must not expand or rename request fields.
- [x] Payload normalization stays deterministic.
- [x] Eligibility gates stay deterministic.
- [x] Privacy projection stays deterministic.
- [x] Sensitive and socio-economic fields stay excluded from merit scoring.
- [x] LLM is optional and must not become the final decision maker.
- [x] The service must always have a deterministic fallback path.
- [ ] Keep all hard review-routing overrides documented in one place.

## What Must Stay Deterministic

These parts are not the place for learned models:

- request schema and backward-compatible normalization
- invalid / incomplete / missing-materials eligibility logic
- privacy and exclusion of protected or sensitive fields from merit scoring
- hard routing guardrails for obviously incomplete or invalid applications
- public score semantics and committee-facing API contract
- baseline claim-to-evidence traceability

Reason:

- these parts are business rules, safety constraints, and trust infrastructure
- if they become opaque, the system becomes harder to defend and harder to debug

## What ML Should Improve

These are the right places to add learned components:

- semantic understanding of essays, Q/A, and interview text
- robustness to weak writing quality, multilingual inputs, and low-polish phrasing
- shortlist ordering against actual committee preference
- hidden-potential detection
- calibration of confidence and authenticity-review signals
- evidence retrieval and cross-section consistency support

## Current Baseline

- [x] Deterministic feature extraction exists.
- [x] Deterministic scoring exists.
- [x] Semantic rubric layer exists, but is still lightweight and heuristic-heavy.
- [x] Offline shortlist ranker artifact exists, but is still simple and limited.
- [x] Committee-facing outputs already separate merit, confidence, authenticity risk, shortlist priority, and guidance.
- [x] Research utilities for offline human-vs-model comparison now exist.

## Main Strategic Decision

We are not building one end-to-end essay scorer first.

We are building the system in layers:

1. deterministic safety and committee logic stays in place
2. NLP improves semantic features
3. a learned ranker improves shortlist ordering
4. a calibrated review-risk model improves authenticity / confidence routing
5. a reranker is added only for top-K or uncertain cases

This is the intended direction because it preserves explainability while still allowing meaningful model gains.

## Phase 1: Labels And Offline Evaluation

This is the highest-priority bottleneck.
Without real human labels, additional ML will mostly optimize against synthetic assumptions.

- [ ] Freeze the offline evaluation protocol before training more models.
- [ ] Define one canonical human label pack for reviewed candidates.
- [ ] Collect committee labels for:
  - recommendation
  - shortlist priority band
  - hidden potential band
  - support needed band
  - authenticity review band
- [ ] Collect pairwise comparison labels inside realistic shortlist batches.
- [ ] Prefer batch-level and pairwise labels over only one scalar score.
- [ ] Add adjudication for disagreement between reviewers.
- [ ] Create a held-out evaluation split.
- [ ] Track slice-based evaluation by:
  - language profile
  - text length
  - evidence richness
  - transcript presence / absence
- [ ] Define the main offline metrics:
  - NDCG@5 and NDCG@10 for shortlist quality
  - pairwise accuracy
  - hidden-potential recall@K
  - calibration error for review-risk outputs

## Phase 2: NLP / Semantic Layer Upgrade

Goal:
upgrade semantic understanding without turning the entire runtime into a black box.

What changes:

- [ ] Keep current heuristic features.
- [ ] Add a stronger multilingual embedding layer as a feature source.
- [ ] Use embeddings for rubric alignment, not as the sole final score.
- [ ] Use embeddings for better evidence retrieval.
- [ ] Use embeddings for cross-section consistency support.
- [ ] Benchmark semantic gains offline before changing runtime defaults.

Candidate embedding models to evaluate first:

- `BAAI/bge-m3`
- `intfloat/multilingual-e5-base`
- `intfloat/multilingual-e5-large-instruct`

Why this layer matters:

- the current semantic layer is prototype-based and useful, but limited
- better embeddings should reduce keyword dependence
- better embeddings should help with English, Russian, mixed-language, and lower-polish applications

Decision rule:

- choose the smallest model that gives a real offline ranking gain
- do not choose a heavier model only because it looks more modern

## Phase 3: Learned Shortlist Ranker

Goal:
move shortlist ordering closer to committee judgment.

What stays the same:

- deterministic scoring still exists
- public scores still exist
- ranker consumes interpretable features rather than raw text only

What changes:

- [ ] Train the first supervised ranker on current aggregate features.
- [ ] Then train a second version on aggregate features plus embedding-derived features.
- [ ] Compare both against the current static offline ranker artifact.
- [ ] Keep runtime deployment artifact-based and offline-trained.
- [ ] Keep a deterministic fallback rank order.

Preferred first models:

- `LightGBM LGBMRanker` with `lambdarank`
- evaluate `rank_xendcg` as a faster alternative
- fallback option: `CatBoostRanker`

Why this is the first learned model to ship:

- ranking is closer to the real committee workflow than absolute scoring
- tree rankers work well on medium-sized labeled datasets
- they preserve feature-level interpretability better than end-to-end text models

## Phase 4: Authenticity Risk And Confidence Calibration

Goal:
make `authenticity_risk` and confidence more stable and better calibrated.

This is not a cheating detector.
This is a review-priority and uncertainty model.

- [ ] Train a separate model for manual-review risk.
- [ ] Train or recalibrate confidence as an assessment-reliability signal.
- [ ] Keep hard contradiction and invalidity rules deterministic.
- [ ] Calibrate probabilities with Platt scaling or isotonic regression.
- [ ] Evaluate false positives carefully on simple but sincere applications.

Target outcome:

- high authenticity risk should mean "needs manual verification"
- it should not mean "the system thinks the candidate is dishonest"

## Phase 5: Top-K Reranker

Goal:
add a more expensive comparison model only where it actually helps.

This is not a first-pass scorer.
It is a second-stage model for shortlist refinement.

- [ ] Do not run a reranker on the entire applicant pool by default.
- [ ] Apply reranking only to top-K or uncertain cases.
- [ ] Evaluate pairwise candidate-vs-candidate reranking.
- [ ] Evaluate query-to-candidate reranking for rubric dimensions if useful.
- [ ] Keep latency and cost budgets explicit before shipping.

When to do this:

- only after Phase 1 labels are good enough
- only after the learned shortlist ranker plateaus

Rationale:

- rerankers can improve fine ordering
- they are much more expensive and less transparent
- they should be used as a second-stage refinement, not as the system backbone

## Phase 6: Monitoring, Drift, And Governance

- [ ] Add offline regression tests for ranking quality.
- [ ] Add slice monitoring for language and evidence-density drift.
- [ ] Track changes in shortlist composition after every ranker update.
- [ ] Keep a frozen benchmark pack for before/after model comparison.
- [ ] Require a rollback path for every ML artifact shipped to runtime.
- [ ] Document model version, training data slice, and evaluation summary for every released artifact.

## What We Should Not Do Yet

- [ ] Do not train an end-to-end black-box essay scorer on weak labels.
- [ ] Do not replace deterministic eligibility or privacy logic with ML.
- [ ] Do not let an LLM write the final recommendation label directly in production.
- [ ] Do not use synthetic labels as the main source of truth for ranking.
- [ ] Do not add a heavy cross-encoder to the full request path before proving offline value.
- [ ] Do not optimize only average score correlation if shortlist behavior is still wrong.

## Working Order

This is the intended execution order:

- [ ] Phase 1: labels and offline evaluation
- [ ] Phase 2: stronger embedding features
- [ ] Phase 3: learned shortlist ranker
- [ ] Phase 4: authenticity / confidence calibration
- [ ] Phase 5: reranker for top-K only if justified
- [ ] Phase 6: monitoring and release discipline

## Definition Of Progress

We only count this roadmap as moving forward when a change satisfies all of the following:

- it improves offline evaluation on real labeled data
- it preserves or improves committee interpretability
- it does not require changing the frozen input contract
- it does not weaken deterministic safety rules
- it has a rollback path

## Reminder To Future Us

If we get distracted by "maybe we just need a better BERT", come back to this:

- first improve labels
- then improve semantic features
- then improve ranking
- only after that consider reranking

The priority is not model novelty.
The priority is better shortlist behavior under human-review constraints.
