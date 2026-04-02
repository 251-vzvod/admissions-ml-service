# Offline Evaluation Protocol

## Purpose

This document freezes the first offline evaluation protocol before training stronger shortlist models.

It is intentionally conservative:

- use the frozen public input contract
- use deterministic runtime features only for the first learned ranker
- compare every learned model against the current static runtime artifact
- treat bootstrap supervision as the time-boxed operational GT for the current delivery window, while still acknowledging that it is not equal to real committee ground truth

## Current Data Inputs

Candidate-level canonical table:

- `data/ml_workbench/exports/training_dataset_v3.csv`
- `data/ml_workbench/exports/training_dataset_v3_manifest.json`

Bootstrap ranking supervision:

- `data/ml_workbench/labels/pairwise_labels.csv`
- `data/ml_workbench/labels/batch_shortlist_tasks.jsonl`

Current runtime baseline artifact:

- `app/assets/offline_shortlist_ranker_v1.json`

## Frozen Split

The current candidate-level split is frozen by `training_dataset_v3_manifest.json`.

Current counts:

- train: `227`
- validation: `50`
- test: `46`

Source coverage:

- `seed_pack`
- `synthetic_batch_v1`
- `contrastive_batch_v2`
- `translated_batch_v3`
- `messy_batch_v4`
- `messy_batch_v5`
- `messy_batch_v5_extension`
- `ordinary_batch_v6`
- `gap_fill_batch_v7`

Important:

- original shortlist batches are mixed across candidate splits
- therefore ranking evaluation uses split-local projected subbatches

## Feature Policy For Ranker V1

The first learned shortlist ranker must use only current deterministic aggregate runtime features.

For every candidate:

1. run the deterministic `ScoringPipeline` with LLM explainability disabled
2. build the feature map with `app.services.offline_ranker.build_offline_ranker_feature_map(...)`

This keeps the first experiment aligned with the roadmap:

- no raw-text end-to-end scorer
- no embedding features yet
- no schema changes
- no opaque runtime behavior

## Projected Ranking Groups

`batch_shortlist_tasks.jsonl` is projected onto the frozen candidate split.

For each original batch:

1. keep only candidates belonging to one split
2. preserve their order from `ranked_candidate_ids`
3. drop projected groups smaller than the required minimum size

Current practical consequence:

- train has useful projected groups of size `>= 3`
- validation has only a small number of projected groups of size `>= 3`
- test now has a small but usable set of projected groups of size `>= 3`
- test-side NDCG remains weaker than train/validation, but is no longer restricted to only size-2 groups

## Primary Metrics

These are the primary offline metrics for the first shortlist-ranker experiment.

### Validation NDCG@3

- computed on validation projected groups with size `>= 3`
- this is the main ranking-quality metric for early model comparison

### Validation NDCG@5

- computed on the same validation projected groups
- use `k = min(5, group_size)` per group

### Validation NDCG@10

- deferred for now
- current projected groups are still too small for NDCG@10 to be informative
- this metric should become active only after we have larger real shortlist batches

### Validation Pairwise Accuracy

- computed on same-split validation rows from `pairwise_labels.csv`
- measures whether the learned score preserves local preference ordering

### Validation Hidden-Potential Recall@3

- computed on validation projected groups that contain at least one `final_hidden_potential_band == true`
- among the model's top-3 ranked candidates in each group, measure recall of hidden-potential positives

## Secondary Metrics

These metrics are still useful, but not strong enough to be the only promotion gate.

### Train Metrics

- train NDCG@3 / NDCG@5
- train pairwise accuracy

Purpose:

- detect underfitting or obvious pipeline bugs
- not a decision metric

### Test Pairwise Accuracy

- computed on same-split test pairs
- currently very small-sample and therefore noisy

### Test NDCG@2

- exploratory secondary metric
- computed on projected test groups of size `>= 2`
- keep it as a stability check, not a promotion gate

### Test NDCG@3

- exploratory only
- now permissible because the test split has a small number of projected groups of size `>= 3`
- still too weak to be a deployment gate

### Calibration Error For Review-Risk Outputs

- not part of this first shortlist-ranker experiment
- remains a Phase 4 metric
- should be added only after authenticity / confidence calibration work starts

## Baseline Comparison Rule

Every learned model must be compared against the current static runtime artifact:

- `app/assets/offline_shortlist_ranker_v1.json`

The comparison must use the exact same:

- candidate pool
- frozen split
- projected ranking groups
- pairwise rows

## Promotion Rule For Ranker V1

A first learned shortlist ranker is only considered a real improvement if all of the following hold:

- validation NDCG@3 improves over the static artifact
- validation NDCG@5 does not regress materially
- validation pairwise accuracy does not regress materially
- validation hidden-potential recall@3 does not regress materially
- the feature set remains interpretable
- deterministic service behavior remains unchanged

Important:

- this is an offline research gate
- passing this gate does not automatically justify runtime deployment

## What This Protocol Does Not Claim

This protocol does not claim that:

- bootstrap labels equal real committee judgment
- current validation metrics are production-grade
- synthetic and translated slices are the same thing as real applicant traffic

It only gives us a disciplined before/after method for the next iteration.

## Immediate Next Step

Use this protocol for:

- `research/scripts/train_shortlist_ranker_v1.py`

Current next step:

- keep the current deterministic aggregate shortlist variant as:
  - `feature_variant_name = drop_support`
  - `learned_blend_alpha = 0.4`
- treat this as the current offline-best compromise, not a runtime promotion
- treat `training_dataset_v3` plus the refreshed bootstrap pairwise / batch artifacts as the frozen operational GT for the current sprint
- use `error_analysis_report.md` and `manual_review_probe_v1` to separate shortlist-ranking problems from manual-review routing problems
- continue ranker and calibration experiments against that frozen GT instead of waiting on more human annotation
- only move to Phase 2 embeddings after aggregate-feature improvements clearly plateau on the frozen GT

Do not move to embedding features or rerankers until this larger aggregate-feature baseline is trained and measured.
