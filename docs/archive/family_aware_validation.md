# Family-Aware Validation

## Why This Exists

Counterfactual variants can create leakage-like inflation in synthetic evaluation.
This report adds family-aware views so near-neighbor candidate variants do not dominate validation.

## Dataset Summary
- candidate_count: 80
- annotated_candidate_count: 80
- family_count: 64
- largest_family_size: 3
- families_with_counterfactuals: 8
- family_size_distribution: {1: 56, 3: 8}

## Candidate-Level Metrics
- annotated_candidate_count: 80
- top_k: 16
- spearman_merit_vs_labels: 0.3798
- pairwise_accuracy: 0.671
- precision_at_k_priority: 0.75
- hidden_potential_recall_at_k: 0.1892
- support_flag_rate_in_top_k: 0.5

## Root-Representative Metrics
- annotated_candidate_count: 64
- top_k: 12
- spearman_merit_vs_labels: 0.3155
- pairwise_accuracy: 0.633
- precision_at_k_priority: 0.8333
- hidden_potential_recall_at_k: 0.2
- support_flag_rate_in_top_k: 0.5

## Family-Aggregated Metrics
- annotated_candidate_count: 64
- top_k: 12
- spearman_merit_vs_labels: 0.3913
- pairwise_accuracy: 0.6696
- precision_at_k_priority: 0.9167
- hidden_potential_recall_at_k: 0.24
- support_flag_rate_in_top_k: 0.5833

## Interpretation
- Candidate-level metrics are the most optimistic because counterfactual relatives appear separately.
- Root-representative metrics are stricter because each family contributes only one canonical candidate.
- Family-aggregated metrics are shortlist-oriented: a family counts once, using the best surfaced candidate in that family.

## Largest Families
- cand_003: size=3
- cand_016: size=3
- cand_023: size=3
- cand_033: size=3
- cand_034: size=3
- cand_037: size=3
- cand_040: size=3
- cand_045: size=3
- cand_001: size=1
- cand_002: size=1

## High-Priority Family Snapshot
- cand_003: family_size=3, max_committee_priority=4.0, top_scored_candidate_id=cand_053, top_scored_merit_score=51
- cand_006: family_size=1, max_committee_priority=4.0, top_scored_candidate_id=cand_006, top_scored_merit_score=18
- cand_008: family_size=1, max_committee_priority=4.0, top_scored_candidate_id=cand_008, top_scored_merit_score=36
- cand_010: family_size=1, max_committee_priority=5.0, top_scored_candidate_id=cand_010, top_scored_merit_score=45
- cand_011: family_size=1, max_committee_priority=4.0, top_scored_candidate_id=cand_011, top_scored_merit_score=13
- cand_012: family_size=1, max_committee_priority=4.0, top_scored_candidate_id=cand_012, top_scored_merit_score=49
- cand_013: family_size=1, max_committee_priority=5.0, top_scored_candidate_id=cand_013, top_scored_merit_score=53
- cand_014: family_size=1, max_committee_priority=4.0, top_scored_candidate_id=cand_014, top_scored_merit_score=30
- cand_016: family_size=3, max_committee_priority=4.0, top_scored_candidate_id=cand_016, top_scored_merit_score=57
- cand_017: family_size=1, max_committee_priority=5.0, top_scored_candidate_id=cand_017, top_scored_merit_score=52
- cand_018: family_size=1, max_committee_priority=5.0, top_scored_candidate_id=cand_018, top_scored_merit_score=29
- cand_019: family_size=1, max_committee_priority=4.0, top_scored_candidate_id=cand_019, top_scored_merit_score=57
- cand_020: family_size=1, max_committee_priority=4.0, top_scored_candidate_id=cand_020, top_scored_merit_score=40
- cand_021: family_size=1, max_committee_priority=4.0, top_scored_candidate_id=cand_021, top_scored_merit_score=51
- cand_023: family_size=3, max_committee_priority=4.0, top_scored_candidate_id=cand_058, top_scored_merit_score=46
