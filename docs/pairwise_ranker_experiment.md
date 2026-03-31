# Pairwise Ranker Experiment

## Scope

- train candidates: `65`
- test candidates: `15`
- train pairs: `2734`
- split: `family_aware_deterministic_80_20`

## Baseline Test Metrics

- spearman: `0.2529`
- pairwise accuracy: `0.6324`
- precision@k priority: `1.0`
- hidden potential recall@k: `0.3333`

## Learned Pairwise Test Metrics

- spearman: `0.5027`
- pairwise accuracy: `0.3529`
- precision@k priority: `1.0`
- hidden potential recall@k: `0.3333`

## Delta Learned Minus Baseline

- spearman delta: `0.2498`
- pairwise accuracy delta: `-0.2795`
- precision@k priority delta: `0.0`
- hidden potential recall@k delta: `0.0`

## Top Feature Weights

- `consistency_score_text`: `-11.3154`
- `confidence_score`: `9.5513`
- `evidence_count_text`: `-8.0617`
- `merit_score`: `6.6549`
- `growth_trajectory_semantic`: `6.1562`
- `polished_but_empty_neg_text`: `-4.7592`
- `shortlist_priority_score`: `3.904`
- `motivation_authenticity_semantic`: `-3.2192`
- `hidden_potential_semantic`: `2.5064`
- `authenticity_risk_neg`: `2.4211`
