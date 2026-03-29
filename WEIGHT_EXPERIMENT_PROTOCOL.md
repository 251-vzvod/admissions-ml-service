# Weight Experiment Protocol (weights-protocol-v1)

Goal: tune deterministic scoring weights and thresholds safely, reproducibly, and without violating fairness/explainability constraints.

## 1. Scope

This protocol applies to:
- `weights.merit_breakdown`
- `weights.confidence_components`
- routing thresholds in `thresholds`

It does not allow:
- adding sensitive demographic/socio-economic fields into merit scoring
- bypassing deterministic recommendation logic
- changing final decision ownership (committee remains final authority)

## 2. Required Inputs

- Frozen validation set (recommended: 50-200 candidates)
- Human review sheet with expected ordering or tier labels
- Baseline config snapshot hash from `data/evaluation_pack/scoring_config_snapshot.json`

## 3. Experiment Steps

1. Export baseline evaluation pack:
   - `python scripts/evaluation_pack.py --input data/candidates.json`
2. Propose one small change batch only:
   - each changed weight delta <= 0.05 absolute
   - sum of `merit_breakdown` weights remains 1.0
   - sum of `confidence_components` weights remains 1.0
3. Re-run evaluation pack on the same frozen validation set.
4. Compare to baseline:
   - recommendation agreement/disagreement
   - fallback rate
   - fairness group deltas
5. Record decision in experiment log.

## 4. Acceptance Criteria

Accept change only if ALL hold:
- No increase in fairness red flags across audit groups
- No degradation in committee-aligned ranking quality on validation set
- Explainability remains coherent (strengths/gaps still supported by evidence)
- Fallback rate not materially worse due to unrelated infra issues

## 5. Logging Template

For each run, store:
- experiment_id
- date/time
- author
- baseline_config_hash
- candidate_set_id
- changed_keys
- before/after values
- key metric deltas
- decision (accept/reject)
- rationale

## 6. Rollback Rule

Any accepted change must keep a rollback path:
- previous config snapshot hash
- previous weight set
- reproducible command to regenerate prior report
