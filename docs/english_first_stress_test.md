# English-First Stress Test

## Scope

- focus candidates: `20`
- perturbations: `concise, polished_wrapper, evidence_removed, transcript_removed`

## concise

- avg shortlist priority delta: `-39.15`
- avg hidden potential delta: `-35.65`
- avg confidence delta: `-29.0`
- avg authenticity risk delta: `38.35`
- recommendation change rate: `0.7`
- cohort change rate: `1.0`
- claim layer change rate: `1.0`

## polished_wrapper

- avg shortlist priority delta: `-0.05`
- avg hidden potential delta: `0.1`
- avg confidence delta: `0.05`
- avg authenticity risk delta: `2.2`
- recommendation change rate: `0.0`
- cohort change rate: `0.0`
- claim layer change rate: `0.5`

## evidence_removed

- avg shortlist priority delta: `-40.2`
- avg hidden potential delta: `-36.45`
- avg confidence delta: `-29.25`
- avg authenticity risk delta: `40.05`
- recommendation change rate: `0.75`
- cohort change rate: `1.0`
- claim layer change rate: `1.0`

## transcript_removed

- avg shortlist priority delta: `-3.1`
- avg hidden potential delta: `-2.0`
- avg confidence delta: `-3.15`
- avg authenticity risk delta: `-0.85`
- recommendation change rate: `0.15`
- cohort change rate: `0.4`
- claim layer change rate: `1.0`

## Interpretation

- `concise` tests whether shorter English answers are punished too aggressively.
- `polished_wrapper` tests sensitivity to presentation polish alone.
- `evidence_removed` tests whether the system appropriately drops confidence and shortlist priority when grounding disappears.
- `transcript_removed` tests reliance on transcript channels.
