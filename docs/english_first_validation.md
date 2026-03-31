# English-First Validation

## Scope

- total candidates: `80`
- english-first candidates analyzed: `49`
- focus: `polish bias + shortlist robustness within English-first inputs`

## Presentation Style

### middle

- count: `15`
- avg merit: `45.8`
- avg confidence: `57.4`
- avg authenticity risk: `28.533`
- avg hidden potential: `54.133`
- avg shortlist priority: `54.333`
- label pairwise accuracy: `0.5806`
- label precision@k priority: `1.0`
- hidden potential recall@k: `0.1667`

### plain

- count: `17`
- avg merit: `33.588`
- avg confidence: `48.529`
- avg authenticity risk: `53.471`
- avg hidden potential: `31.765`
- avg shortlist priority: `33.941`
- label pairwise accuracy: `0.8824`
- label precision@k priority: `1.0`
- hidden potential recall@k: `0.2`

### polished

- count: `17`
- avg merit: `46.0`
- avg confidence: `56.235`
- avg authenticity risk: `28.941`
- avg hidden potential: `56.941`
- avg shortlist priority: `57.471`
- label pairwise accuracy: `0.7423`
- label precision@k priority: `0.3333`
- hidden potential recall@k: `0.125`

## Verbosity

### concise

- count: `17`
- avg merit: `33.588`
- avg confidence: `48.529`
- avg authenticity risk: `53.471`
- avg hidden potential: `31.765`
- avg shortlist priority: `33.941`
- label pairwise accuracy: `0.8824`
- label precision@k priority: `1.0`
- hidden potential recall@k: `0.2`

### medium

- count: `15`
- avg merit: `48.0`
- avg confidence: `60.267`
- avg authenticity risk: `23.133`
- avg hidden potential: `54.4`
- avg shortlist priority: `56.0`
- label pairwise accuracy: `0.625`
- label precision@k priority: `1.0`
- hidden potential recall@k: `0.1429`

### verbose

- count: `17`
- avg merit: `44.059`
- avg confidence: `53.706`
- avg authenticity risk: `33.706`
- avg hidden potential: `56.706`
- avg shortlist priority: `56.0`
- label pairwise accuracy: `0.7097`
- label precision@k priority: `0.3333`
- hidden potential recall@k: `0.1429`

## Evidence Profile

### evidence_middle

- count: `15`
- avg merit: `38.8`
- avg confidence: `48.267`
- avg authenticity risk: `43.267`
- avg hidden potential: `54.6`
- avg shortlist priority: `50.4`
- label pairwise accuracy: `0.5443`
- label precision@k priority: `0.3333`
- hidden potential recall@k: `0.0`

### evidence_strong

- count: `17`
- avg merit: `52.176`
- avg confidence: `64.294`
- avg authenticity risk: `15.941`
- avg hidden potential: `56.529`
- avg shortlist priority: `60.941`
- label pairwise accuracy: `0.3485`
- label precision@k priority: `0.3333`
- hidden potential recall@k: `0.1111`

### evidence_thin

- count: `17`
- avg merit: `33.588`
- avg confidence: `48.529`
- avg authenticity risk: `53.471`
- avg hidden potential: `31.765`
- avg shortlist priority: `33.941`
- label pairwise accuracy: `0.8824`
- label precision@k priority: `1.0`
- hidden potential recall@k: `0.2`

## Transcript Presence

### transcript_absent

- count: `4`
- avg merit: `19.0`
- avg confidence: `34.25`
- avg authenticity risk: `56.75`
- avg hidden potential: `29.0`
- avg shortlist priority: `25.25`
- label pairwise accuracy: `0.5`
- label precision@k priority: `0.0`
- hidden potential recall@k: `0.25`

### transcript_present

- count: `45`
- avg merit: `43.644`
- avg confidence: `55.667`
- avg authenticity risk: `35.6`
- avg hidden potential: `48.978`
- avg shortlist priority: `50.4`
- label pairwise accuracy: `0.6513`
- label precision@k priority: `0.7778`
- hidden potential recall@k: `0.15`

## Interpretation

- Use this report to detect whether shortlist behavior changes materially across English-first presentation slices.
- The most important comparison is no longer `RU vs EN`, but `polished vs plain`, `verbose vs concise`, and `evidence-strong vs evidence-thin`.
- Any future mitigation should improve these slices without making the scorer opaque.
