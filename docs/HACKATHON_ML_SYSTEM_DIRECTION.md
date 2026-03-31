# Hackathon ML/NLP Direction

## Main Positioning

The service should not become "just another essay scorer".

The strongest positioning for the hackathon is:

- we do not reward who writes more polished applications
- we help the committee surface candidates with real leadership potential and growth trajectory
- we keep the system transparent and human-in-the-loop

Core product framing:

`hidden potential + trajectory of growth + committee support workflow`

## What Is Already Strong

- Human-in-the-loop by design
- Explainability already exists
- Merit / confidence / authenticity risk are separated
- Sensitive fields are excluded from merit scoring
- Batch scoring already exists
- There is already a baseline, semantic layer, annotations, and validation story
- Shortlist-oriented outputs now exist
- Hidden-potential logic is now materially stronger than the earlier scorer-only version
- English-first operation is now the practical product assumption
- Lightweight multilingual support still exists, but is no longer the main optimization target

## What Is Still Weak

- Core is still closer to a strong heuristic scorer than a stronger ML ranking system
- Fairness is still not fully closed, but the main risk is now polish bias within English rather than multilingual handling
- Runtime LLM is unstable and slow, so it cannot be a hard dependency
- AI detection must remain a weak review signal only
- Ranking is stronger than before, but still not a true learned ranker
- Section consistency is now implemented, but claim-to-evidence grounding still needs to become stronger and more visible

## What Must Be Reduced

- Any penalty for weak self-presentation by itself
- Any hidden bias toward polished English
- Too much merit weight on documents / portfolio / video presence
- Any wording that sounds like "AI text detected" as a final verdict
- Any narrative that sounds like "LLM decides"
- Any product surface that exposes too many raw correlated internal features

## What Has Already Been Added

- Hidden Potential Score
- Support Needed Score
- Shortlist Priority Score
- Batch shortlist view
- Evidence Coverage Score
- Better growth trajectory extraction
- Better committee-facing guidance
- Lightweight multilingual semantic upgrade

## What Still Needs To Be Added

- Better section consistency logic
- Better claim-evidence extraction
- Pairwise ranking logic
- Better top-N shortlist reranking

Status:

- section consistency is now implemented
- claim-evidence extraction is now active
- pairwise ranking is now the most valuable next ML step
- stronger multilingual encoder is optional future work, not a core hackathon priority anymore

## ML / NLP Priorities

### Priority 1

- strengthen ranking / shortlist logic
- strengthen hidden potential logic
- strengthen trajectory of growth
- improve fairness for English-first inputs without making the system black-box
- make the committee workflow more explicit in outputs

Status:

- largely in progress
- hidden potential / shortlist / trajectory are now substantially stronger than before
- fairness still needs better slice-based evaluation and mitigation

### Priority 2

- section-to-section consistency checks
- claim-evidence extraction improvements
- pairwise ranking logic
- stronger validation on shortlist ordering

Status:

- section consistency is now done
- claim-evidence extraction is in progress
- pairwise ranking and shortlist validation are now the main unfinished ML block

### Priority 3

- reranking for top-N candidates
- better predictive / engagement features when data exists
- richer multimodal signals when real product inputs exist

Status:

- not a hackathon-critical priority yet

## LLM Policy

- LLM must not drive final numeric scoring
- deterministic scoring first
- LLM only for explanation and committee support

Use LLM for:

- committee summary
- follow-up interview question
- verification checklist
- concise reviewer card

Do not use LLM for:

- final merit scoring
- final shortlist authority
- authenticity verdicts

## Explainable AI Direction

The service should tell the committee not only what the model thinks, but also:

- why the candidate surfaced
- why the candidate is not top priority yet
- what evidence is missing
- what should be verified manually
- what support might unlock the candidate

Status:

- mostly implemented at the product layer
- claim-to-evidence grounding is now partially implemented, but still needs stronger coverage and calibration

## Current Build Priorities

1. Keep shortlist-first outputs stable and demo-ready
2. Improve claim-to-evidence grounding
3. Add transparent pairwise shortlist reranking
4. Continue fairness work through English-first slice-based checks, not multilingual expansion by default
5. Keep the shortlist-first narrative tightly aligned with the actual product behavior

## Guardrails

- No autonomous admission decision
- No demographic or socio-economic merit proxies
- No black-box ranking narrative
- No AI detector as final truth
- No deploy path that depends on heavy optional models by default

## Working Rule

All future improvements should be judged by one question:

Does this make the system better at surfacing strong early-stage leaders with transparent reasoning and lower bias?

If not, it is not a priority for the hackathon build.
