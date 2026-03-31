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

## What Is Weak

- Core is still closer to a strong heuristic scorer than a stronger ML ranking system
- RU bias is not fully addressed
- Runtime LLM is unstable and slow, so it cannot be a hard dependency
- AI detection must remain a weak review signal only
- The system is stronger at individual scoring than shortlist ranking

## What Must Be Reduced

- Any penalty for weak self-presentation by itself
- Any hidden bias toward polished English
- Too much merit weight on documents / portfolio / video presence
- Any wording that sounds like "AI text detected" as a final verdict
- Any narrative that sounds like "LLM decides"

## What Must Be Added

- Hidden Potential Score
- Support Needed Score
- Shortlist Priority Score
- Batch shortlist view
- Evidence Coverage Score
- Better section consistency logic
- Better growth trajectory extraction

## ML / NLP Priorities

### Priority 1

- strengthen ranking / shortlist logic
- strengthen hidden potential logic
- strengthen trajectory of growth
- improve fairness for RU / EN
- make the committee workflow more explicit in outputs

### Priority 2

- multilingual semantic encoder
- pairwise ranking logic
- claim-evidence extraction improvements
- section-to-section consistency checks

### Priority 3

- reranking for top-N candidates
- better predictive / engagement features when data exists
- richer multimodal signals when real product inputs exist

## LLM Policy

- LLM must not drive final numeric scoring
- deterministic scoring first
- LLM only for explanation and committee support

Use LLM for:

- committee summary
- follow-up interview question
- verification checklist
- concise reviewer card

## Explainable AI Direction

The service should tell the committee not only what the model thinks, but also:

- why the candidate surfaced
- why the candidate is not top priority yet
- what evidence is missing
- what should be verified manually
- what support might unlock the candidate

## Current Build Priorities

1. Add transparent hidden potential / support needed / shortlist priority outputs
2. Add shortlist summary to batch scoring
3. Keep the service stable without mandatory LLM runtime dependence
4. Improve fairness without making the system black-box

## Guardrails

- No autonomous admission decision
- No demographic or socio-economic merit proxies
- No black-box ranking narrative
- No AI detector as final truth

## Working Rule

All future improvements should be judged by one question:

Does this make the system better at surfacing strong early-stage leaders with transparent reasoning and lower bias?

If not, it is not a priority for the hackathon build.
