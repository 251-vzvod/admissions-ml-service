# Fairness Note

## Purpose

This note explains:

- what fairness risk currently matters most in the service
- what has already been mitigated
- what small targeted mitigation was added most recently
- what is still not solved

The goal is to be honest and operational, not to overclaim fairness.

## Main Fairness Risk

The current practical fairness risk is no longer "multilingual support at any cost".

The product assumption is now English-first.

That means the main remaining fairness risk is:

- polished English vs plain English
- verbose narrative vs concise but concrete narrative
- confident self-presentation vs real but under-marketed signal

## What Was Already In Place

Before the latest mitigation, the system already had several fairness-oriented design choices:

- sensitive personal and socio-economic fields are excluded from merit scoring
- missing modalities reduce assessment confidence, not candidate worth by default
- hidden-potential logic explicitly tries to surface candidates whose underlying signal is stronger than their presentation quality
- the semantic layer already includes a lightweight multilingual bridge, but this is no longer the main optimization focus
- section-to-section consistency is used instead of relying only on polished essay quality

These choices do not solve fairness, but they prevent some of the most obvious failure modes.

## Latest Targeted Mitigation

The latest fairness change is intentionally narrow and transparent.

### What changed

For Cyrillic-heavy and mixed-script profiles, the system now slightly reduces style-based penalties when there is already reasonable evidence that the profile is coherent.

This stays in the codebase as a narrow safeguard, not as the main fairness strategy.

This mitigation uses:

- `cyrillic_text_share`
- `latin_text_share`
- `evidence_count`
- `specificity_score`
- `consistency_score`
- `cross_section_mismatch_score`

### Where it applies

It only affects style-sensitive penalties such as:

- genericness-based penalty
- polished-but-empty penalty

It does **not**:

- guarantee a higher merit score
- suppress contradiction risk
- override low evidence
- change the human-in-the-loop requirement

### Why this is still a reasonable compromise

The goal is not to "boost Russian candidates".

The goal is to avoid a narrower failure mode:

`coherent Cyrillic or mixed-language profiles being over-penalized just because the style layer reads them as less polished`

## What Is Still Not Solved

- fairness is still not proven on real applicant distributions
- English-first fairness slices still need explicit evaluation:
  - polished vs plain
  - verbose vs concise
  - high evidence vs low evidence
  - transcript present vs absent
- style bias can still leak through because the system is still partly heuristic
- the AI detector remains English-first and should stay optional

## How To Present This Honestly

The correct fairness story is:

- we identified bias from presentation style and language handling as a real risk
- we reduced one narrow multilingual failure mode with a targeted safeguard
- we are now focusing fairness work primarily on English-first polish bias
- we did not pretend it is solved
- we kept the system transparent enough for the committee to understand and override it

## Recommended Next Step

The next stronger fairness step is:

- run explicit English-first fairness slices over current outputs
- compare:
  - polished vs plain English
  - concise vs verbose English
  - high evidence vs low evidence
- keep only the mitigation that improves shortlist fairness without destroying explainability
