# Fairness Note

## Purpose

This note explains the main fairness risk in the current ML service and the current mitigation posture.

## Current Product Assumption

The runtime system is English-only in practice.

That means the main fairness risk is no longer multilingual handling.

The main risk is now style bias within English:

- polished English vs plain English
- verbose narrative vs concise but concrete narrative
- confident self-presentation vs real but under-marketed signal

## What Is Already In Place

- sensitive personal and socio-economic fields are excluded from merit scoring
- missing modalities reduce confidence, not candidate worth
- hidden-potential logic tries to surface candidates whose underlying signal is stronger than their writing quality
- authenticity review relies on evidence, consistency, and groundedness rather than on polish alone
- LLM is optional and does not own final scoring

## Current Mitigation Direction

The main mitigation is now evidence-first scoring:

- action over rhetoric
- grounded examples over abstract ambition
- growth and adaptation over smooth self-presentation
- community/value orientation over prestige-coded language

There is also a small style-bias safeguard in the scoring/authenticity layer.

It reduces style-sensitive penalties only when there is already enough evidence that the profile is concrete and internally consistent.

## What Is Still Not Solved

- fairness is still not proven on real applicant distributions
- English-only style bias can still leak through heuristic layers
- transcript-present vs transcript-absent slices still need explicit evaluation
- weak evidence and weak writing can still be hard to disentangle in edge cases

## Honest Presentation

The correct fairness story is:

- we are not claiming fairness is solved
- we intentionally removed multilingual ambition from the runtime scope
- we are focusing on a narrower and more honest fairness problem: English-language polish bias
- the system remains transparent enough for committee override
