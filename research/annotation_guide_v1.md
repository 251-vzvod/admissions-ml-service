# Annotation Guide V1

## Purpose

This guide is for the first human annotation round on the English-only admissions seed set.

Use this guide with:

- `data/ml_workbench/processed/annotation_packs/admissions_seed_annotation_pack_v1.jsonl`
- `data/ml_workbench/processed/annotation_packs/admissions_seed_annotation_pack_v1.json`
- `data/ml_workbench/processed/annotation_packs/admissions_seed_annotation_pack_v1_table.csv`
- `data/ml_workbench/labels/human_labels_individual.csv`

## Ground Rules

- annotate the candidate, not the writing polish alone
- do not try to guess what the current model would predict
- do not infer hidden metadata not present in the pack
- treat this as shortlist support, not final admission
- notes should reference evidence from the application, not vague impressions

Important:

- this pack is intentionally sanitized
- synthetic hints and non-contract metadata were removed to reduce leakage

## What To Read

For each candidate, read:

- `structured_data.education`
- `text_inputs.motivation_letter_text`
- `text_inputs.motivation_questions`
- `text_inputs.interview_text`
- `behavioral_signals`

Do not over-weight:

- pure polish
- grammar quality alone
- generic ambition without evidence

## Main Labels

These are the fields you should fill in `human_labels_individual.csv`.

### `recommendation`

Allowed values:

- `review_priority`
- `standard_review`
- `manual_review_required`
- `insufficient_evidence`
- `incomplete_application`
- `invalid`

How to choose:

- `review_priority`: strong candidate for early committee attention
- `standard_review`: looks viable, but not urgent priority
- `manual_review_required`: promising or uncertain case that needs human checking because of inconsistency, weak grounding, or ambiguity
- `insufficient_evidence`: too thin to judge fairly
- `incomplete_application`: materially incomplete application
- `invalid`: broken / unusable / disallowed input

### `committee_priority`

Use a `1-5` scale:

- `1`: clearly low priority for shortlist
- `2`: below average priority
- `3`: borderline / medium priority
- `4`: strong shortlist candidate
- `5`: top shortlist priority

Use this as a shortlist-read-order judgment, not as a writing score.

### `shortlist_band`

Boolean:

- `true`: this candidate deserves shortlist-level attention
- `false`: this candidate does not currently deserve shortlist-level attention

Set `true` when the candidate appears strong enough that the committee should seriously consider them in a top group.

### `hidden_potential_band`

Boolean:

- `true`: the underlying candidate signal looks stronger than the self-presentation quality
- `false`: hidden-potential framing is not justified

Set `true` only when most of the following are true:

- there is real evidence of growth, initiative, or community orientation
- the candidate seems underestimated by writing quality or polish
- the case is not just weak and underdeveloped

Do not use `hidden_potential_band=true` for:

- generic but unsupported optimism
- polished but thin applications
- candidates with little actual evidence

### `support_needed_band`

Boolean:

- `true`: candidate looks promising but may need onboarding, language, academic, or adaptation support
- `false`: this framing is not central to the case

Set `true` when the candidate has visible promise but also looks likely to need structured support to succeed.

### `authenticity_review_band`

Boolean:

- `true`: this application should receive extra manual checking for groundedness or consistency
- `false`: no special authenticity review needed

Set `true` when you see:

- contradictions
- weak grounding for strong claims
- suspicious mismatch between sections
- over-polished but low-evidence writing

Do not set `true` just because:

- the English is simple
- the writing is awkward
- the candidate is modest

### `reviewer_confidence`

Use a `1-5` scale:

- `1`: very low confidence in your judgment
- `2`: low confidence
- `3`: medium confidence
- `4`: high confidence
- `5`: very high confidence

This measures your certainty about the judgment, not the quality of the candidate.

### `notes`

Write `1-4` short lines worth of reasoning.

Good notes:

- mention concrete evidence
- explain why the candidate is priority / borderline / manual-review
- mention what is missing if evidence is thin

Bad notes:

- `good profile`
- `seems smart`
- `not sure`

## Suggested Annotation Order

For each candidate:

1. decide whether the application is usable
2. judge shortlist priority on substance, not polish
3. decide whether hidden potential is present
4. decide whether support-needed framing is present
5. decide whether authenticity review is needed
6. assign recommendation
7. assign confidence
8. write short evidence-based notes

## Practical Heuristics

### Signs Of Stronger Merit

- concrete action, not only aspiration
- growth after difficulty or setback
- evidence of initiative, helping others, or responsibility
- credible examples with outcomes or specifics
- grounded motivation for the program

### Signs Of Thin Evidence

- mostly generic future statements
- vague talk about impact without examples
- no concrete scene, action, or result
- repeated claims that never get supported

### Signs Of Hidden Potential

- modest or weak self-presentation, but clear real actions
- meaningful growth under constrained conditions
- community orientation or initiative that is small-scale but credible
- better underlying signal than writing polish suggests

### Signs Of Support Needed

- candidate seems capable but underprepared
- lower confidence due to language or academic transition risk
- good promise combined with adaptation challenges

### Signs Of Authenticity Review Need

- noticeable contradictions
- very polished style with low evidence
- different sections feel misaligned
- claims seem larger than the concrete support provided

## What We Are Not Labeling Yet

Do not try to label:

- final admit / reject
- scholarship decision
- exact future success

This round is about shortlist support and review routing.

## First-Round Goal

For v1, consistency matters more than sophistication.

A smaller, cleaner, repeatable annotation set is more useful than overly clever notes with unstable judgment.
