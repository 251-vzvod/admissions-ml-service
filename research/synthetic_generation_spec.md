# Synthetic Candidate Generation Spec

## Purpose

This file defines how to expand the English-only seed set with synthetic candidates without confusing synthetic coverage data with human ground truth.

Use this file when:

- generating new candidate batches for annotation
- building stress-test packs for the scorer
- increasing rubric coverage before real applications arrive

Related files:

- strategy: [roadmap.md](./roadmap.md)
- data program: [data_collection_spec.md](./data_collection_spec.md)
- human labeling rules: [annotation_guide_v1.md](./annotation_guide_v1.md)

## Core Rule

Synthetic candidates are useful for coverage, robustness checks, and additional human-reviewed examples.

Synthetic candidates are not ground truth by themselves.

Nothing in this workflow should treat generator intent, hidden archetype metadata, or model-produced labels as the final truth.

## Why We Are Doing This Now

Current English seed size is only `18` candidates.

Recent independent re-annotation showed substantial disagreement across previously labeled examples.
That means the immediate priority is still rubric stability.

So the role of synthetic generation is:

- expand scenario coverage
- create more annotation material
- stress-test the baseline on harder edge cases

It is not:

- a replacement for adjudicated labels
- a shortcut to a trusted training set
- a reason to skip disagreement analysis

## Batch V1 Scope

Target batch size for the first synthetic expansion:

- `48` English-only candidates

Target archetype mix:

- `8` `hidden_potential_low_polish`
- `6` `quiet_technical_builder`
- `6` `community_oriented_helper`
- `6` `hardship_responsibility_growth`
- `6` `academically_strong_but_narrow`
- `6` `polished_but_thin`
- `6` `support_needed_promising`
- `4` `borderline_manual_review`

Target ambiguity mix:

- `12` clear / easier cases
- `24` borderline cases
- `12` hard / ambiguous cases

This mix is deliberate.
We want more borderline and disagreement-prone cases than easy ones.

## What The Payload Must Look Like

Every generated candidate must stay inside the frozen public input contract accepted by the ML service.

Allowed top-level fields:

- `candidate_id`
- `structured_data`
- `text_inputs`
- `behavioral_signals`
- `metadata`

Allowed public content shape:

- `structured_data.education`
- `text_inputs.motivation_letter_text`
- `text_inputs.motivation_questions`
- `text_inputs.interview_text`

Notes:

- do not add `scenario_meta`
- do not add hidden tags inside the candidate payload
- do not add any extra label fields
- keep the payload valid for the current API contract

## Realism Constraints

All synthetic candidates in Batch V1 must be:

- English-language applications
- plausibly from Kazakhstan contexts
- varied in city, family situation, school context, confidence level, and opportunity access
- realistic in achievement level
- distinct in voice and examples

Avoid:

- perfect essay polish across the whole batch
- repetitive hardship narratives
- repeated projects with only names swapped
- too many elite competitions or rare international opportunities
- cartoonishly fake contradiction cases
- direct wording that reveals the intended archetype

Good diversity dimensions:

- urban vs smaller-city contexts
- strong academics vs average academics
- shy vs articulate self-presentation
- solo builder vs community-oriented helper
- family responsibility vs school-driven activity
- high evidence vs low evidence
- complete but modest vs borderline thin

## Text Mix Targets

Batch V1 should not be uniform.

Target text-length mix:

- `10` short
- `22` medium
- `16` long

Target interview availability:

- about `75%` with non-empty `interview_text`
- about `25%` without interview text

Target Q/A richness:

- some candidates with strong `motivation_questions` coverage
- some with only a few usable answers
- some borderline-thin but still valid

## What To Generate For Each Candidate

Each candidate needs:

- realistic `candidate_id`
- plausible education signals in `structured_data.education`
- one motivation letter
- optional but usually present `motivation_questions`
- optional `interview_text`
- plausible `behavioral_signals`
- metadata that clearly marks the batch as synthetic in the raw file only

Recommended `candidate_id` pattern:

- `syn_eng_v1_001`
- `syn_eng_v1_002`
- ...

Recommended raw metadata:

- `source = "synthetic_batch_v1"`
- realistic `submitted_at`
- `scoring_version = null`

## Hidden Generation Metadata

Generation metadata must be stored separately from the reviewer pack.

For each candidate, hidden generation metadata should include:

- `candidate_id`
- `intended_archetype`
- `intended_ambiguity`
- `intended_primary_signals`
- `intended_primary_risks`
- `generator_notes`

This metadata is useful for coverage analysis.
It must not be shown to annotators while they are labeling.

## File Layout

Raw generation package:

- `data/ml_workbench/raw/generated/batch_v1/synthetic_batch_v1_api_input.jsonl`
- `data/ml_workbench/raw/generated/batch_v1/synthetic_batch_v1_generation_manifest.jsonl`
- `data/ml_workbench/raw/generated/batch_v1/synthetic_batch_v1_summary.json`

Sanitized annotation package derived from the raw generation package:

- `data/ml_workbench/processed/annotation_packs/synthetic_batch_v1/synthetic_batch_v1_annotation_pack.jsonl`
- `data/ml_workbench/processed/annotation_packs/synthetic_batch_v1/synthetic_batch_v1_annotation_pack.json`
- `data/ml_workbench/processed/annotation_packs/synthetic_batch_v1/synthetic_batch_v1_annotation_pack_table.csv`
- `data/ml_workbench/processed/annotation_packs/synthetic_batch_v1/synthetic_batch_v1_annotation_pack_manifest.json`

Important:

- the raw package may contain `metadata`
- the sanitized annotation pack should remove `metadata`
- the sanitized annotation pack should contain only:
  - `candidate_id`
  - `structured_data`
  - `text_inputs`
  - `behavioral_signals`

## Annotation Policy For Synthetic Batches

Annotators must use the same rubric as the current English seed pack.

Rules:

- do not show generator intent
- do not show hidden archetype metadata
- do not prefill labels
- do not compare against the baseline during annotation

Synthetic candidates enter the useful dataset only after human review.

## Quality Gate Before Annotation

Before Batch V1 is accepted for annotation, verify:

- all records validate against the frozen `CandidateInput` schema
- all records are English-only
- there are no leaked archetype or label hints in visible fields
- candidate voices are sufficiently varied
- the archetype counts match the intended mix
- the ambiguity mix is roughly correct
- no more than a small minority feel obviously duplicated

## Success Criteria For Batch V1

Batch V1 is successful if it gives us:

- a broader set of realistic English-only candidate profiles
- more disagreement-bearing cases for rubric refinement
- better regression coverage for hidden potential, support-needed, polished-thin, and manual-review edge cases

Batch V1 is not successful if it produces:

- many near-duplicate essays
- obvious archetype leakage
- overly polished synthetic sameness
- labels that come from the generator instead of a reviewer

## Good Directions For Later Batches

After Batch V1 is reviewed, the strongest next synthetic directions are:

- contrastive candidate generation
- Russian-to-English translation candidates
- broader, values-aware motivation question sets

## Current Program Status

Synthetic expansion has now moved beyond Batch V1.

Current generated coverage includes:

- `contrastive_batch_v2`
- `translated_batch_v3`
- `messy_batch_v4`
- `messy_batch_v5`
- `messy_batch_v5_extension`
- `ordinary_batch_v6`
- `gap_fill_batch_v7`

Current role of these later batches:

- `v2`: contrastive supervision around nearby decision boundaries
- `v3`: translated Russian-origin candidates in English contract form
- `v4-v5`: messy-realism and lower-polish application coverage
- `v6`: ordinary disagreement-prone university applicants
- `v7`: targeted gap-fill for manual-review, insufficient-evidence, no-interview, translated-thinking-English, and support-needed-not-hidden-star cases

Operational rule:

- new synthetic batches should now be justified by a concrete slice gap or evaluation failure
- avoid adding more generic review-priority-heavy batches without a specific supervision need

### Contrastive Candidate Generation

Useful idea:

- take an existing candidate pattern and generate a deliberately contrasting candidate

Examples:

- strong academics vs weak academics
- high polish vs low polish
- real initiative vs mostly aspiration
- quiet builder vs visible social leader
- sincere modest profile vs borderline over-framed profile

Important:

- do not make contrastive candidates cartoonishly opposite
- they should feel like plausible neighboring cases, not parody
- keep the hidden relationship only in generation metadata, not in the visible reviewer pack

### Russian-To-English Translation Batches

Useful idea:

- take Russian-language internal candidates and create English translations for English-only research slices

Rules:

- preserve substance, evidence, and uncertainty
- do not silently improve polish too much
- keep the candidate's original level of confidence and specificity
- store a hidden link to the original source candidate id outside the reviewer pack

These translated cases should be treated as separate synthetic-translation research items, not as substitutes for original-language evaluation.

### Values-Aware Motivation Questions

Motivation question sets do not need to stay locked to one fixed list forever.

Good next step:

- mix standard personal-history prompts with inVision U value prompts

Examples of useful value-oriented prompts:

- teamwork under pressure
- learning across differences
- what kind of community the candidate wants to help build
- what local issue the candidate would turn into an inVision project
- what they expect to contribute to peers, not only gain for themselves

Important:

- vary question sets by batch and archetype
- do not let the question design itself reveal the intended label
