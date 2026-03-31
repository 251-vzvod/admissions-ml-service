# Annotation Guide V2

## Purpose

This guide defines how to build the closest practical ground-truth dataset for the InVisionU screening task.

The target is not:

- final admission truth
- a black-box accept or reject label

The target is:

- committee-style primary review consensus
- shortlist priority
- hidden potential
- support need
- authenticity review need

This guide is intentionally built for the actual InVisionU use case:

- action over polish
- growth over prestige
- community contribution over self-marketing
- human-in-the-loop review support

## Annotation Unit

One annotation unit = one candidate application package.

Each package may include:

- structured application facts
- motivation letter
- answers to motivation questions
- video presentation transcript
- interview transcript
- roleplay notes or transcript
- optional additional materials summary

Annotators must judge the candidate only from the provided package.

Do not reward:

- polished English by itself
- school prestige
- family status
- geography by itself
- expensive extracurricular access

## Required Label Set

Each candidate must receive the following individual labels:

- `leadership_through_action` (1..5)
- `growth_trajectory` (1..5)
- `community_orientation` (1..5)
- `project_based_readiness` (1..5)
- `motivation_groundedness` (1..5)
- `evidence_strength` (1..5)
- `shortlist_priority` (1..5)
- `hidden_potential_flag` (`yes/no`)
- `needs_support_flag` (`yes/no`)
- `authenticity_review_needed` (`low/medium/high`)
- `label_confidence` (`low/medium/high`)

Each annotation must also include:

- `2-4 evidence bullets`
- short `annotation_notes`

## Recommended Review Setup

For the main benchmark:

- `3` independent annotators per candidate
- `1` adjudicator for disagreement resolution

For early-stage collection, the minimum acceptable setup is:

- `2` annotators per candidate
- adjudication only on disagreements

But true benchmark-quality work should aim for `3-way` review.

## Scale Definition

Use `1..5` consistently:

- `1` = very weak
- `2` = weak
- `3` = mixed / moderate
- `4` = strong
- `5` = very strong

Do not collapse the scale into only `3` and `4`.

## Rubric Definitions

### `leadership_through_action`

Question:

Did the candidate take responsibility, initiate action, organize people, or create a useful change?

Use:

- `1`: no real action, only aspiration
- `2`: small isolated action, mostly reactive
- `3`: some initiative, but limited scope or weak evidence
- `4`: clear self-driven action affecting other people or a group
- `5`: repeated, concrete, high-agency action with visible ownership or coordination

Reward:

- starting something
- taking responsibility without being asked
- building useful structure
- coordinating people
- improving a real situation

Do not reward:

- titles alone
- polished leadership language without action

### `growth_trajectory`

Question:

Does the profile show challenge, adaptation, reflection, and development over time?

Use:

- `1`: static self-description, no growth evidence
- `2`: mentions difficulty but no real adaptation
- `3`: some change or reflection, but shallow or isolated
- `4`: clear before/after growth story
- `5`: repeated, credible pattern of learning, adaptation, and resilience

### `community_orientation`

Question:

Is the candidate oriented toward improving life for others, not only personal advancement?

Use:

- `1`: fully self-focused ambition
- `2`: mentions helping others in abstract terms only
- `3`: some community concern, but weakly evidenced
- `4`: clear orientation toward useful contribution in real settings
- `5`: repeated, grounded pattern of community-oriented action

Reward:

- noticing local problems
- helping peers or community
- linking learning with contribution

### `project_based_readiness`

Question:

Does the candidate show readiness for a project-based, practical, iterative learning environment?

Use:

- `1`: no sign of building, testing, or practical implementation
- `2`: mostly theoretical or passive engagement
- `3`: one small practical example, but weak iteration
- `4`: real project or useful implementation with some adaptation
- `5`: repeated builder behavior with iteration, ownership, and problem-solving

### `motivation_groundedness`

Question:

Is the candidate's motivation grounded in real reasons, observed problems, and personal experience?

Use:

- `1`: generic ambition, template-like motivation
- `2`: partly personal but still mostly abstract
- `3`: mixed grounding
- `4`: specific, believable motivation linked to real experience
- `5`: deeply grounded motivation with clear program fit and practical self-awareness

### `evidence_strength`

Question:

How well are the important claims supported by examples, details, outcomes, and cross-section consistency?

Use:

- `1`: mostly unsupported claims
- `2`: very thin evidence
- `3`: some real evidence, but limited or uneven
- `4`: multiple concrete examples with useful grounding
- `5`: rich, consistent, well-supported evidence across sections

Important:

If the candidate sounds polished but evidence is thin, lower `evidence_strength` first.

### `shortlist_priority`

Question:

How strongly should this candidate be surfaced for closer committee review?

Use:

- `1`: low review priority
- `2`: likely below shortlist line
- `3`: plausible middle-case, compare manually
- `4`: strong shortlist candidate
- `5`: should clearly be surfaced near the top

This is not admission.
This is committee review priority.

### `hidden_potential_flag`

Question:

Does the candidate appear stronger than their self-presentation quality?

Mark `yes` only if:

- there is real action or trajectory signal
- the candidate looks under-marketed or modest relative to signal
- the evidence is not perfect, but there is enough real substance

Do not mark `yes` for:

- polished-but-thin cases
- generic motivation with weak evidence

### `needs_support_flag`

Question:

Would this candidate likely benefit from real onboarding, language, academic, or adjustment support if surfaced?

Mark `yes` when:

- there is credible promise
- current preparation looks incomplete
- the support need is real but does not erase upside

This is not a negative label.

### `authenticity_review_needed`

Question:

How strongly should the committee verify groundedness and consistency manually?

Use:

- `low`: no strong review concern
- `medium`: some concern, manual probing advised
- `high`: clear need for deeper verification

This is not a cheating verdict.

## Evidence Annotation Rules

Each annotation should include `2-4` evidence bullets.

Each evidence bullet should contain:

- source field
- short snippet
- why it supports the label

Example:

- source: `motivation_letter_text`
- snippet: `I rewrote the bot after classmates complained the schedule was confusing.`
- why: `Shows initiative plus adaptation after feedback.`

## Adjudication Rules

Adjudication is required when:

- `shortlist_priority` differs by `2+`
- `hidden_potential_flag` disagrees
- `authenticity_review_needed` disagrees by `2 levels`
- `label_confidence` is low for any annotator

Adjudicator should:

- read the full package
- compare annotator notes
- preserve evidence-backed disagreement when needed
- avoid averaging by reflex

## Pairwise And Batch Labels

In addition to per-candidate labels, collect:

- pairwise comparisons
- batch shortlist tasks

These are often more useful for ranking than a single scalar label.

### Pairwise Task

Prompt for reviewer:

Which candidate should be reviewed first by the committee, and why?

### Batch Task

Prompt for reviewer:

From this set of `10-20` candidates:

- pick top shortlist candidates
- pick hidden-potential candidates
- pick promising-but-needs-support candidates
- pick authenticity-review candidates

## Fairness Guardrails

Annotators must not use the following as merit advantages:

- family income
- family hardship by itself
- school prestige
- city prestige
- region prestige
- gender
- ethnicity
- social class cues

These may be documented separately for fairness audit only.

## Recommended Dataset Composition

For a strong benchmark, aim for:

- `300-500` real candidate packages
- `100-150` triple-annotated
- `50-100` adjudicated
- `300+` pairwise comparisons
- `20-30` batch shortlist tasks

And keep `Foundation` and `Bachelor` tagged separately.

## Final Rule

Annotators are not choosing who "deserves" admission.

They are producing the closest practical ground truth for:

- primary review priority
- hidden potential
- support-aware surfacing
- evidence-aware committee judgment
