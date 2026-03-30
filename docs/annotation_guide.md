# Annotation Guide

## Purpose

This guide defines how to create the `gold dataset` for candidate ranking.

The goal is not to predict admission automatically.
The goal is to teach and evaluate the system against committee-style judgment on:

- hidden potential
- growth trajectory
- leadership potential
- motivation authenticity
- evidence strength
- practical committee prioritization

## Unit of Annotation

One annotation row corresponds to one candidate profile.

Candidate evidence may include:

- `motivation_letter_text`
- `motivation_questions[].answer`
- `interview_text`
- `video_interview_transcript_text`
- `video_presentation_transcript_text`

Annotators should judge the candidate based only on the provided payload.
Do not infer demographic advantage, family status, city prestige, or school prestige as merit.

## Label Set

Each candidate should be annotated with:

- `leadership_potential`
- `growth_trajectory`
- `motivation_authenticity`
- `evidence_strength`
- `committee_priority`
- `hidden_potential_flag`
- `needs_support_flag`
- `authenticity_review_flag`
- `review_notes`

## Scale

Use `1..5` for numeric labels:

- `1` = very weak
- `2` = weak
- `3` = mixed / moderate
- `4` = strong
- `5` = very strong

Important:

- use the full scale
- avoid giving everyone `3` or `4`
- compare candidates to the rubric, not to each other

## Rubric Definitions

### `leadership_potential`

Question:
Does this candidate show evidence of agency, responsibility, influence, initiative, or organizing capacity?

Use:

- `1`: no evidence of initiative; passive aspirations only
- `2`: small isolated examples; mostly reactive, limited agency
- `3`: some initiative or responsibility, but narrow in scope or weakly evidenced
- `4`: clear self-driven action affecting other people or a group
- `5`: repeated, concrete, high-agency behavior with visible influence, coordination, or ownership

Look for:

- started something
- organized others
- took responsibility
- defended values under pressure
- created structure, process, or improvement

Do not over-reward:

- polished wording without action
- pure academic success without agency

### `growth_trajectory`

Question:
Does the profile show learning, adaptation, reflection, and development over time?

Use:

- `1`: static self-description, no reflection, no growth evidence
- `2`: mentions hardship or failure but no real learning
- `3`: some reflection and some change, but limited depth
- `4`: clear before/after story with lessons and adaptation
- `5`: strong pattern of reflection, iteration, resilience, and growth across multiple examples

Look for:

- setbacks
- what changed afterwards
- repeated effort
- re-planning after mistakes
- clearer worldview after experience

### `motivation_authenticity`

Question:
Does the candidate sound genuinely motivated for this program and grounded in real reasons?

Use:

- `1`: generic ambition, empty dream language, no concrete reason
- `2`: some personal language, but mostly template-like and vague
- `3`: partly grounded, partly generic
- `4`: clear reasons tied to personal experience, goals, or values
- `5`: highly concrete, believable, specific motivation linked to mission, context, and future direction

Look for:

- why this program specifically
- why now
- connection between experience and goals
- non-generic language

Do not over-reward:

- fluent English
- formal tone
- “I want to change the world” without grounded reasons

### `evidence_strength`

Question:
How well are the candidate's claims supported by examples, facts, actions, outcomes, or reflection?

Use:

- `1`: claims are unsupported; almost no examples
- `2`: weak examples, mostly vague narrative
- `3`: some examples, but incomplete or weakly grounded
- `4`: multiple concrete examples with actions and outcomes
- `5`: rich, specific, cross-sectionally consistent evidence with believable detail

Look for:

- who / what / when / why
- concrete actions
- outcomes
- details that make the story believable
- consistency across sections

### `committee_priority`

Question:
If this were a real shortlist process, how strongly should the committee prioritize this profile for deeper review or interview?

Use:

- `1`: low priority
- `2`: probably not worth prioritizing
- `3`: borderline / maybe
- `4`: strong priority for review
- `5`: top priority candidate

This is the most operational label.
It should reflect the committee's practical ranking judgment after seeing all evidence.

Important:

- `committee_priority` is not the same as current polish
- hidden-potential candidates may deserve `4` or `5` even if their presentation is weak

## Binary Flags

### `hidden_potential_flag`

Set to `true` when:

- the candidate appears stronger than their self-presentation suggests
- growth, values, agency, or resilience are meaningful
- a naive polished-writing scorer could underrate them

Typical pattern:

- weak English or simple expression
- limited formal achievements
- but strong lived responsibility, initiative, reflection, or community behavior

### `needs_support_flag`

Set to `true` when:

- the candidate looks promising
- but may need language support, academic support, onboarding support, or confidence support

This flag does not mean weak merit.
It means promising candidate + probable support need.

### `authenticity_review_flag`

Set to `true` when:

- text is overly generic
- claims are unusually polished but weakly grounded
- sections feel mismatched
- examples feel thin, inflated, or suspiciously template-like

This flag is for human review only.
It is not proof of AI use or dishonesty.

## Annotation Procedure

For each candidate:

1. Read all available text inputs.
2. Mark short evidence snippets for yourself.
3. Score the four rubric dimensions.
4. Assign `committee_priority`.
5. Set the three binary flags.
6. Write 1-3 short notes in `review_notes`.

Recommended annotation order:

1. `evidence_strength`
2. `growth_trajectory`
3. `leadership_potential`
4. `motivation_authenticity`
5. `committee_priority`
6. flags

This reduces the risk of judging the candidate by writing polish first.

## Annotation Principles

### Judge potential, not polish

The task is to help the system notice strong candidates even when they describe themselves weakly.

### Do not reward privilege proxies

Do not directly reward:

- elite school name
- city prestige
- English fluency alone
- certificates alone
- access to extracurricular resources alone

These may affect readiness, but they should not dominate potential.

### Do not punish simple language

Simple language is not equal to low potential.

### Reward grounded evidence

Specific action beats abstract aspiration.

### Human uncertainty is allowed

If the candidate is ambiguous, use `3` and explain why in `review_notes`.

## Suggested Reviewer Setup

Minimum:

- `1` reviewer for all profiles

Better:

- `2` reviewers for at least `25-30` profiles

Recommended workflow:

- first pass: independent labeling
- second pass: resolve high-disagreement cases
- final pass: update rubric wording if confusion repeats

## Disagreement Rules

Revisit the candidate when:

- any numeric label differs by `2+`
- one reviewer sets `hidden_potential_flag=true` and another sets `false`
- `committee_priority` differs by `2+`

In those cases:

- compare evidence
- discuss rubric interpretation
- keep final adjudicated label
- preserve disagreement notes if useful

## Example Patterns

### High hidden potential

- weak polish
- strong family or community responsibility
- concrete initiative
- strong reflection after hardship
- limited formal achievements

Expected pattern:

- `growth_trajectory=4 or 5`
- `leadership_potential=3 or 4`
- `committee_priority=4`
- `hidden_potential_flag=true`

### Polished but low-signal

- fluent, formal, impressive tone
- vague motivation
- no grounded examples
- generic impact language

Expected pattern:

- `motivation_authenticity=2 or 3`
- `evidence_strength=1 or 2`
- `committee_priority=2 or 3`
- `authenticity_review_flag=true` possible

## Data Format

Use the generated pack:

```bash
python scripts/build_annotation_pack.py --input data/candidates.json --output data/annotation_pack.json
```

The resulting file contains:

- candidate context
- evidence payload
- empty rubric fields to fill

## Minimal Target Size

For the first useful gold dataset:

- `80-150` candidate profiles
- at least `20-30` hidden-potential positives
- at least `20-30` polished-but-low-evidence negatives
- at least `25-30` double-annotated profiles

## Output Quality Goal

After annotation, the dataset should allow us to answer:

- does the system align with committee judgment?
- does it recover hidden-potential candidates?
- does it over-reward polished writing?
- does it behave differently on RU vs EN profiles?
