# Candidate Profiles And Input Format

## Goal

This document answers three practical questions:

1. where to get candidate profiles for the `gold dataset`
2. how those profiles should be constructed
3. what input format the scoring service expects

Use this together with:

- [annotation_guide.md](C:/Users/Admin/Desktop/decentrathon/ml-service/docs/annotation_guide.md)
- [input_example.md](C:/Users/Admin/Desktop/decentrathon/ml-service/docs/input_example.md)
- [ml_nlp_dataset_plan.md](C:/Users/Admin/Desktop/decentrathon/ml-service/docs/ml_nlp_dataset_plan.md)

## Important Principle

Do not wait for a perfect real admissions dataset.

For the hackathon, the right move is to build a `candidate-like gold dataset`:

- realistic enough to reflect committee judgment
- safe enough to avoid privacy problems
- diverse enough to train and test ranking logic

## Where To Get Candidate Profiles

### 1. Current synthetic candidate set

Use the profiles you already have in:

- `data/candidates.json`

This is your fastest starting point.

What to do with it:

- keep it as the first annotation batch
- identify weak archetype coverage
- add new profiles only where the coverage is thin

### 2. Counterfactual rewrites of existing candidates

This is one of the highest-value sources.

Take one existing candidate and create:

- stronger self-presentation version
- weaker self-presentation version
- same underlying life story, different writing polish

Why it matters:

- directly tests whether the model over-rewards polish
- directly supports `hidden_potential_flag`
- gives strong pairwise evaluation examples

Example pair:

- `same candidate, weak English, concrete actions`
- `same candidate, polished essay, vague evidence`

### 3. Team-written candidate archetypes

Ask your team or mentors to write short candidate-like profiles based on realistic archetypes:

- rural high-growth candidate
- olympiad student with weak motivation
- community helper with weak English
- polished high-status applicant with low evidence
- resilient candidate with family burden and strong agency

These are not fake in a bad sense.
They are controlled archetypes for evaluation.

### 4. Public stories transformed into candidate-style profiles

You can use open public material as inspiration, but not by copying it directly.

Good sources of inspiration:

- youth volunteering stories
- social initiative stories
- student competition profiles
- young founder stories
- scholarship stories

How to use them safely:

- paraphrase
- anonymize
- convert into a candidate-style profile
- remove names and identifying details

Do not:

- scrape private or semi-private social data
- use real minors' identifiable profiles
- copy personal statements verbatim

### 5. Internal demo profiles from committee perspective

If you can talk to domain experts or mentors, ask them:

- what kinds of candidates are often underestimated?
- what kinds of candidates look strong on paper but disappoint later?

Turn those answers into candidate profiles.

This is often more valuable than another generic external corpus.

## Best Mix For The Gold Dataset

Recommended first version:

- `40-60` current synthetic profiles
- `20-30` counterfactual rewrites
- `20-40` new archetype-based profiles

This gives:

- realistic coverage
- stress-testing against polish bias
- enough variety for annotation and ranking evaluation

## What Good Candidate Profiles Should Contain

Each candidate should provide enough evidence for at least two logical text channels:

- motivation letter
- question/answer block
- interview or video transcript

Best case:

- all three are present

Weak but still usable:

- motivation letter + Q/A
- Q/A + interview

Poor for annotation:

- only one short paragraph

## Required And Optional Inputs

### Required

- `candidate_id`
- `text_inputs`

### Strongly recommended

- `motivation_letter_text`
- `motivation_questions`
- `interview_text` or video transcript

### Optional but useful

- `behavioral_signals`
- `structured_data.education`
- `structured_data.application_materials`
- metadata for analysis only

## Canonical Input Shape

Below is the practical candidate shape the service expects.

```json
{
  "candidate_id": "cand_001",
  "structured_data": {
    "education": {
      "english_proficiency": {
        "type": "IELTS",
        "score": 6.5
      },
      "school_certificate": {
        "type": "high school diploma",
        "score": 89
      }
    },
    "application_materials": {
      "documents": [],
      "attachments": [],
      "portfolio_links": [],
      "video_presentation_link": null
    }
  },
  "text_inputs": {
    "motivation_letter_text": "...",
    "motivation_questions": [
      {
        "question": "...",
        "answer": "..."
      }
    ],
    "interview_text": "...",
    "video_interview_transcript_text": "...",
    "video_presentation_transcript_text": "..."
  },
  "behavioral_signals": {
    "completion_rate": 1.0,
    "returned_to_edit": false,
    "skipped_optional_questions": 0,
    "meaningful_answers_count": 12,
    "scenario_depth": 0.8
  },
  "metadata": {
    "source": "synthetic|internal_demo|counterfactual|partner",
    "submitted_at": "2026-03-31T12:00:00Z",
    "scoring_version": "v1"
  },
  "consent": true
}
```

## Fields That Should Not Drive Merit

These may exist in payloads, but should not be used as direct merit signals:

- gender
- ethnicity
- race
- religion
- income
- social background
- full address
- family status as a prestige proxy

The service already excludes these from scoring.

## Recommended Profile Construction Rules

When writing or transforming candidate profiles:

- keep the story internally consistent
- include at least one concrete action
- include at least one reflection or lesson
- include at least one future-facing motivation statement
- avoid over-optimizing language polish

Good profile structure:

1. context
2. challenge
3. action
4. outcome
5. reflection
6. motivation for program

## Recommended Profile Categories To Cover

Make sure the dataset contains all of these:

- high growth, low polish
- strong academics, weak motivation
- strong motivation, weak evidence
- community-oriented candidate
- resilient candidate with family responsibilities
- polished but generic candidate
- candidate needing support
- candidate with authenticity-review risk

## Recommended Language Mix

Current focus:

- Russian
- English
- mixed RU/EN

Suggested first-pass mix:

- `35-45%` Russian
- `35-45%` English
- `10-20%` mixed

## How To Store The Gold Dataset

Keep two linked files:

### Candidate file

Contains raw candidate payloads:

- `data/candidates.json`
- or future `data/gold_candidates.json`

### Annotation file

Contains committee labels:

- `data/annotation_pack.json`
- or future adjudicated `data/gold_annotations.json`

This separation is important.
It keeps raw evidence separate from labels.

## Practical Next Step

1. Start from `data/candidates.json`
2. Create `20-30` counterfactual rewrites
3. Add `20-40` new archetype profiles
4. Generate annotation pack
5. Annotate using the rubric guide
6. Run:

```bash
python scripts/evaluation_pack.py --input data/candidates.json --annotations data/annotation_pack.json --output-dir data/evaluation_pack
```

## Short Answer

Where to get profiles:

- current synthetic set
- counterfactual rewrites
- team-written archetypes
- anonymized public-story-inspired profiles

What should candidate input look like:

- one JSON object per candidate
- structured fields + text inputs
- ideally at least two text channels
- realistic, evidence-bearing, and privacy-safe
