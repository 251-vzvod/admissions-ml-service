# ML/NLP Dataset Plan

## Goal

Build a dataset that supports three separate tasks:

1. `Ranking hidden potential`
2. `Evidence-grounded explainability`
3. `Authenticity / uncertainty review routing`

The target is not "find one ideal candidate vector".
The target is to place candidates in a rubric-shaped semantic space with separate axes:

- leadership potential
- growth trajectory
- motivation authenticity
- evidence strength
- community orientation
- support needs

## Recommended Dataset Shape

Use one row per candidate plus one annotation object.

### Candidate payload

```json
{
  "candidate_id": "cand_001",
  "structured_data": {},
  "text_inputs": {
    "motivation_letter_text": "...",
    "motivation_questions": [
      {"question": "...", "answer": "..."}
    ],
    "interview_text": "...",
    "video_interview_transcript_text": "...",
    "video_presentation_transcript_text": "..."
  },
  "metadata": {
    "language_profile": "ru|en|kk|mixed",
    "source_type": "open|synthetic|partner|internal_demo"
  }
}
```

### Annotation payload

```json
{
  "candidate_id": "cand_001",
  "leadership_potential": 4,
  "growth_trajectory": 5,
  "motivation_authenticity": 4,
  "evidence_strength": 3,
  "committee_priority": 4,
  "hidden_potential_flag": true,
  "needs_support_flag": true,
  "authenticity_review_flag": false,
  "review_notes": "Strong growth and agency, weak English, should not be down-ranked."
}
```

## How To Use The Dataset

### Baseline phase

- Use committee labels as the main target.
- Evaluate whether merit ranking agrees with `committee_priority`.
- Track `hidden_potential_recall_at_k`.
- Track pairwise ranking:
  `high_potential_weak_self_presentation > low_potential_strong_presentation`

### ML/NLP phase

- Encode candidate chunks with a multilingual embedding model.
- Build rubric prototypes, not one global "ideal candidate".
- Example prototypes:
  - `high_growth_low_polish`
  - `community_leader_early_stage`
  - `mission_fit_builder`
  - `promising_but_support_needed`
- Score each candidate against rubric dimensions separately.

## Semantic Space Strategy

Do not compare every candidate to one "perfect student".
That would overfit to style and punish non-standard talent.

Use one semantic anchor set per rubric dimension:

- `leadership potential` anchors
- `growth trajectory` anchors
- `mission fit` anchors
- `evidence strength` anchors
- `authenticity review` negative anchors

Each anchor can be represented by:

- short rubric descriptions
- committee-written positive/negative examples
- labeled candidate excerpts

Then candidate scoring becomes:

1. retrieve strongest evidence chunks
2. compare chunk embeddings to rubric anchors
3. aggregate dimension-level scores
4. keep deterministic routing and explanations

## Open-Source Data To Collect

Collect data by artifact type, not by one single "admissions dataset":

- scholarship / fellowship application essays
- student leadership essays and personal statements
- debate, volunteer, startup, olympiad, community initiative stories
- youth interview transcripts or long-form Q/A
- synthetic counterfactual rewrites:
  same underlying candidate, stronger vs weaker self-presentation

## Minimum Useful Mix

- 80-150 candidate-like profiles for annotation
- 20-30 hidden-potential positives
- 20-30 polished-but-low-evidence negatives
- balanced language coverage across `ru`, `en`, and `mixed`
- at least 2 reviewers on 25-30 profiles for agreement check

## What To Avoid

- one scalar "ideal candidate similarity"
- demographic proxies in merit scoring
- language fluency as a shortcut for potential
- authenticity score as an auto-rejection mechanism

## Workflow In This Repo

1. Build annotation pack:
   `python scripts/build_annotation_pack.py --input data/candidates.json --output data/annotation_pack.json`
2. Fill committee labels in the generated pack.
3. Run evaluation:
   `python scripts/evaluation_pack.py --input data/candidates.json --annotations data/annotation_pack.json --output-dir data/evaluation_pack`
4. Fetch external corpora:
   `python scripts/fetch_external_datasets.py --mode open`
5. Build seed corpus from downloaded data:
   `python scripts/build_seed_corpus.py --output data/external/seed_corpus.jsonl`
6. Optional: manually place Kaggle datasets into `data/external/raw/learning_agency_*` and rerun the seed builder.

## Practical Source Mix In This Repo

### Open pull now

- `CLEAR Corpus`: essay/readability support
- `PERSUADE 2.0`: argumentative student essays
- `MediaSum`: interview transcript support
- your internal RU/EN candidate corpus and synthetic counterfactual rewrites

### Kaggle pull when credentials are available

- `ASAP 2.0`
- `AIDE`
- `PIILO`
- `KLICKE`

`ASAP 2.0`, `AIDE`, and `PIILO` are consumed directly by `scripts/build_seed_corpus.py`.
`KLICKE` is kept as a raw writing-process corpus and is not included in the seed corpus by default.

These Kaggle datasets are officially listed by The Learning Agency here:
https://the-learning-agency.com/guides-resources/datasets/

For the scoring/ranking evolution narrative used in demos and slides, see:
`docs/scoring_system_evolution.md`
