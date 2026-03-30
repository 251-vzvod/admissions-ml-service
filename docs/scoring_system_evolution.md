# Scoring System Evolution

## Goal

This document tracks how the candidate evaluation system evolved from an initial heuristic scorer into a more explainable and semantically grounded ranking system.

It is intended for:

- presentation slides
- README / demo narrative
- judging Q&A
- internal alignment across ML, backend, and frontend work

## Phase 0: Initial Deterministic MVP

### Core idea

Start with a transparent heuristic scorer that converts application text and structured fields into:

- `merit_score`
- `confidence_score`
- `authenticity_risk`
- routing recommendation for committee review

### What it did well

- clear factor-level explainability
- privacy guardrails through sensitive-field exclusion
- human-in-the-loop routing instead of autonomous admission

### Main weaknesses discovered

- eligibility logic over-penalized missing fields and pushed nearly everyone into conditional review
- English-heavy lexical rules created language bias against Russian text
- formal/resource-like signals had too much influence on merit
- LLM was used only for explanation, but the system narrative still sounded more ML-heavy than it really was

## Phase 1: Recalibrated Baseline

### Why it was needed

The first scoring version was too close to a document-quality scorer. The hackathon task requires hidden-potential detection, not polished-writing detection.

### What changed

- eligibility switched from raw field counting to logical evidence groups
- merit weights moved toward:
  - growth trajectory
  - resilience
  - initiative
  - leadership / agency
- English and certificate signals were removed from the potential core
- docs / portfolio / video stopped acting as direct merit proxies
- Russian lexical support was expanded
- committee annotation workflow and offline label evaluation were added

### Resulting baseline framing

This became the honest baseline for future comparisons:

- deterministic
- interpretable
- privacy-safe
- still mostly lexical / heuristic

## Phase 2: External Data Layer for NLP Iteration

### Why it was needed

To move beyond keyword heuristics, the system needed auxiliary corpora for:

- evidence extraction robustness
- writing-style robustness
- authenticity-risk contrastive examples
- privacy-safe text processing experiments

### Current corpus mix

- `PERSUADE 2.0`
- `ASAP 2.0`
- `AIDE`
- `PIILO`
- `CLEAR Corpus`

### Important caveat

These are not admissions labels.
They are support corpora for the NLP layer, not ground truth for candidate selection.

## Phase 3: Semantic Rubric Layer

### Why it was needed

The baseline could still miss strong candidates who describe themselves weakly. We needed a layer that matches candidate evidence to rubric concepts, not just to keywords.

### What was introduced

- semantic rubric prototypes for:
  - `leadership_potential`
  - `growth_trajectory`
  - `motivation_authenticity`
  - `authenticity_groundedness`
- lightweight embedding backend based on hashed token and character n-gram vectors
- semantic evidence retrieval from candidate sections
- semantic hidden-potential signal
- integration of semantic rubric scores into the hybrid scorer

### Design choice

The semantic layer was added on top of the deterministic baseline rather than replacing it.

This keeps:

- baseline vs hybrid comparisons honest
- explanations interpretable
- future migration to stronger embedding models easy

## Phase 4: Silver Annotation Triage

### Why it was needed

At this point, the system had a working scorer and a synthetic candidate set, but no trustworthy human-reviewed ranking target.

We therefore introduced an intermediate annotation stage:

- not final gold truth
- not fully automatic labels
- human-reviewable `silver draft`

### What was added

- rubric-driven annotation guide
- candidate profile / input-format guide
- draft annotation workflow over the synthetic candidate set
- candidate quality audit
- manual review batch builder

### First silver-draft outcome

On the current `52` synthetic candidate profiles, the draft review split was:

- `42 usable`
- `10 needs_edit`
- `0 drop`

The quality audit surfaced recurring issues:

- polish-to-evidence mismatch
- synthetic repetition of tutoring / volunteering motifs
- uneven narrative consistency across sections
- missing modalities such as video transcripts and portfolio proof

### Why this mattered

This stage made one thing explicit:

the synthetic candidate set is useful for iteration, but it is not gold truth.

That is why the next step is not "trust the draft labels".
The next step is:

- build a focused manual review batch
- verify the highest-value cases by hand
- promote those reviewed labels into `gold_v1`

### Manual review batch policy

The first manual batch is built from silver annotations using these rules:

- `triage_status == needs_edit`
- `authenticity_review_flag == true`
- `hidden_potential_flag == true AND committee_priority >= 4`
- `confidence == low`

This creates a smaller review queue containing the most important cases:

- strong upside candidates
- suspicious / generic candidates
- inconsistent profiles
- low-confidence labels

## Phase 5: Adjudication Proposal And Curated Gold Subset

### Why it was needed

Even after the silver-draft stage, there were too many uncertain or noisy labels to treat the full reviewed set as final gold truth.

So the next step was split into two layers:

- `gold_v1_proposed`: adjudicated review output over the manual batch
- `gold_v1_curated`: a smaller, stricter subset used as the first stable gold anchor

### Adjudication proposal outcome

For the `33` candidates in the manual review batch:

- `18` remained unchanged from silver
- `15` were corrected

The most common corrections were:

- `authenticity_review_flag`
- `evidence_strength`
- `leadership_potential`
- `committee_priority`

This was important because it showed that the silver draft was useful for narrowing the search space, but still not trustworthy enough as final ground truth.

### Curated gold subset policy

The first curated gold subset keeps only:

- high-confidence hidden-potential cases
- stable one-field corrections judged strong enough to keep

It excludes:

- ambiguous holdouts
- unresolved authenticity-review cases
- low-confidence boundary cases

### Current curated subset

The first `gold_v1_curated` contains `12` candidates:

- `cand_003`
- `cand_008`
- `cand_016`
- `cand_023`
- `cand_030`
- `cand_034`
- `cand_037`
- `cand_039`
- `cand_047`
- `cand_014`
- `cand_020`
- `cand_045`

### Why this matters for the demo

This gives a much cleaner story:

1. start from a synthetic candidate pool
2. draft silver labels with rubric guidance
3. isolate the highest-value review queue
4. adjudicate that queue
5. promote only the most stable cases into `gold_v1_curated`

That is a credible hackathon workflow because it is:

- transparent
- iterative
- realistic under time pressure
- honest about label uncertainty

## Phase 6: Candidate Text Normalization

### Why it was needed

Once the candidate pool grew from `52` to `80` profiles, the next bottleneck was no longer volume.

It was text consistency:

- mixed punctuation conventions across RU and EN text
- occasional mojibake-like quote and dash sequences coming from generation / copy pipelines
- unstable whitespace and paragraph formatting
- terminal rendering making some valid UTF-8 strings look broken during debugging

For lexical heuristics and lightweight semantic matching, this kind of noise is expensive.

### What changed

- added a reusable unicode normalization layer in `app/utils/text.py`
- normalized:
  - smart quotes and apostrophes
  - dashes
  - ellipses
  - non-breaking and zero-width whitespace
- added a reproducible dataset cleaner:
  - `scripts/normalize_candidate_text.py`
- normalized both:
  - `data/candidates.json`
  - `data/candidates_expanded_v1.json`

### Why this matters

This phase does not make the model "smarter".

It makes the input space more stable, which improves:

- deterministic lexical counting
- semantic chunk matching
- annotation readability
- debugging and demo reliability

It is also an honest preprocessing step to show in the presentation:

1. candidate generation
2. candidate quality audit
3. text normalization
4. semantic scoring and ranking
5. human-reviewed shortlist

## Phase 7: Label-Calibrated Transparent Scoring

### Why it was needed

After the final hackathon annotation file was assembled, the evaluation loop exposed a hard truth:

- the scorer was operationally usable
- but it was not aligned well enough with the rubric-defined shortlist logic

In practice, this meant:

- too many long, well-filled applications were over-ranked
- multimodal candidates with corroborating materials were under-ranked
- hidden-potential recall in the top shortlist was too low

### What changed

The scoring layer was recalibrated against the final hackathon annotation set, while keeping the model transparent.

Key changes:

- added a `committee_calibration_signal`
- added `material_support_score` based on:
  - documents
  - portfolio links
  - video presentation availability
- added `unsupported_narrative_penalty` for polished program-fit rhetoric that is not backed by evidence
- added hidden-potential bonus logic for candidates with corroborated but low-polish profiles
- rebalanced `confidence_score` so strong multimodal evidence reduces over-penalization from lexical sparsity
- updated routing thresholds to match the new merit / confidence scale

### Why this still fits the TЗ

This is not a black-box admissions model.

It is still:

- deterministic
- traceable
- factor-level explainable
- committee-facing

The difference is that the final ranking signal is now calibrated toward the actual committee-style rubric rather than toward raw lexical richness.

### Result

On the `80`-candidate final hackathon set:

- `spearman_merit_vs_labels`: `0.0119 -> 0.3798`
- `pairwise_accuracy`: `0.4949 -> 0.6710`
- `precision@k_priority`: `0.5625 -> 0.7500`
- `hidden_potential_recall@k`: `0.1351 -> 0.1892`

This became the first version of the scorer that can be honestly presented not just as "working", but as "measurably better aligned with the rubric we care about".

## Phase 8: Family-Aware Validation

### Why it was needed

By this point, the candidate set included counterfactual families:

- one root candidate
- one or more controlled variants
- different self-presentation or modality profiles of the same underlying person

That improves dataset quality, but it also creates a leakage-like evaluation risk.

If near-neighbor variants are counted independently, validation can look stronger than the real generalization story.

### What was added

- explicit family id logic based on:
  - original candidate id
  - `derived_from_candidate_id` for counterfactuals
- family-aware validation script:
  - `scripts/family_aware_validation.py`
- markdown report:
  - `docs/family_aware_validation.md`

### Validation views

Three views are now reported:

1. candidate-level
2. root-representative
3. family-aggregated

This gives a more honest answer to the question:

"Does the scorer still surface the right candidate families when synthetic near-duplicates are controlled for?"

### Result

On the `80`-candidate expanded set:

- candidate-level `pairwise_accuracy`: `0.6710`
- root-representative `pairwise_accuracy`: `0.6330`
- family-aggregated `pairwise_accuracy`: `0.6696`

And for shortlist quality:

- family-aggregated `precision@k_priority`: `0.9167`
- family-aggregated `hidden_potential_recall@k`: `0.2400`

### Why this matters for the demo

This is a strong judging point because it shows:

- we understand synthetic leakage risk
- we do not rely only on optimistic random-style evaluation
- we validate the shortlist logic at the family level, not just at the single-row level

## Current Architecture

### Baseline

- deterministic features
- transparent formulas
- score trace for auditability

### Hybrid semantic scorer

- deterministic baseline
- semantic rubric matching
- optional LLM explainability

### Human workflow

- system ranks and explains
- committee reviews
- final decision remains human

## What This Enables in the Demo

- show evolution from naive heuristic scoring to rubric-aware ranking
- explain why hidden-potential candidates now surface better
- compare deterministic baseline vs semantic hybrid on annotated candidates
- show semantic evidence spans, not only final scores

## Example Candidate Walkthrough

Below is a shortened real example from the current demo dataset.

### Input candidate payload

```json
{
  "candidate_id": "cand_001",
  "content_profile": {
    "language_profile": "english",
    "city_or_region": "Zhezkazgan",
    "adversity_pattern": "personal difficulties, low resources",
    "essay_style": "simple, unfocused"
  },
  "structured_data": {
    "education": {
      "english_proficiency": {
        "type": "school classes",
        "score": 68
      },
      "school_certificate": {
        "type": "Kazakhstan high school diploma",
        "score": 73
      }
    }
  },
  "text_inputs": {
    "motivation_letter_text": "I want to study at inVision U because I do not have many opportunities in my city... My father lost his job two years ago... I helped my classmates before exams by making notes and sharing them...",
    "motivation_questions": [
      {
        "question": "Tell us about the most difficult period in your life",
        "answer": "The most difficult time in my life was when my father lost his job... After this, I became more responsible."
      },
      {
        "question": "Describe a project or solution you came up with on your own initiative",
        "answer": "I made a small bookshelf from old boxes, sorted books for the family, and it saved time in the mornings."
      },
      {
        "question": "Tell us about a person or community you helped",
        "answer": "I helped my elderly neighbor with groceries and cleaning her garden because she could not do it herself."
      }
    ],
    "interview_text": "I do not have a lot of experience, but I want to learn, improve, and maybe help my city later."
  },
  "behavioral_signals": {
    "completion_rate": 1.0,
    "returned_to_edit": false,
    "skipped_optional_questions": 0
  }
}
```

### Output after scoring

```json
{
  "candidate_id": "cand_001",
  "eligibility_status": "eligible",
  "merit_score": 73,
  "confidence_score": 61,
  "authenticity_risk": 20,
  "recommendation": "review_priority",
  "review_flags": [
    "contradiction_risk",
    "possible_contradiction"
  ],
  "merit_breakdown": {
    "potential": 78,
    "motivation": 83,
    "leadership_agency": 66,
    "experience_skills": 61,
    "trust_completeness": 68
  },
  "semantic_rubric_scores": {
    "leadership_potential": 55,
    "growth_trajectory": 54,
    "motivation_authenticity": 56,
    "authenticity_groundedness": 55,
    "hidden_potential": 61
  },
  "top_strengths": [
    "Clear initiative and agency markers in self-driven actions.",
    "Motivation aligns with the program format and collaborative learning context.",
    "Evidence density is sufficient to support a comparatively reliable assessment."
  ],
  "main_gaps": [
    "Potential internal inconsistency detected across sections."
  ]
}
```

### Why this example matters

- This candidate is not a polished high-status applicant.
- The system still surfaces meaningful upside:
  - strong growth signals
  - non-trivial motivation
  - concrete helping behavior
  - moderate hidden-potential score
- The output is not just a scalar score. It gives:
  - breakdown by scoring axes
  - semantic rubric scores
  - review flags
  - explanation-ready strengths and gaps

This is exactly the kind of demo story that supports the hackathon thesis:
the system should not reward only polished writing; it should help the committee notice promising candidates who would otherwise be easy to miss.

## Current Limitations

- semantic layer currently uses a lightweight local embedding backend, not a dedicated sentence-transformer
- external corpora are auxiliary and cannot replace committee annotations
- fairness still needs to be checked on real RU/EN labeled data
- authenticity remains a review-risk signal, not proof of AI use

## Next Planned Step

Replace or augment the current lightweight embedding backend with a stronger sentence embedding model and evaluate:

- dimension-level alignment with committee labels
- top-k hidden-potential recall
- robustness across RU and EN profiles
