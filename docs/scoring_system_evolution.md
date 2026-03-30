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
