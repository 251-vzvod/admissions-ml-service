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

## Phase 8.5. Configurable Semantic Backend Experiment

### Why this experiment was needed

After calibration and family-aware validation, the biggest remaining ML weakness was still multilingual fairness, especially for Russian-language candidates.

The existing semantic rubric layer used a lightweight local hash embedding backend.
That backend was transparent and fast, but it was still too weak to confidently claim that semantic matching was robust across `RU/EN`.

### What was implemented

The semantic layer was refactored into a configurable backend abstraction.

Added backend options:

- `hash`
- `tfidf_char_ngram`
- `sentence_transformer`

Configuration was exposed through environment variables and included in the scoring snapshot:

- `SEMANTIC_BACKEND`
- `SEMANTIC_MODEL`

This made it possible to run controlled experiments without rewriting the scoring pipeline.

### What was tested

A first multilingual-oriented experiment switched the semantic layer from `hash` to `tfidf_char_ngram`.

The idea was:

- reduce English-shaped lexical dependence
- strengthen character-level multilingual matching
- improve semantic rubric alignment with the final hackathon labels

### What happened in evaluation

The result was honest but not strong enough to keep as the default.

Compared with the calibrated `v3` setup, the `tfidf_char_ngram` experiment:

- slightly reduced overall rank alignment
- did not improve hidden-potential recall
- did not improve Russian fairness enough

Observed direction of change:

- `spearman_merit_vs_labels`: `0.3798 -> 0.371`
- `pairwise_accuracy`: `0.6710 -> 0.6662`
- `hidden_potential_recall@k`: no meaningful gain
- Russian fairness metrics remained weak and in some places became slightly worse

### Decision

The backend abstraction was kept, but the default backend was switched back to `hash`.

This was an intentional product decision:

- keep the useful infrastructure
- do not claim improvement where there is none
- leave stronger backends available as controlled experimental modes

### Final outcome of this phase

What stayed in the repo:

- configurable semantic backend infrastructure
- `hash` / `tfidf_char_ngram` / `sentence_transformer` switchability
- semantic backend settings in config snapshots

What did **not** become the new default:

- `tfidf_char_ngram`

### Honest takeaway

This phase did not produce a metric win, but it was still valuable.

It converted the semantic layer from a fixed implementation into an experimental platform, so future multilingual upgrades can be tested honestly and reproducibly instead of being hardcoded into the pipeline.

## Phase 9. Committee-Facing Guidance Layer

### Problem

The scoring API already returned transparent scores and explanations, but the output was still more model-centric than committee-centric.

The admissions team needs workflow guidance, not just numeric axes.

### What was added

- `committee_cohorts`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`

### Why this matters

This keeps the system aligned with the TЗ:

- explainable AI
- human-in-the-loop
- decision support instead of autonomous admissions

The output now tells the committee not only **what** the score is, but also:

- why this candidate surfaced
- what kind of case this is
- what the committee should verify next
- what question is most useful to ask in follow-up review

### Design choice

This layer stays deterministic and grounded in transparent feature signals.

We did **not** turn follow-up guidance into an opaque LLM-only decision layer.

### Practical effect

The ML/NLP stack now supports two levels of explainability:

1. scoring transparency
2. committee-facing actionability

## Phase 10. Auxiliary AI-Generation Detector

### Why this was added

The TЗ explicitly highlights the problem of generative AI reducing trust in essays.

At the same time, a raw "AI-generated probability" is too weak and too risky to use as a final decision signal.

So the detector was added as a constrained auxiliary feature, not as a standalone verdict engine.

### What was implemented

An optional authenticity sublayer based on:

- `desklib/ai-text-detector-v1.01`

with strict guardrails:

- English-only by default
- weak signal only
- never affects `merit_score`
- never auto-rejects a candidate
- contributes only to authenticity review routing

### New output fields

- `authenticity_review_reasons`
- `ai_detector`

The API now returns:

- whether the detector was enabled
- whether it was applicable
- detected language
- probability of AI-generated text
- detector note / fallback note

### Why this design is safer

This keeps the system aligned with the project's explainability principles.

The detector is not treated as truth.
It is treated as one more weak signal alongside:

- genericness
- evidence density
- cross-section mismatch
- contradiction risk
- groundedness

### Practical interpretation

The correct committee-facing interpretation is:

"This profile should be reviewed more carefully for authenticity."

not:

"This profile was proven to be AI-generated."

## Phase 11. Shortlist-First Committee Layer

### Why this was added

The system already produced explainable candidate scores, but it still behaved more like an individual scorer than a shortlist engine for committee workflow.

The main hackathon need is not:

- "score this essay"

but:

- "help the committee surface hidden-potential candidates and build a smarter shortlist"

So the next iteration shifted the system toward shortlist logic, trajectory awareness, and explicit committee support.

### What was added

New derived outputs:

- `hidden_potential_score`
- `support_needed_score`
- `shortlist_priority_score`
- `evidence_coverage_score`
- `trajectory_score`

New batch outputs:

- `ranked_candidate_ids`
- `shortlist_candidate_ids`
- `hidden_potential_candidate_ids`
- `support_needed_candidate_ids`
- `authenticity_review_candidate_ids`

### Why this is different from just adding more scores

These are workflow-oriented signals:

- `hidden_potential_score`: helps identify candidates whose underlying growth and leadership signal is stronger than their self-presentation
- `support_needed_score`: reframes some borderline candidates as promising but likely needing onboarding, language, or academic support
- `shortlist_priority_score`: moves the system closer to committee review order, not just independent candidate scoring
- `trajectory_score`: makes growth-path logic more explicit
- `evidence_coverage_score`: helps separate strong signal with support from good narrative with thin grounding

### What changed in NLP features

The text layer was extended with explicit trajectory-related signals:

- `trajectory_challenge_score`
- `trajectory_adaptation_score`
- `trajectory_reflection_score`
- `trajectory_outcome_score`

This was done to make the system better at detecting:

- challenge
- action
- adaptation
- outcome
- reflection

instead of relying only on broader growth or resilience heuristics.

### Committee-facing effect

The guidance layer now more explicitly surfaces cases like:

- `Hidden potential`
- `Trajectory-led candidate`
- `Promising but needs support`

This improves alignment with the hackathon thesis:

- do not reward polish alone
- surface early-stage leaders
- support human review with next-step guidance

## Phase 11.5. Public API Simplification

Once the shortlist-oriented fields were added, the public response became too noisy.

The service had started to expose too many engineering-facing diagnostics alongside committee-facing outputs.

So the public contract was simplified.

Removed from the main committee-facing API surface:

- `feature_snapshot`
- prompt versioning metadata
- extractor versioning metadata

Kept public:

- core scoring outputs
- shortlist-oriented outputs
- explanation and reviewer guidance
- optional reviewer-detail fields such as `merit_breakdown`, `semantic_rubric_scores`, `authenticity_review_reasons`, and `ai_detector`

This made the service easier to consume for:

- judges
- frontend
- backend
- committee demo flows

## Phase 12. Hidden Potential Recalibration

### Why this was needed

After the shortlist-first layer was added, one important weakness remained:

- `hidden_potential_score` existed
- but its scale was still too conservative on realistic growth cases
- and committee cohorts could still miss candidates whose underlying signal was stronger than their writing quality

### What changed

The hidden-potential logic was recalibrated around a more explicit distinction between:

- `underlying signal`
- `self-presentation quality`
- `credible action / evidence floor`
- `overstatement risk`

This made the signal less dependent on noisy genericness alone and more sensitive to:

- adaptation
- reflection
- challenge-response pattern
- evidence that a candidate actually acted, even if the writing remained imperfect

### Committee-layer alignment

The committee guidance layer was also aligned with the derived shortlist signals.

Instead of using a separate hidden-potential heuristic that could disagree with the score, the cohort logic now relies more directly on:

- `hidden_potential_score`
- `trajectory_score`
- `evidence_coverage_score`

### Practical result

A realistic contrast case was checked locally:

- candidate with stronger growth / adaptation / early leadership but imperfect presentation
- candidate with smoother, thinner, more generic narrative

Observed behavior after recalibration:

- the growth-oriented case now surfaces as `Hidden potential`
- the polished-thin case does not
- `why_candidate_surfaced` now better matches the intended product story

### Why this matters

This phase does not solve multilingual fairness yet.

But it does strengthen the most important product thesis for the hackathon:

- do not reward polish alone
- surface early-stage leaders whose signal is stronger than their self-presentation

## Phase 13. Lightweight Multilingual Semantic Upgrade

### Why this was needed

The semantic layer was originally designed to be multilingual-friendly, but in practice it still had two weaknesses:

- the default backend remained too shallow for cross-language semantic matching
- part of the Russian rubric prototype layer had degraded text quality

This meant the system could still under-read Russian growth and leadership narratives even before the fairness layer was addressed directly.

### What changed

The semantic prototype layer was cleaned and upgraded without adding heavy runtime requirements to the default deploy path.

Changes:

- repaired Russian rubric prototypes
- added a bilingual RU/EN concept bridge for the default hash semantic backend
- aligned semantic matching around shared concepts such as:
  - leadership
  - growth
  - motivation
  - groundedness
  - challenge
  - adaptation
  - impact

### Why this approach was chosen

A full multilingual encoder is still a strong next step, but it is heavier operationally.

For the hackathon build, the best tradeoff was:

- improve multilingual semantic behavior now
- keep the default deploy lightweight
- preserve an optional path to `sentence-transformers` later

### Deployment-safe outcome

The default semantic backend remains lightweight.

Optional stronger semantic path:

- `SEMANTIC_BACKEND=sentence-transformer`
- `SEMANTIC_MODEL=intfloat/multilingual-e5-base`

but the system no longer depends on that path to get basic RU/EN semantic behavior.

### Why this matters

This phase is important because it improves the system in exactly the direction defined by the hackathon thesis:

- lower bias toward polished English
- better semantic matching for multilingual candidates
- stronger hidden-potential and trajectory reasoning without making the deploy heavier by default
