# Next ML Improvements Plan

This document is the working roadmap for the next ML / NLP / LLM / explainability improvements.

It exists so the team can keep a stable priority order and avoid losing the main direction during rapid hackathon iteration.

## Current Situation

What is already strong:

- transparent scoring pipeline
- final hackathon annotation set
- calibrated scorer with measurable uplift
- family-aware validation
- privacy-safe merit scoring
- human-in-the-loop positioning
- presentation and project narrative docs

What is still weak:

- Russian-language fairness
- hidden-potential recall
- committee-facing explainability depth
- authenticity review logic granularity

## Priority Order

## Priority 1. Multilingual Semantic Layer

### Current status

- configurable semantic backend infrastructure implemented
- available modes:
  - `hash`
  - `tfidf_char_ngram`
  - `sentence_transformer` (optional dependency path)
- default remains `hash` for now

### Why default is still `hash`

A first `tfidf_char_ngram` experiment was implemented and evaluated, but it did not produce a clean enough uplift on the final hackathon validation pack.

So the infrastructure is now ready, but the system has **not** yet switched its default backend away from `hash`.

### Goal

Reduce RU bias and make semantic scoring less dependent on English-shaped lexical rules.

### Why this matters

This is currently the biggest ML weakness in the system.
The fairness audit still shows a large gap between English / mixed profiles and Russian profiles.

### Recommended approach

Replace or augment the current lightweight hash-embedding layer with a stronger multilingual encoder.

Best pragmatic options:

- `intfloat/multilingual-e5-base`
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

### Expected effect

- better semantic matching for `leadership / growth / motivation`
- reduced lexical bias
- stronger fairness story for the judges

### Implementation note

Do not replace the whole transparent pipeline.
Use the embedding model as a better semantic feature backend, while keeping the scorer and trace layer explainable.

## Priority 2. Hidden Potential Uplift

### Goal

Improve recall for candidates who are stronger than their self-presentation suggests.

### Why this matters

This is one of the core differentiators of the whole project.
If the system cannot reliably surface hidden-potential candidates, it misses the most important pain point from the TЗ.

### Recommended approach

Add a stronger explicit signal for:

- real growth
- real agency
- real evidence
- weaker polish or weaker presentation quality

Potential implementation:

- `presentation_vs_signal_gap`
- stronger `hidden_potential_score`
- better boost logic in shortlist ranking

### Expected effect

- better `hidden_potential_recall@k`
- stronger product differentiation
- stronger demo story

## Priority 3. Committee-Facing Explainability

### Goal

Make the output more useful for the actual admissions workflow.

### Why this matters

The current output is already explainable, but it can become much more committee-friendly.

### Recommended additions

For each candidate, generate:

- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`
- `why_hidden_potential` if relevant
- `why_support_needed` if relevant

### Expected effect

- stronger Explainable AI story
- stronger human-in-the-loop narrative
- better demo UX even before frontend polish

## Priority 4. Authenticity Review Logic

### Goal

Improve the system's ability to flag suspiciously generic or weakly grounded profiles without pretending to detect cheating with certainty.

### Why this matters

The TЗ explicitly highlights the problem of generative AI reducing trust in essays.

### Recommended approach

Strengthen distinction between:

- weak but honest candidate
- generic candidate
- polished but weakly grounded candidate
- suspiciously mismatched candidate

Possible improvements:

- stronger cross-section mismatch logic
- more groundedness checks
- more detailed authenticity review reasons

### Expected effect

- better review routing
- stronger trust story for the committee
- better handling of GenAI-related concerns

## Priority 5. Cohort-Level Candidate Views

### Goal

Make ranking outputs easier to use and explain.

### Recommended cohorts

- `High priority`
- `Hidden potential`
- `Promising but needs support`
- `Polished but low-evidence`
- `Authenticity review needed`

### Expected effect

- stronger demo and UX
- more intuitive output for judges and committee

## LLM Usage Guidance

### What LLM should NOT do

- final autonomous admissions decision
- ungrounded final scoring
- opaque black-box ranking

### What LLM can do well

- explanation generation
- evidence summarization
- follow-up question generation
- structured rubric extraction
- committee-facing natural-language output

### Practical runtime guidance

- for local development and no-credit mode: use `Ollama`
- for offline evaluation: keep LLM disabled when not needed
- for final polished demo: optionally switch back to OpenAI

## Suggested Next Execution Order

1. implement multilingual semantic backend
2. improve hidden-potential uplift logic
3. upgrade committee-facing explainability
4. refine authenticity-review logic
5. package cohort-style outputs for demo

### Progress note

Priority 3 has already started:

- committee-facing cohorts
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`

## Current Working Rule

When choosing the next task, prefer:

- fairness improvement
- hidden-potential detection improvement
- explainability improvement

over:

- collecting more synthetic data
- adding more random model complexity
- cosmetic refactors with low demo impact
