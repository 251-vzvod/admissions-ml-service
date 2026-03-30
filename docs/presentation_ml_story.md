# Presentation ML Story

This document is a presentation-ready draft for the ML / NLP part of the hackathon pitch.

It is optimized for:

- fast slide writing
- clear judging narrative
- alignment with the current repository artifacts

## Slide 1. Problem

### Slide text

**Current screening sees the application, not the person.**

- Strong candidates with weak self-presentation get filtered out too early.
- Traditional essays are becoming less reliable because of generative AI.
- Manual review does not scale with candidate volume.
- inVision U needs a system that helps identify leadership potential, growth, and authentic motivation early.

### Speaker note

The core failure of a standard admissions pipeline is that it overweights how well a candidate describes themselves. For inVision U, that is dangerous, because the mission is to find future changemakers, not just polished applicants.

## Slide 2. Our Hypothesis

### Slide text

**We should rank candidates by potential, trajectory, and evidence, not by polish alone.**

We evaluate:

- leadership potential
- growth trajectory
- motivation authenticity
- evidence strength
- hidden potential

### Speaker note

Our goal is not to replace the committee. Our goal is to give the committee a better first-pass ranking and a clearer explanation of why a candidate deserves attention.

## Slide 3. What The System Sees

### Slide text

**Multimodal candidate representation**

Input sources:

- structured application fields
- motivation letter
- long-form Q&A
- interview transcript
- video transcript
- optional supporting materials

### Speaker note

We intentionally model the candidate as a bundle of signals, not as a single essay. This is important because real applicants are uneven: some write well, some speak better than they write, some have stronger evidence in transcripts or supporting materials.

### Repo references

- `docs/input_example.md`
- `docs/candidate_profiles_and_input_format.md`

## Slide 4. Baseline

### Slide text

**We started with a transparent baseline, not a black box.**

Baseline outputs:

- `merit_score`
- `confidence_score`
- `authenticity_risk`
- committee routing recommendation

Baseline principles:

- deterministic
- explainable
- privacy-safe
- human-in-the-loop

### Speaker note

This matters for the TЗ. We did not start from an opaque admissions model. We started from an interpretable scorer that the committee can audit.

### Repo references

- `app/services/scoring.py`
- `app/services/pipeline.py`
- `docs/scoring_system_evolution.md`

## Slide 5. Why Baseline Was Not Enough

### Slide text

**The first scorer still had important weaknesses.**

- It overvalued well-filled and polished applications.
- It was too lexical and too surface-level.
- It underperformed on hidden-potential candidates.
- It showed language bias, especially for Russian profiles.

### Speaker note

This was the turning point. The baseline was useful, but not good enough for the mission. It was still closer to a document-quality scorer than to a hidden-potential detection system.

## Slide 6. How We Improved It

### Slide text

**We added semantic rubric scoring and then calibrated the scorer against a rubric-defined benchmark.**

Improvements:

- semantic rubric layer
- final merged hackathon annotation set
- label-calibrated transparent scoring
- family-aware validation to reduce synthetic leakage optimism

### Speaker note

The semantic layer lets us move beyond keywords. The calibration stage then aligns the ranking logic with the shortlist behavior we actually want. Family-aware validation makes sure we do not fool ourselves with near-duplicate synthetic variants.

### Repo references

- `data/final_hackathon_annotations_v1.json`
- `docs/final_hackathon_annotation_report.md`
- `docs/family_aware_validation.md`

## Slide 7. ML / NLP Architecture

### Slide text

**ML extracts signals. Transparent scoring ranks. Human committee decides.**

Pipeline:

1. candidate preprocessing and normalization
2. heuristic + semantic feature extraction
3. transparent scoring
4. hidden-potential / support / authenticity signals
5. explanation with evidence spans
6. human committee decision

### Speaker note

This is important strategically. We are not pitching autonomous admissions. We are pitching a decision-support system with explainability and explicit human review.

### Repo references

- `app/utils/text.py`
- `app/services/text_features.py`
- `app/services/semantic_rubrics.py`
- `app/services/scoring.py`

## Slide 8. Data Strategy

### Slide text

**We used two data layers.**

Layer 1:

- external corpora for NLP robustness
- essay quality
- authenticity-risk contrast
- privacy-safe processing

Layer 2:

- candidate-like benchmark
- controlled counterfactual variants
- final hackathon annotations

### Speaker note

External datasets were not used as admissions truth. They were support corpora for the NLP layer. The shortlist logic was calibrated on our candidate benchmark and final annotation layer.

### Repo references

- `docs/ml_nlp_dataset_plan.md`
- `data/candidates_expanded_v1.json`
- `data/final_hackathon_annotations_v1.json`

## Slide 9. Measured Improvement

### Slide text

**After calibration, shortlist alignment improved materially.**

- `spearman_merit_vs_labels: 0.0119 -> 0.3798`
- `pairwise_accuracy: 0.4949 -> 0.6710`
- `precision@k_priority: 0.5625 -> 0.7500`
- `hidden_potential_recall@k: 0.1351 -> 0.1892`

Family-aware validation:

- `root-representative pairwise_accuracy = 0.6330`
- `family-aggregated precision@k_priority = 0.9167`

### Speaker note

These metrics are not claims about real-world deployment accuracy yet. They are claims about how much better the scorer now aligns with the rubric-defined shortlist behavior we designed and validated on the benchmark.

### Repo references

- `data/evaluation_pack_final_hackathon_v3/label_evaluation.json`
- `data/evaluation_pack_final_hackathon_v3/family_aware_validation.json`

## Slide 10. What The Committee Gets

### Slide text

**The output is not just a score. It is a review-ready candidate card.**

For each candidate:

- ranking recommendation
- top strengths
- main gaps
- uncertainties
- evidence spans
- hidden potential signal
- needs support signal
- authenticity review signal

### Speaker note

This is the product value. The committee gets a shortlist with reasons, not a black-box number.

### Repo references

- `README.md`
- `app/services/pipeline.py`
- `app/services/scoring.py`

## Slide 11. Fairness, Ethics, And Human-In-The-Loop

### Slide text

**The system is explicitly designed as a support tool, not an autonomous decision maker.**

- sensitive personal fields are excluded from merit scoring
- authenticity is a review signal, not an accusation
- final decision remains with the committee
- fairness audit is built into the evaluation flow

### Speaker note

This directly addresses the restrictions in the TЗ. We treat privacy, transparency, and human review as core design requirements, not as afterthoughts.

### Repo references

- `app/services/privacy.py`
- `data/evaluation_pack_final_hackathon_v3/fairness_audit.json`

## Slide 12. Limitations And Next Steps

### Slide text

**Current limitations**

- the final hackathon labels are pragmatic, not a production gold benchmark
- Russian profiles still underperform relative to English and mixed profiles
- synthetic multimodal evidence is cleaner than real applicant evidence

**Next steps**

- stronger multilingual semantic encoder
- further fairness calibration for RU / EN
- richer committee feedback loop
- real pilot with consented candidate data

### Speaker note

Being honest here makes the whole project stronger. We are showing a working, measurable, explainable system, while also being clear about what would make it more production-ready.

## Closing Message

If only one line is remembered, it should be this:

**We are not building a machine that decides who gets in. We are building an explainable ranking system that helps the committee find hidden-potential candidates that traditional screening would miss.**
