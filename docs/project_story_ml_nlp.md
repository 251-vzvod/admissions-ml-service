# Project Story: ML / NLP Layer

This document tells the story of how the ML / NLP layer evolved during the hackathon.

It is written as a narrative rather than a technical changelog, so it can be reused for:

- presentation preparation
- defense Q&A
- README background
- internal project alignment

## 1. Where We Started

At the beginning, the problem looked deceptively simple: score candidate applications and help the admissions committee review them faster.

But once we mapped the task to the inVision U mission, it became obvious that a naive screening system would fail for exactly the candidates the university is most likely to care about.

The central challenge was not just ranking essays.
The central challenge was:

**How do we find real leadership potential, authentic motivation, and growth trajectory early, especially when a candidate is weak at self-presentation?**

That question shaped the whole ML / NLP direction.

We quickly aligned on one core principle:

**The system must support the committee, not replace it.**

So from the beginning, we treated this as an explainable decision-support system, not an autonomous admissions engine.

## 2. The First Version

The first working version was intentionally conservative.

We built a deterministic baseline that took:

- structured application fields
- motivation letter
- long-form motivation questions
- interview text
- optional transcripts

and converted them into:

- `merit_score`
- `confidence_score`
- `authenticity_risk`
- workflow recommendation for the committee

This version had some important strengths:

- it was transparent
- it was auditable
- it excluded sensitive fields from merit scoring
- it fit the human-in-the-loop requirement from the TЗ

At that point, this was exactly the right move.
We needed something simple, explainable, and safe before we introduced more complex NLP behavior.

## 3. The First Problems We Discovered

Once the baseline started running on candidate-like profiles, several problems became visible almost immediately.

The first major issue was that the system behaved too much like a document-quality scorer.

In practice this meant:

- longer, more structured, better-filled applications looked stronger than they should
- well-worded motivation sometimes got rewarded more than actual signal
- candidates with weak writing but strong underlying stories were still easy to underrate

The second problem was eligibility and completeness logic.

Early on, the system penalized missingness too mechanically.
It looked at the wrong kind of absence and over-pushed applications into conditional review.
This was bad both operationally and conceptually: a candidate should not look weak simply because one modality is missing.

The third problem was language bias.

The original heuristic layer was too lexical and too English-shaped.
That created a serious risk:

- English and mixed-language candidates looked more coherent
- Russian candidates were more likely to be treated as low-confidence or risky

This was directly in tension with the fairness expectations in the TЗ.

## 4. Reframing The Goal

This was the first major turning point.

We stopped thinking of the system as:

"a way to score application quality"

and reframed it as:

**"a way to surface hidden potential and support committee review."**

That reframing changed the logic of the whole ML / NLP layer.

Instead of asking:

- who sounds strongest?

we started asking:

- who shows agency?
- who shows repeated effort and growth?
- who is grounded in real actions?
- who may be stronger than their self-presentation suggests?

This became the conceptual foundation for the rubric.

## 5. Building The Rubric

At that point, we needed a target structure that was closer to the university's real admissions values.

So we defined a rubric built around the dimensions that matter for early talent identification:

- `leadership_potential`
- `growth_trajectory`
- `motivation_authenticity`
- `evidence_strength`
- `committee_priority`
- `hidden_potential_flag`

This was important for two reasons.

First, it gave us a better language for what the model should optimize for.

Second, it let us separate different kinds of signal:

- raw merit
- confidence / evidence sufficiency
- authenticity review risk
- support needs

That separation turned out to be one of the strongest parts of the system.

## 6. Data Reality Check

At first, it was tempting to think that more external datasets would solve the problem.

We collected external corpora for essay robustness, authenticity contrast, and privacy-safe NLP processing.
That helped the engineering side of the NLP stack.

But it did not solve the real selection problem.

Why?

Because external datasets are not admissions truth.
They do not tell us who the committee would actually want to surface.

This led to another key realization:

**We do not need one ideal candidate in semantic space.**
**We need a rubric-driven benchmark that teaches the system how to distinguish different types of promising and non-promising profiles.**

So the project moved toward:

- candidate-like synthetic profiles
- controlled counterfactual variants
- annotation layers
- final hackathon labels

That gave us a benchmark that was not perfect, but was much closer to the real decision-support task.

## 7. Annotation Under Time Pressure

The next challenge was annotation quality.

A proper gold benchmark with multiple human annotators would have been ideal.
But hackathon time is limited, and team bandwidth was limited too.

So we took a pragmatic route:

- draft labels
- adjudication proposal
- curated high-confidence subset
- final merged hackathon annotations

This was not a scientific gold dataset.
But it was good enough to create a meaningful feedback loop between rubric, scoring logic, and evaluation.

That tradeoff matters.

Instead of pretending to have perfect labels, we documented the label quality honestly and used the annotation layers for what they were good at:

- calibration
- ranking diagnostics
- shortlist evaluation
- presentation evidence

## 8. Adding A Semantic Layer

The deterministic baseline was useful, but still too brittle.

To go beyond surface heuristics, we introduced a semantic rubric layer.

Its role was simple:

- match candidate evidence to rubric concepts
- not just to literal keywords

This layer was used for:

- leadership potential
- growth trajectory
- motivation authenticity
- authenticity groundedness
- hidden potential

Even though the first semantic backend was lightweight, this was still a major architectural step.

It let the system begin reasoning in terms of rubric-aligned evidence rather than only text density or lexical overlap.

## 9. Another Hard Truth: The Model Still Wasn't Aligned

Once we finally had a merged final annotation set over the expanded candidate pool, we ran the full evaluation loop.

That exposed a difficult but useful truth:

the pipeline was working,
but the scorer still was not aligned well enough with the shortlist behavior we wanted.

The initial metrics were weak:

- low rank correlation with committee-style labels
- near-random pairwise ordering in many cases
- poor hidden-potential recall in the shortlist

This was one of the most valuable moments in the project.

Instead of hiding the result, we used it.

It told us that the bottleneck was no longer data collection.
The bottleneck was the scoring logic itself.

## 10. Calibrating The Scorer

This became the next major phase.

We recalibrated the transparent scorer against the final hackathon annotation set.

The important part is how we did it:

not by replacing the system with a black box,
but by making the transparent scorer behave more like the rubric we had defined.

The recalibration focused on several things:

- stronger weight for corroborated evidence
- stronger role of multimodal support
- less inflation from polished but weakly supported narratives
- better separation between hidden potential and pure polish
- more realistic routing thresholds for shortlist behavior

This gave us a new transparent, traceable, calibrated scoring layer.

And for the first time, the evaluation moved in the right direction in a measurable way.

## 11. The Result Of Calibration

After calibration, the system showed a meaningful improvement on the final hackathon benchmark.

The ranking metrics improved substantially:

- `spearman_merit_vs_labels: 0.0119 -> 0.3798`
- `pairwise_accuracy: 0.4949 -> 0.6710`
- `precision@k_priority: 0.5625 -> 0.7500`
- `hidden_potential_recall@k: 0.1351 -> 0.1892`

This did not magically solve everything.
But it changed the status of the system.

Before calibration, the scorer was mostly a promising prototype.

After calibration, it became something we could honestly present as:

**a measurable, explainable ranking engine aligned with the benchmark we designed for the task.**

## 12. Dealing With Synthetic Leakage Risk

Another issue appeared once the dataset included counterfactual variants.

The moment you create families like:

- original candidate
- stronger self-presentation version
- weaker self-presentation version
- multimodal variant

you also create a leakage-like validation risk.

The model can look stronger simply because very similar profiles appear multiple times.

So we added family-aware validation.

This was a strong methodological step because it showed that we were not just optimizing for optimistic candidate-level metrics.

We explicitly asked:

**If we collapse near-neighbor synthetic families, does the shortlist logic still hold?**

The answer was good enough to strengthen the story:

- root-representative validation stayed reasonably strong
- family-aggregated shortlist precision was high

That gave us a much more credible validation narrative for the judges.

## 13. Where The System Is Strong Now

At this stage, the project is strong in several ways.

First, it is aligned with the TЗ structurally:

- it is explainable
- it is not autonomous
- it supports committee review
- it avoids sensitive-field merit scoring
- it works over multimodal candidate inputs

Second, it has a real evaluation loop:

- benchmark candidates
- annotation layers
- scorer calibration
- fairness audit
- family-aware validation

Third, it has a strong product story:

**the system is designed to find hidden-potential candidates that traditional screening might miss.**

That is the right differentiation for this track.

## 14. Where The System Is Still Weak

The biggest unresolved weakness is fairness across language groups.

Even after calibration, Russian candidates still underperform relative to English and mixed profiles.

This is important.

It means we improved ranking quality and routing behavior,
but we have not fully solved multilingual fairness.

That should be acknowledged directly in the presentation.

Another limitation is that the final annotation file is still a pragmatic hackathon artifact, not a production admissions benchmark.

And finally, synthetic multimodal evidence is still cleaner and more structured than real-world application data is likely to be.

These are not reasons to hide the system.
They are reasons to present it honestly.

## 15. What The System Ultimately Became

The system started as a transparent heuristic scorer.

It evolved into:

- a rubric-driven ranking engine
- with semantic and heuristic feature extraction
- with explicit hidden-potential logic
- with explainability and committee routing
- with benchmark-based calibration
- with leakage-aware validation

That is a much stronger end state than what we had at the beginning.

And most importantly, it now reflects the mission of the task more closely.

This is no longer just:

"an AI that scores essays."

It is:

**an explainable AI support system that helps the committee identify hidden-potential applicants, understand why they surfaced, and review them more fairly and efficiently.**

## 16. Final One-Sentence Summary

If this entire project has to be reduced to one sentence, it should be this:

**We built a transparent, rubric-calibrated candidate ranking system designed to surface hidden-potential applicants that polished-writing-based screening would otherwise miss.**
