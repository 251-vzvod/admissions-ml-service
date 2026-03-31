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

Very quickly, however, another nuance became clear.

Adding a semantic layer is not the same thing as proving semantic improvement.

So we treated the semantic backend itself as an experimental component.
We refactored it into a configurable layer with multiple backend options:

- `hash`
- `tfidf_char_ngram`
- `sentence_transformer`

This was not just an engineering cleanup.
It was a methodological decision:

**if multilingual fairness is one of the biggest open problems, then semantic matching has to become something we can test honestly, not something we hardcode once and assume is good enough.**

## 9. Semantic Backend Experiment: Useful Infrastructure, No Metric Win Yet

Once the configurable semantic layer existed, we ran the first stronger backend experiment using `tfidf_char_ngram`.

The hypothesis was reasonable:

- character-level multilingual matching might reduce English-shaped lexical dependence
- semantic rubric alignment might improve
- Russian fairness might move in the right direction

The result was important precisely because it was not a clean success.

The experiment produced a useful platform improvement, but not a clear evaluation win.

Compared with the calibrated baseline setup, `tfidf_char_ngram`:

- did not improve overall shortlist alignment enough
- did not improve hidden-potential recall meaningfully
- did not improve Russian fairness enough to justify becoming the default backend

That led to a disciplined decision:

- keep the backend abstraction
- keep `tfidf_char_ngram` and `sentence_transformer` as switchable experimental modes
- return the default backend to `hash`

This was one of the more mature decisions in the project.

We did not pretend an experiment worked just because it was technically more advanced.
We kept the useful infrastructure and rejected the inflated claim.

## 10. Another Hard Truth: The Model Still Wasn't Aligned

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

## 11. Calibrating The Scorer

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

## 12. The Result Of Calibration

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

## 13. A Failed Shortcut: Naive Hidden-Potential Uplift

Once calibration worked, it was tempting to push the system even harder toward the main product thesis:

**surface hidden-potential candidates more aggressively.**

So we tried a more explicit uplift logic based on the gap between:

- underlying signal
- and presentation quality

In theory, this sounded exactly right.
If someone has strong growth, real effort, and real evidence, but presents themselves weakly, the system should pull them upward.

But in practice, the first implementation was too broad.

It did not just boost the candidates we wanted.
It also boosted candidates who were already strong, and in some cases it amplified the wrong profiles.

The measurable outcome was clear:

- overall pairwise ranking quality dropped
- hidden-potential recall did not move enough
- fairness did not improve in a convincing way

So we rolled that change back.

This was another useful moment in the project.

It reminded us that:

**product intuition is not enough; every uplift has to survive evaluation.**

That rollback made the system better, not worse, because it protected the integrity of the benchmark story.

## 14. Dealing With Synthetic Leakage Risk

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

## 15. Moving From Model Transparency To Committee Actionability

By this point, the numeric scoring layer had become much stronger.

But there was still one product gap:

the output explained the score,
yet it still did not fully explain what the committee should do next.

This led to the next layer of improvement:

- `committee_cohorts`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`

This changed the product from:

"a system that returns a score and some explanation"

into:

**a system that returns a review-ready candidate card.**

This was important because it pushed explainability one step further.

Now the output was not just technically interpretable.
It became operationally useful for a real admissions workflow.

That is also much closer to what the TЗ actually asks for:

- transparency
- human-in-the-loop
- decision support
- reduced manual load

## 16. Adding AI Detection Without Turning It Into A False Verdict

One of the tensions in the project was obvious:

the problem statement explicitly mentions generative AI,
but most "AI detector" stories become misleading very quickly.

If we simply attached one probability like:

"this text is 87% AI-generated"

we would create a dangerous illusion of certainty.

So when we decided to add a detector-based signal, we did it under strict constraints.

We integrated an auxiliary detector based on `desklib/ai-text-detector-v1.01`, but only as:

- an English-first signal
- a weak review feature
- a component of `authenticity_risk`
- a source of review reasons

and explicitly **not** as:

- proof of cheating
- proof of dishonesty
- a merit penalty
- an auto-rejection trigger

This design choice mattered a lot.

It let us say:

**yes, we address the GenAI problem from the TЗ**

without claiming something we cannot honestly prove.

The detector now helps answer the right operational question:

"Should the committee read this profile more carefully for groundedness and consistency?"

instead of the wrong question:

"Did the system prove that this candidate used AI?"

## 17. Where The System Is Strong Now

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

And now it also has a much stronger committee-facing interface contract:

- score
- supported claims vs weakly supported claims
- evidence
- review routing
- cohorts
- manual verification prompts
- follow-up interview question

That combination is much more compelling than a plain ranking output.

## 18. Where The System Is Still Weak

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

Another unresolved point is that the semantic layer is still in an intermediate state.

The backend abstraction is now ready for stronger multilingual encoders,
but the default production-like mode still relies on the lightweight hash backend because the first stronger experiment did not yet justify a switch.

That should be framed as:

- not a failure of direction
- but an honest intermediate engineering state

Another limitation is that the new AI detector layer is still only a cautious auxiliary signal.

That is deliberate.
It is a better product decision for this track than pretending to have perfect AI-use detection.

Another fairness limitation is that multilingual mitigation is still incremental, not complete.

We now do three things that help:

- a lightweight RU/EN semantic bridge
- section-consistency checks instead of essay-polish-only trust
- a small targeted mitigation that reduces style-based penalties for coherent Cyrillic-heavy profiles

But that is still a mitigation layer, not proof that fairness is solved.

At the same time, the product focus evolved.

Once we accepted that the real product will mostly receive English applications, the fairness question had to become more specific.

The more relevant question is no longer only:

- "Are Russian candidates disadvantaged?"

It is also:

- "Are polished English applicants still being overrewarded relative to plainer but credible English applicants?"

That is why we added an English-first slice validation layer over:

- polished vs plain
- verbose vs concise
- evidence-strong vs evidence-thin
- transcript-present vs transcript-absent

This was a useful methodological shift.
It gave us a more realistic fairness and validation story for the actual product direction.

We then pushed that logic one step further with perturbation-based stress tests.

Instead of only evaluating fixed profiles, we asked:

- what happens if the same candidate becomes shorter?
- what happens if we remove evidence?
- what happens if we add a polished wrapper?
- what happens if transcripts disappear?

That gave us a much more honest view of robustness.

The results were mixed in a useful way:

- the system does not seem overly vulnerable to generic polish added on top
- it reacts strongly when evidence is removed, which is correct
- transcript removal is not catastrophic on average
- but concise variants still get punished too hard

That is a real product weakness, and exactly the kind of weakness that should be discovered before a demo, not after.

We also tested a more ambitious next step: an offline learned pairwise ranker on transparent features.

This experiment mattered because one of the standing concerns about the system is that it still contains a lot of heuristic backbone.

So we asked a direct question:

**Can a learned pairwise layer over transparent features outperform the current shortlist logic on a family-aware split?**

The answer was not cleanly positive.

The learned ranker improved one correlation metric, but it worsened pairwise ranking accuracy materially and did not improve the most important shortlist outcomes enough to justify replacing the current runtime logic.

That was still a valuable result.

It means the project now has:

- not just a better scorer
- but also a clearer sense of which next-step model ideas are actually worth keeping

## 19. What The System Ultimately Became

The system started as a transparent heuristic scorer.

It evolved into:

- a rubric-driven ranking engine
- with semantic and heuristic feature extraction
- with explicit hidden-potential logic
- with explainability and committee routing
- with benchmark-based calibration
- with leakage-aware validation
- with English-first shortlist slice validation
- with English-first stress-test robustness checks
- with an offline learned-ranker experiment that was tested honestly and not promoted prematurely

That is a much stronger end state than what we had at the beginning.

And most importantly, it now reflects the mission of the task more closely.

This is no longer just:

"an AI that scores essays."

It is:

**an explainable AI support system that helps the committee identify hidden-potential applicants, understand why they surfaced, and review them more fairly and efficiently.**

## 20. Final One-Sentence Summary

If this entire project has to be reduced to one sentence, it should be this:

**We built a transparent, rubric-calibrated candidate ranking system designed to surface hidden-potential applicants that polished-writing-based screening would otherwise miss.**
