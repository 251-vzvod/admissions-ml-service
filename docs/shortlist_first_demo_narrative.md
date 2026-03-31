# Shortlist-First Demo Narrative

## Core Demo Message

The demo should not present the system as:

- an essay scorer
- an AI detector
- an autonomous admissions engine

The demo should present the system as:

- a committee support tool
- a shortlist builder
- a hidden-potential surfacing layer

## One-Line Positioning

We do not rank who wrote the most polished application.
We help the committee surface early-stage leaders with real growth signals, even when their writing is imperfect.

## Demo Structure

The strongest demo flow is:

1. show the shortlist view
2. show one hidden-potential candidate
3. show one polished-but-low-evidence candidate
4. show one promising-but-needs-support candidate
5. show the committee guidance fields

This keeps the demo aligned with the admissions workflow.

## Screen 1. Batch Shortlist

Start from `POST /score/batch`.

What to highlight:

- `ranked_candidate_ids`
- `shortlist_candidate_ids`
- `hidden_potential_candidate_ids`
- `support_needed_candidate_ids`
- `authenticity_review_candidate_ids`

Main message:

The system is already useful before anyone opens a single candidate card.
It helps the committee decide who to read first and why.

## Screen 2. Hidden Potential Case

Choose a candidate whose self-presentation is imperfect but whose trajectory and leadership signals are real.

What to highlight:

- `hidden_potential_score`
- `trajectory_score`
- `why_candidate_surfaced`
- `committee_cohorts`
- `what_to_verify_manually`

Main message:

This candidate would be easy to underrate in a polish-first review process.
Our system pushes them into committee attention because the underlying signal is stronger than the writing quality.

## Screen 3. Polished But Low-Evidence Case

Choose a candidate with smoother writing but weaker grounding.

What to highlight:

- lower `evidence_coverage_score`
- `authenticity_risk`
- `authenticity_review_reasons`
- `committee_cohorts`

Main message:

The system does not confuse polished language with strong potential.
It routes this case for closer review instead of over-rewarding style.

## Screen 4. Promising But Needs Support

Choose a candidate with real signal and likely need for onboarding or academic support.

What to highlight:

- `support_needed_score`
- `committee_cohorts`
- `suggested_follow_up_question`
- `what_to_verify_manually`

Main message:

This is important for mission-fit.
The system does not flatten such candidates into слабый / сильный.
It helps the committee see promising candidates who may need support to unlock performance.

## Screen 5. Reviewer Guidance

For one candidate card, highlight the fields that make the workflow practical:

- `top_strengths`
- `main_gaps`
- `uncertainties`
- `evidence_spans`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`

Main message:

The output is not just a score.
It is a review-ready card that saves committee time and improves first-pass quality.

## What To Say Explicitly

Say:

- the system is deterministic at scoring time
- LLM is optional and only used for explanation
- final decision always remains with the committee
- authenticity is a review signal, not an accusation

Do not say:

- the model decides who gets in
- the AI detector proves cheating
- the system objectively knows the best candidate

## Best Narrative Arc

1. Manual review does not scale.
2. Polish-first screening misses hidden potential.
3. We built a transparent ranking and shortlist engine.
4. It separates:
   - merit
   - confidence
   - authenticity risk
5. It adds:
   - hidden potential
   - support needed
   - shortlist priority
   - trajectory
6. It gives committee-ready guidance, not just a score.

## Final Closing Line

The value of the system is not that it automates admissions.
The value is that it helps the committee notice the right candidates earlier, with clearer reasoning and less bias toward polish.
