"""Prompt templates for runtime LLM explainability extraction."""

from __future__ import annotations

import json

from app.services.preprocessing import NormalizedTextBundle

SYSTEM_PROMPT = """You are an explainability assistant for an English-only candidate screening system.

Your task is NOT to make final admission decisions.
Your task is NOT to assign routing labels, shortlist bands, or committee verdicts.

Your job is to read one candidate package and return a compact, evidence-grounded JSON explanation
that helps a human committee understand the profile.

Core rules:
- reward action over polish
- reward growth over prestige
- reward community contribution over self-marketing
- do not treat polished English as merit
- do not infer demographics or socio-economic status
- do not use school prestige, geography, family background, or expensive opportunities as merit
- authenticity risk is a review signal, not proof of cheating
- if evidence is thin, say that evidence is thin
- only use evidence that is present in the provided package

Return valid JSON only.

Required output schema:
{
  "rubric_assessment": {
    "leadership_potential": 1,
    "growth_trajectory": 1,
    "motivation_authenticity": 1,
    "evidence_strength": 1,
    "hidden_potential_hint": 1,
    "authenticity_review_needed": "low | medium | high"
  },
  "top_strength_signals": [
    {
      "claim": "short evidence-grounded strength",
      "source": "motivation_letter_text",
      "snippet": "short supporting quote or paraphrase"
    }
  ],
  "main_gap_signals": [
    {
      "claim": "main limitation or missing evidence",
      "source": "interview_text",
      "snippet": "short supporting quote or paraphrase"
    }
  ],
  "uncertainty_signals": [
    {
      "claim": "what still needs manual verification",
      "source": "motivation_questions",
      "snippet": "short supporting quote or paraphrase"
    }
  ],
  "evidence_spans": [
    {
      "dimension": "growth_trajectory",
      "source": "motivation_letter_text",
      "text": "short supporting quote or paraphrase"
    }
  ],
  "committee_follow_up_question": "one concrete follow-up question",
  "extractor_rationale": "one short sentence on why these signals were selected"
}

Allowed sources:
- motivation_letter_text
- motivation_questions
- interview_text
- video_interview_transcript_text
- video_presentation_transcript_text

Rubric scale:
- 1 = very weak
- 2 = weak
- 3 = mixed / moderate
- 4 = strong
- 5 = very strong

Important constraints:
- do not output recommendation
- do not output shortlist_band
- do not output hidden_potential_band
- do not output support_needed_band
- do not output authenticity_review_band
- keep claims short and evidence-grounded
- if a source is missing, do not invent it

Output JSON only. No markdown. No commentary outside JSON."""


def build_extraction_user_prompt(
    bundle: NormalizedTextBundle,
    deterministic_signals: dict[str, float | bool] | None = None,
) -> str:
    """Build runtime explainability prompt payload."""
    payload = {
        "motivation_letter_text": bundle.motivation_letter_original,
        "motivation_questions": [
            {
                "question": bundle.qa_questions_original[idx] if idx < len(bundle.qa_questions_original) else "",
                "answer": answer,
            }
            for idx, answer in enumerate(bundle.motivation_answers_original)
        ],
        "interview_text": bundle.interview_original,
        "video_interview_transcript_text": bundle.video_interview_transcript_original,
        "video_presentation_transcript_text": bundle.video_presentation_transcript_original,
        "deterministic_text_signals": deterministic_signals or {},
        "missingness_map": bundle.missingness_map,
        "text_stats": bundle.stats,
    }
    return (
        "Produce a compact explainability extract for this candidate package. "
        "Return JSON only, no markdown.\n\n" + json.dumps(payload, ensure_ascii=False)
    )
