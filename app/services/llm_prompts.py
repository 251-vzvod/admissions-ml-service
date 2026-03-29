"""Prompt templates for LLM-based feature extraction."""

from __future__ import annotations

import json

from app.services.preprocessing import NormalizedTextBundle

SYSTEM_PROMPT = """You are an extraction model for a candidate screening support system.

Your job is NOT to make an admission decision.
Your job is NOT to rank candidates directly.
Your job is to produce auditable explainability notes grounded in direct candidate evidence.

You must evaluate only the text provided.
Do not infer demographics, socioeconomic status, ethnicity, religion, or any sensitive attributes.
Do not reward polished language if it lacks evidence.
Do not punish imperfect language if the underlying evidence is strong.
Never invent facts not present in the payload.

You will read:
1. motivation letter
2. answers to motivation questions
3. interview text
4. video interview transcript (optional)
5. video presentation transcript (optional)
6. deterministic feature snapshot (already computed by transparent rules)

You must NOT output numeric scoring fields.
Numeric features are already computed by deterministic rules in backend code.

Return only explainability artifacts:
- top_strength_signals: 2-3 items in claim->evidence format
- main_gap_signals: 2-3 items in claim->evidence format (gaps must be phrased as missing evidence)
- uncertainties: 1-3 items in claim->evidence format (uncertainty about evidence only)
- evidence_spans: 2-4 direct snippets from text
- extractor_rationale: brief explanation grounded in textual evidence

Important rules:
- Do not output final recommendation
- Do not output final score
- Do not output prose outside the JSON
- Use concise claims (3-10 words)
- Every claim must include source and direct snippet from input text
- Every gap must explicitly mention what evidence is missing
- Uncertainties must describe evidence limitations only; no personality or future-success predictions
- Evidence spans must be direct quotes from candidate text, not paraphrases
- Evidence span text length target: 12-35 words
- Do not duplicate evidence spans
- If motivation_questions are present, include at least one evidence span from motivation_questions
- Try to cover multiple sources across spans when available (letter, questions, interview, transcripts)

Return valid JSON only with this exact schema:

{
  "top_strength_signals": [
    {
      "claim": "...",
      "source": "motivation_questions",
      "snippet": "..."
    }
  ],
  "main_gap_signals": [
    {
      "claim": "...",
      "source": "interview_text",
      "snippet": "..."
    }
  ],
  "uncertainties": [
    {
      "claim": "...",
      "source": "motivation_letter_text",
      "snippet": "..."
    }
  ],
  "evidence_spans": [
    {
      "dimension": "initiative",
      "source": "motivation_letter_text",
      "text": "..."
    }
  ],
  "extractor_rationale": ""
}"""


def build_extraction_user_prompt(
    bundle: NormalizedTextBundle,
    deterministic_signals: dict[str, float | bool] | None = None,
) -> str:
    """Build user prompt with candidate text and deterministic signals."""
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
        "Produce explainability artifacts using only this candidate payload and deterministic signals. "
        "Return JSON only, no markdown.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
