"""Prompt templates for runtime LLM explainability extraction."""

from __future__ import annotations

import json

from app.services.preprocessing import NormalizedTextBundle

PROMPT_VERSION = "llm-explainability-v2-evidence-first"

EXPLAINABILITY_SCHEMA = {
    "rubric_assessment": {
        "leadership_potential": 1,
        "growth_trajectory": 1,
        "motivation_authenticity": 1,
        "evidence_strength": 1,
        "hidden_potential_hint": 1,
        "authenticity_review_needed": "low | medium | high",
    },
    "top_strength_signals": [
        {
            "claim": "short evidence-grounded strength",
            "source": "motivation_letter_text",
            "snippet": "short supporting quote or paraphrase",
        }
    ],
    "main_gap_signals": [
        {
            "claim": "main limitation or missing evidence",
            "source": "interview_text",
            "snippet": "short supporting quote or paraphrase",
        }
    ],
    "uncertainty_signals": [
        {
            "claim": "what still needs manual verification",
            "source": "motivation_questions",
            "snippet": "short supporting quote or paraphrase",
        }
    ],
    "evidence_spans": [
        {
            "dimension": "growth_trajectory",
            "source": "motivation_letter_text",
            "text": "short supporting quote or paraphrase",
        }
    ],
    "committee_follow_up_question": "one concrete follow-up question",
    "extractor_rationale": "one short sentence on why these signals were selected",
}


SYSTEM_PROMPT = """You are an explainability assistant for inVision U university admissions decision support.

Your task is NOT to make final admission decisions.
Your task is NOT to assign routing labels, shortlist bands, or committee verdicts.

Your job is to read one candidate package and return a compact, evidence-grounded JSON explanation
that helps a human committee understand the profile.

Important domain context:
- these are applicants to inVision U, a university
- treat them as university applicants, not as applicants to a company, fellowship, or accelerator
- reward substance, action, reflection, contribution, and growth potential over polished self-marketing
- weak English, translated-thinking phrasing, or modest self-presentation are not authenticity problems by themselves

Core rules:
- reward action over polish
- reward growth over prestige
- reward community contribution over self-marketing
- do not treat polished English as merit
- do not infer demographics or socio-economic status
- do not use school prestige, geography, family background, or expensive opportunities as merit
- authenticity risk is a review signal, not proof of cheating
- if evidence is thin, say that evidence is thin
- explicitly look for unsupported strong claims, cross-section mismatch, and suspicious style/tone shifts
- treat section-to-section inconsistency as important only when it affects trust or groundedness, not just because one section is shorter or rougher
- if one section sounds much more polished or consultant-like than the others, mention that only as a review uncertainty, not as proof
- only use evidence that is present in the provided package
- deterministic_text_signals are hints, not evidence
- never cite deterministic_text_signals as proof
- every strength, gap, and uncertainty claim should be anchored in candidate text
- avoid generic praise such as "shows leadership potential" unless the claim is tied to a concrete action or example
- if a point is not grounded in the package, omit it

Return valid JSON only.

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
- do not wrap the response inside {"answer": "..."} or {"response": "..."}
- output the schema keys directly at the top level
- committee_follow_up_question must target the single most important remaining uncertainty
- committee_follow_up_question must be concrete, interview-usable, and not compound
- prefer "insufficient evidence for X" over speculative psychological interpretation

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
        "institution_context": {
            "institution_name": "inVision U",
            "institution_type": "university",
            "task_type": "committee explainability support",
        },
    }
    return (
        "Produce a compact explainability extract for this candidate package.\n"
        "Work evidence-first.\n"
        "Use deterministic_text_signals only as hints for where to look more carefully.\n"
        "Do not treat deterministic_text_signals as evidence and do not mention them in snippets.\n"
        "Good claim style: concrete action, grounded in one visible source.\n"
        "Bad claim style: generic verdict without text support.\n"
        "When relevant, use `main_gap_signals` or `uncertainty_signals` to capture: unsupported strong claims, sections that do not align, or tone/style shifts across sections.\n"
        "If the candidate sounds rough, translated-in-thinking, or modest, do not mistake that for weak merit or authenticity problems by itself.\n"
        "Return a single JSON object with these exact top-level keys:\n"
        f"{json.dumps(EXPLAINABILITY_SCHEMA, ensure_ascii=False)}\n"
        "Do not rename keys. Do not wrap the result inside an `answer` field.\n"
        "If evidence is thin, say so in `main_gap_signals` or `uncertainty_signals`.\n"
        "For `committee_follow_up_question`, ask one focused question that would genuinely help a committee verify the most important uncertainty.\n"
        "Return JSON only, no markdown.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


REPAIR_SYSTEM_PROMPT = """You are a JSON repair assistant.

Your task is to convert a previous model answer into the exact explainability schema required by the runtime.

Rules:
- return JSON only
- keep the exact top-level keys from the required schema
- do not wrap the result in {"answer": "..."}
- use only evidence from the provided candidate package or previous answer
- if a signal is missing, use an empty list or empty string instead of prose outside the schema
- rubric scores must be integers 1..5
- authenticity_review_needed must be one of: low, medium, high
- preserve evidence-grounded wording
- remove generic verdicts that are not grounded in the candidate package
"""


def build_repair_user_prompt(
    bundle: NormalizedTextBundle,
    prior_response: str,
    deterministic_signals: dict[str, float | bool] | None = None,
) -> str:
    """Build a repair prompt when the first model pass ignores the target schema."""
    payload = {
        "required_schema": EXPLAINABILITY_SCHEMA,
        "previous_model_answer": prior_response,
        "candidate_package": {
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
        },
    }
    return "Repair the previous answer into the exact required schema. Return JSON only.\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
    )
