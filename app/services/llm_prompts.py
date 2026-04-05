"""Prompt templates for runtime LLM explainability extraction."""

from __future__ import annotations

import json

from app.services.preprocessing import NormalizedTextBundle

PROMPT_VERSION = "llm-explainability-v3-invision-agenda"

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

COMMITTEE_WRITER_SCHEMA = {
    "summary": "plain-language committee summary",
    "committee_cohorts": ["short committee-facing labels"],
    "why_candidate_surfaced": ["decision-relevant reasons this candidate surfaced"],
    "top_strengths": ["short reviewer-facing strengths"],
    "main_gaps": ["short reviewer-facing gaps"],
    "uncertainties": ["short remaining uncertainties or open questions"],
    "what_to_verify_manually": ["short reviewer-facing verification priorities"],
    "suggested_follow_up_question": "one concrete follow-up question",
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
- inVision U is trying to identify future leaders who are willing to give more than they take
- inVision U values trajectory, reflection, service orientation, mission fit, and practical problem-solving
- inVision U is not looking for passive prestige-seekers or applicants who only want credentials
- false positives matter: avoid overstating top-priority strength when the evidence is mixed

Core rules:
- reward action over polish
- reward growth over prestige
- reward community contribution over self-marketing
- reward leadership-through-service over status, titles, or self-branding
- reward evidence of improving something for other people, not only personal ambition
- treat interview and Q&A evidence as more reliable than polished essay phrasing when they conflict
- treat IELTS / TOEFL / UNT style readiness only as context, not as the sole signal of merit
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
- if motivation sounds prestige-driven, parent-driven, backup-option-driven, or externally imposed, surface that as a gap or uncertainty
- if motivation appears specific to inVision U's mission, practice-based learning, and contribution to society, surface that clearly
- if the profile is low-polish but high-substance, do not understate it just because the writing is rough
- prefer "not enough evidence to confirm X" over "candidate lacks X" when the real issue is uncertainty

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
            "mission": "develop leaders who create positive impact for society and are willing to give back to their communities and region",
            "leadership_definition": "leadership means willingness to give more than take, take responsibility, and improve real conditions for other people",
            "selection_priorities": [
                "trajectory over polish",
                "service orientation over self-promotion",
                "interview-confirmed motivation over polished essay language",
                "practical problem-solving and reflection over prestige",
                "do not lose strong candidates who present themselves modestly"
            ],
            "review_risk_posture": "false positives at the top of the list are costly; do not overstate top-priority strength when evidence is mixed",
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
        "For motivation, prioritize interview and Q&A over polished essay wording when they diverge.\n"
        "For leadership, prioritize responsibility, contribution, and problem-solving for others over titles or prestige.\n"
        "If the candidate seems mainly driven by prestige, credentials, parental pressure, or a backup plan, surface that as a gap or uncertainty instead of praising motivation.\n"
        "If the candidate clearly aligns with inVision U's mission, project-based learning, and social contribution, surface that explicitly.\n"
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


COMMITTEE_WRITER_SYSTEM_PROMPT = """You are a committee-writing assistant for inVision U admissions support.

You do not assign scores.
You do not assign recommendation labels.
You do not make final decisions.

Your job is to rewrite structured candidate findings into clear reviewer-facing language.

Rules:
- write in plain committee language, not model or pipeline jargon
- be concise by default
- if the case is straightforward, keep the summary short
- if the case is high-uncertainty or manual-review heavy, write a slightly fuller summary
- emphasize the most decision-relevant facts first
- prefer interview and Q&A evidence over polished essay phrasing when there is tension
- reflect inVision U's mission: future leaders should create positive impact and give more than they take
- treat leadership as responsibility, contribution, and action for others, not just confidence or titles
- treat trajectory and reflection as important, especially for candidates with imperfect polish
- be careful with top-priority language when evidence is mixed; avoid overselling
- do not invent evidence
- do not mention internal feature names, model names, or hidden heuristics
- do not restate every score unless it helps clarify uncertainty
- authenticity risk is a review signal, not proof
- avoid generic praise
- every strength or gap should be meaningfully grounded in the provided evidence summary
- use committee_cohorts only when they are actually justified by the evidence package
- why_candidate_surfaced should explain why this candidate deserves attention, not repeat generic traits
- uncertainties should capture the 1-3 main unresolved questions that affect confidence or prioritization
- if the application looks prestige-driven, parent-driven, backup-option-driven, or weakly aligned with inVision U's mission, surface that clearly
- suggested_follow_up_question must be one concrete interview-usable question

Return valid JSON only using the exact top-level keys in the required schema."""


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


def build_committee_writer_user_prompt(
    *,
    detail_level: str,
    candidate_id: str,
    recommendation: str,
    merit_score: int,
    confidence_score: int,
    authenticity_risk: int,
    review_flags: list[str],
    committee_cohorts: list[str],
    why_candidate_surfaced: list[str],
    what_to_verify_manually: list[str],
    suggested_follow_up_question: str,
    supported_claims: list[dict[str, str]],
    weakly_supported_claims: list[dict[str, str]],
    top_strengths: list[str],
    main_gaps: list[str],
    uncertainties: list[str],
    authenticity_review_reasons: list[str],
    semantic_rubric_scores: dict[str, int],
    review_signals: dict[str, float],
    policy_bands: dict[str, bool],
    evidence_highlights: list[dict[str, str]],
    bundle: NormalizedTextBundle,
) -> str:
    payload = {
        "required_schema": COMMITTEE_WRITER_SCHEMA,
        "detail_level": detail_level,
        "candidate_id": candidate_id,
        "committee_playbook": {
            "mission": "identify future leaders who create positive social impact and give more than they take",
            "top_priority_doctrine": [
                "trajectory over polish",
                "service orientation over self-promotion",
                "interview-confirmed intent over polished essay language",
                "practical action and reflection over titles and prestige"
            ],
            "negative_patterns_to_surface": [
                "prestige-only motivation",
                "network-or-credential-only motivation",
                "parent-driven application",
                "backup-option motivation",
                "weak contribution-to-others framing"
            ],
            "risk_posture": "false positives are costly for top-priority surfacing; if evidence is mixed, prefer caution and verification over praise",
        },
        "current_outputs": {
            "recommendation": recommendation,
            "merit_score": merit_score,
            "confidence_score": confidence_score,
            "authenticity_risk": authenticity_risk,
            "review_flags": review_flags,
            "committee_cohorts": committee_cohorts,
            "why_candidate_surfaced": why_candidate_surfaced,
            "what_to_verify_manually": what_to_verify_manually,
            "suggested_follow_up_question": suggested_follow_up_question,
            "supported_claims": supported_claims,
            "weakly_supported_claims": weakly_supported_claims,
            "top_strengths": top_strengths,
            "main_gaps": main_gaps,
            "uncertainties": uncertainties,
            "authenticity_review_reasons": authenticity_review_reasons,
            "semantic_rubric_scores": semantic_rubric_scores,
            "review_signals": review_signals,
            "policy_bands": policy_bands,
            "evidence_highlights": evidence_highlights,
        },
        "source_priority": [
            "interview_text",
            "video_interview_transcript_text",
            "motivation_questions",
            "motivation_letter_text",
            "video_presentation_transcript_text",
        ],
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
        },
    }
    return (
        "Rewrite the current candidate-facing outputs into cleaner committee-facing language.\n"
        "If detail_level is `brief`, write 2-3 sentences max in `summary`.\n"
        "If detail_level is `standard`, write one compact paragraph.\n"
        "If detail_level is `detailed`, write one fuller but still compact paragraph focused on the main decision logic and the main uncertainty.\n"
        "Use `committee_cohorts` sparingly: keep only labels that a reviewer would actually find useful.\n"
        "Use `why_candidate_surfaced` to say why the candidate deserves attention now, not to restate every score.\n"
        "Keep `top_strengths`, `main_gaps`, and `what_to_verify_manually` short, readable, and useful for reviewers.\n"
        "Use `uncertainties` for remaining review-critical unknowns.\n"
        "Use `supported_claims`, `weakly_supported_claims`, and `evidence_highlights` as the strongest available evidence anchors.\n"
        "Use `policy_bands` and `review_signals` only to prioritize what matters; do not repeat them verbatim.\n"
        "Use committee language that reflects inVision U values: contribution, responsibility, trajectory, and mission fit.\n"
        "If evidence is mixed, do not oversell the candidate; prefer a precise caution over generic praise.\n"
        "Return JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
