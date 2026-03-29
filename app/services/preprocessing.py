"""Text preprocessing and normalization for heuristic scoring."""

from __future__ import annotations

from dataclasses import dataclass

from app.utils.math_utils import safe_div
from app.utils.text import char_count, maybe_text, sentence_count, split_sections, to_lower, word_count


@dataclass(slots=True)
class NormalizedTextBundle:
    """Canonical text bundle used across downstream extractors."""

    motivation_letter_original: str
    motivation_letter_lower: str
    interview_original: str
    interview_lower: str
    motivation_answers_original: list[str]
    motivation_answers_lower: list[str]
    qa_questions_original: list[str]
    full_text_original: str
    full_text_lower: str
    sections: dict[str, list[str]]
    missingness_map: dict[str, bool]
    stats: dict[str, float | int]


def preprocess_text_inputs(text_inputs: dict[str, object]) -> NormalizedTextBundle:
    """Normalize text inputs while preserving originals for evidence spans."""
    motivation_letter = maybe_text(text_inputs.get("motivation_letter_text"))
    interview_text = maybe_text(text_inputs.get("interview_text"))

    raw_qas = text_inputs.get("motivation_questions") or []
    qa_answers: list[str] = []
    qa_questions: list[str] = []
    for qa in raw_qas:
        if not isinstance(qa, dict):
            continue
        qa_questions.append(maybe_text(qa.get("question")))
        qa_answers.append(maybe_text(qa.get("answer")))

    non_empty_answers = [a for a in qa_answers if a]
    full_text_parts = [motivation_letter, interview_text, *non_empty_answers]
    full_text = "\n\n".join(part for part in full_text_parts if part)

    answer_word_counts = [word_count(a) for a in non_empty_answers]

    sections = {
        "motivation_letter_text": split_sections(motivation_letter),
        "motivation_questions": non_empty_answers,
        "interview_text": split_sections(interview_text),
    }

    missingness_map = {
        "motivation_letter_text": not bool(motivation_letter),
        "interview_text": not bool(interview_text),
        "motivation_questions": len(non_empty_answers) == 0,
    }

    stats = {
        "word_count": word_count(full_text),
        "char_count": char_count(full_text),
        "sentence_count": sentence_count(full_text),
        "avg_answer_length_words": safe_div(sum(answer_word_counts), len(answer_word_counts), default=0.0),
        "non_empty_answer_count": len(non_empty_answers),
        "non_empty_text_sources": sum(
            [
                int(bool(motivation_letter)),
                int(bool(interview_text)),
                int(len(non_empty_answers) > 0),
            ]
        ),
    }

    return NormalizedTextBundle(
        motivation_letter_original=motivation_letter,
        motivation_letter_lower=to_lower(motivation_letter),
        interview_original=interview_text,
        interview_lower=to_lower(interview_text),
        motivation_answers_original=non_empty_answers,
        motivation_answers_lower=[to_lower(a) for a in non_empty_answers],
        qa_questions_original=qa_questions,
        full_text_original=full_text,
        full_text_lower=to_lower(full_text),
        sections=sections,
        missingness_map=missingness_map,
        stats=stats,
    )
