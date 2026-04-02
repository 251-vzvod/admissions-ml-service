from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.input import CandidateInput

DATA_ROOT = ROOT / "data" / "ml_workbench"
LABELS_CSV = DATA_ROOT / "labels" / "human_labels_individual_llm_v2.csv"
SUMMARY_MD = DATA_ROOT / "labels" / "generated_batches_refresh_summary.md"

NEW_BATCHES = [
    {
        "name": "messy_batch_v4",
        "prefix": "syn_messy_v4_",
        "review_round": "v2_messy_batch_v4",
        "raw": DATA_ROOT / "raw" / "generated" / "messy_batch_v4" / "messy_batch_v4_api_input.jsonl",
        "san_dir": DATA_ROOT / "processed" / "annotation_packs" / "messy_batch_v4",
    },
    {
        "name": "messy_batch_v5",
        "prefix": "syn_messy_v5_",
        "review_round": "v2_messy_batch_v5",
        "raw": DATA_ROOT / "raw" / "generated" / "messy_batch_v5" / "messy_batch_v5_api_input.jsonl",
        "san_dir": DATA_ROOT / "processed" / "annotation_packs" / "messy_batch_v5",
    },
    {
        "name": "messy_batch_v5_extension",
        "prefix": "syn_messy_v5_",
        "review_round": "v2_messy_batch_v5_extension",
        "raw": DATA_ROOT / "raw" / "generated" / "messy_batch_v5_extension" / "messy_batch_v5_extension_api_input.jsonl",
        "san_dir": DATA_ROOT / "processed" / "annotation_packs" / "messy_batch_v5_extension",
    },
    {
        "name": "ordinary_batch_v6",
        "prefix": "syn_ord_v6_",
        "review_round": "v2_ordinary_batch_v6",
        "raw": DATA_ROOT / "raw" / "generated" / "ordinary_batch_v6" / "ordinary_batch_v6_api_input.jsonl",
        "san_dir": DATA_ROOT / "processed" / "annotation_packs" / "ordinary_batch_v6",
    },
    {
        "name": "gap_fill_batch_v7",
        "prefix": "syn_gap_v7_",
        "review_round": "v2_gap_fill_batch_v7",
        "raw": DATA_ROOT / "raw" / "generated" / "gap_fill_batch_v7" / "gap_fill_batch_v7_api_input.jsonl",
        "san_dir": DATA_ROOT / "processed" / "annotation_packs" / "gap_fill_batch_v7",
        "manifest": DATA_ROOT / "raw" / "generated" / "gap_fill_batch_v7" / "gap_fill_batch_v7_generation_manifest.jsonl",
    },
]

SUBJECTS = [
    "computer science",
    "data science",
    "public health",
    "environmental science",
    "renewable energy",
    "robotics",
    "mathematics",
    "economics",
    "psychology",
    "chemistry",
    "history",
    "law",
    "agriculture",
    "engineering",
    "biology",
    "education",
]

ISSUES = [
    "air quality",
    "pollution",
    "water",
    "waste",
    "recycling",
    "healthcare",
    "internet",
    "education quality",
    "language preservation",
    "agricultural modernization",
    "employment",
    "migration",
]

ACTION_KEYWORDS = [
    "organized",
    "started",
    "created",
    "built",
    "founded",
    "led",
    "tutored",
    "tutor",
    "volunteer",
    "volunteered",
    "helped",
    "mentored",
    "study group",
    "club",
    "project",
    "cleanup",
    "prototype",
    "website",
    "platform",
    "robotics",
    "neighbors",
    "classmates",
    "younger students",
]

GENERIC_PHRASES = [
    "university is important for my future",
    "i want to study at invision u",
    "i will work hard",
    "i want to learn more",
    "education will help me",
    "i think it is useful and also interesting",
    "i am highly motivated",
    "mission of excellence",
    "academic excellence",
    "communication skills and collaborative mindset",
]

LOW_POLISH_MARKERS = [
    "this is hard question for me",
    "i am still figuring out exact words",
    "i explain myself better in conversation",
    "i am a bit nervous",
    "my english is not perfect",
    "i am not the most polished applicant",
    "i kept deleting my first sentences",
    "this letter will sound simple",
]

AUTHENTICITY_MARKERS = [
    "not exactly sure",
    "maybe 9th or maybe 10th",
    "i cannot promise i will not change my mind again",
    "more settled on paper than",
    "my letter can sound like i already chose",
    "path may sound slightly split",
    "less composed than in writing",
    "my timeline is honest but not perfectly sharp",
    "i am aware that my writing can make it sound larger than it was",
    "not the same as building a real platform from scratch",
    "admissions committee",
    "committee",
]

BOILERPLATE_REPLACEMENTS = {
    "Dear Admissions Committee,\n\n": "",
    "Dear inVision U Admissions Committee,\n\n": "",
    "Respected Members of the Admissions Committee,\n\n": "",
    "Respected members of the admissions committee,\n\n": "",
    "Thank you for considering my application.": "",
    "Please consider my application.": "I hope this application shows where I am honestly right now.",
    "I would be honored to join the inVision U community and contribute to its mission of excellence.": "I want to join inVision U because I think the environment would help me grow and contribute in a more real way.",
    "In the end, I hope the admissions committee will see that": "In the end, I hope the people reading this can see that",
}

SQUARE_PLACEHOLDER_RE = re.compile(r"\[[^\]]{1,24}\]")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
THREE_PLUS_NL_RE = re.compile(r"\n{3,}")


@dataclass
class Annotation:
    recommendation: str
    committee_priority: int
    shortlist_band: bool
    hidden_potential_band: bool
    support_needed_band: bool
    authenticity_review_band: bool
    reviewer_confidence: int
    notes: str


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": record["candidate_id"],
        "structured_data": record["structured_data"],
        "text_inputs": record["text_inputs"],
        "behavioral_signals": record["behavioral_signals"],
    }


def normalize_behavioral_signals(signals: dict[str, Any] | None) -> dict[str, Any]:
    signals = signals or {}
    return {
        "completion_rate": signals.get("completion_rate"),
        "returned_to_edit": signals.get("returned_to_edit"),
        "skipped_optional_questions": signals.get("skipped_optional_questions"),
    }


def write_sanitized_pack(batch_name: str, san_dir: Path, raw_records: list[dict[str, Any]]) -> None:
    san_dir.mkdir(parents=True, exist_ok=True)
    sanitized = [sanitize_record(record) for record in raw_records]
    base = san_dir / f"{batch_name}_annotation_pack"
    write_jsonl(base.with_suffix(".jsonl"), sanitized)
    base.with_suffix(".json").write_text(json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with base.with_name(f"{batch_name}_annotation_pack_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "english_proficiency_type",
                "english_proficiency_score",
                "school_cert_type",
                "school_cert_score",
                "motivation_letter_length",
                "num_questions",
                "has_interview",
                "completion_rate",
                "returned_to_edit",
                "skipped_optional",
            ],
        )
        writer.writeheader()
        for record in sanitized:
            education = record["structured_data"]["education"]
            text_inputs = record["text_inputs"]
            writer.writerow(
                {
                    "candidate_id": record["candidate_id"],
                    "english_proficiency_type": education.get("english_proficiency", {}).get("type"),
                    "english_proficiency_score": education.get("english_proficiency", {}).get("score"),
                    "school_cert_type": education.get("school_certificate", {}).get("type"),
                    "school_cert_score": education.get("school_certificate", {}).get("score"),
                    "motivation_letter_length": len(text_inputs.get("motivation_letter_text") or ""),
                    "num_questions": len(text_inputs.get("motivation_questions") or []),
                    "has_interview": int(bool((text_inputs.get("interview_text") or "").strip())),
                    "completion_rate": record["behavioral_signals"].get("completion_rate"),
                    "returned_to_edit": int(bool(record["behavioral_signals"].get("returned_to_edit"))),
                    "skipped_optional": record["behavioral_signals"].get("skipped_optional_questions"),
                }
            )


def cleanup_visible_text(text: str, candidate_id: str) -> str:
    if not text:
        return text

    cleaned = text
    for before, after in BOILERPLATE_REPLACEMENTS.items():
        cleaned = cleaned.replace(before, after)

    cleaned = SQUARE_PLACEHOLDER_RE.sub("", cleaned)
    cleaned = cleaned.replace("Dear Admissions Committee,", "").replace("Dear inVision U Admissions Committee,", "")
    cleaned = cleaned.replace("Respected Members of the Admissions Committee,", "")
    cleaned = cleaned.replace("Respected members of the admissions committee,", "")
    cleaned = cleaned.replace("Please consider my application.", "I hope this application shows where I am honestly right now.")
    cleaned = cleaned.replace("the admissions committee", "the people reading this")
    cleaned = cleaned.replace("The admissions committee", "The people reading this")

    variants = [
        "I know this application is uneven, but it is honest.",
        "I do not want to end with a formal line, only to say that this application is sincere.",
        "I am applying because I want a real chance to learn, not because I have a perfect story.",
        "That is where I am right now, and why I decided to apply.",
        "I hope the application reads as honest even if it is not polished.",
        "That is the simplest reason I decided to apply this year.",
    ]
    if candidate_id.startswith("syn_messy_v5_"):
        replacement = variants[sum(ord(ch) for ch in candidate_id) % len(variants)]
        cleaned = cleaned.replace("  ", " ")
        if cleaned.endswith("\n"):
            cleaned = cleaned.rstrip("\n")
        if "Thank you for considering my application." not in text and replacement not in cleaned:
            pass
        else:
            cleaned = cleaned.replace("Thank you for considering my application.", replacement)

    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned)
    cleaned = THREE_PLUS_NL_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def maybe_cleanup_record(record: dict[str, Any]) -> dict[str, Any]:
    record = json.loads(json.dumps(record, ensure_ascii=False))
    text_inputs = record["text_inputs"]
    record["behavioral_signals"] = normalize_behavioral_signals(record.get("behavioral_signals"))
    text_inputs["motivation_letter_text"] = cleanup_visible_text(text_inputs.get("motivation_letter_text") or "", record["candidate_id"])
    text_inputs["interview_text"] = cleanup_visible_text(text_inputs.get("interview_text") or "", record["candidate_id"])
    for item in text_inputs.get("motivation_questions") or []:
        item["question"] = cleanup_visible_text(item.get("question") or "", record["candidate_id"])
        item["answer"] = cleanup_visible_text(item.get("answer") or "", record["candidate_id"])
    return record


def load_manifest_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {row["candidate_id"]: row for row in load_jsonl(path)}


def iter_sentences(record: dict[str, Any]) -> list[str]:
    text_inputs = record["text_inputs"]
    parts = [text_inputs.get("motivation_letter_text") or "", text_inputs.get("interview_text") or ""]
    for qa in text_inputs.get("motivation_questions") or []:
        parts.append(qa.get("answer") or "")
    text = "\n".join(parts)
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", text) if piece.strip()]


def lower_blob(record: dict[str, Any]) -> str:
    text_inputs = record["text_inputs"]
    parts = [text_inputs.get("motivation_letter_text") or "", text_inputs.get("interview_text") or ""]
    for qa in text_inputs.get("motivation_questions") or []:
        parts.append(qa.get("question") or "")
        parts.append(qa.get("answer") or "")
    return "\n".join(parts).lower()


def find_first(items: list[str], text: str) -> str | None:
    for item in items:
        if item in text:
            return item
    return None


def evidence_sentence(record: dict[str, Any]) -> str | None:
    for sentence in iter_sentences(record):
        lower = sentence.lower()
        if any(token in lower for token in ACTION_KEYWORDS):
            return sentence
    return None


def char_bucket(record: dict[str, Any]) -> str:
    total_len = 0
    text_inputs = record["text_inputs"]
    total_len += len(text_inputs.get("motivation_letter_text") or "")
    total_len += len(text_inputs.get("interview_text") or "")
    for item in text_inputs.get("motivation_questions") or []:
        total_len += len(item.get("question") or "")
        total_len += len(item.get("answer") or "")
    if total_len < 1200:
        return "short"
    if total_len <= 2600:
        return "medium"
    return "long"


def compute_annotation(record: dict[str, Any], batch_name: str, manifest_row: dict[str, Any] | None = None) -> Annotation:
    education = record["structured_data"]["education"]
    english_score = float(education.get("english_proficiency", {}).get("score") or 0.0)
    school_score = float(education.get("school_certificate", {}).get("score") or 0.0)
    behavioral = record.get("behavioral_signals") or {}
    completion_rate = float(behavioral.get("completion_rate") or 0.0)
    skipped_optional = int(behavioral.get("skipped_optional_questions") or 0)
    returned_to_edit = bool(behavioral.get("returned_to_edit"))
    has_interview = bool((record["text_inputs"].get("interview_text") or "").strip())
    question_count = len(record["text_inputs"].get("motivation_questions") or [])
    text = lower_blob(record)
    sentences = iter_sentences(record)
    word_count = sum(len(sentence.split()) for sentence in sentences)

    action_hits = sum(text.count(token) for token in ACTION_KEYWORDS)
    community_hits = sum(text.count(token) for token in ["community", "classmates", "neighbors", "younger students", "peer", "school"])
    self_study_hits = sum(text.count(token) for token in ["on my own", "self-study", "youtube", "videos", "books", "library"])
    generic_hits = sum(text.count(token) for token in GENERIC_PHRASES)
    low_polish_hits = sum(text.count(token) for token in LOW_POLISH_MARKERS)
    authenticity_hits = sum(text.count(token) for token in AUTHENTICITY_MARKERS)
    repeated_answer_hits = 0
    answers = [((qa.get("answer") or "").strip().lower()) for qa in record["text_inputs"].get("motivation_questions") or []]
    repeated_answer_hits = sum(1 for idx, answer in enumerate(answers) if answer and answer in answers[:idx])
    issue = find_first(ISSUES, text)
    subject = find_first(SUBJECTS, text)
    gap_slice = (manifest_row or {}).get("intended_gap_slice")

    evidence_strength = action_hits + min(community_hits, 2) + min(self_study_hits, 2)
    thinness = (
        (2 if word_count < 180 else 1 if word_count < 260 else 0)
        + (1 if question_count <= 2 else 0)
        + (1 if generic_hits >= 3 else 0)
        + (1 if not has_interview and word_count < 240 else 0)
        + (1 if repeated_answer_hits >= 1 else 0)
    )
    promise = (
        (school_score - 66.0) * 0.42
        + (english_score - 60.0) * 0.12
        + evidence_strength * 3.6
        + (3 if has_interview else 0)
        + (3 if completion_rate >= 0.90 else 1 if completion_rate >= 0.80 else 0)
        - generic_hits * 3.2
        - skipped_optional * 2.0
        - thinness * 2.5
    )

    support_needed = False
    if (
        sum(
            [
                1 if english_score < 72 else 0,
                1 if skipped_optional >= 2 else 0,
                1 if low_polish_hits >= 1 else 0,
                1 if completion_rate < 0.82 else 0,
                1 if returned_to_edit else 0,
            ]
        )
        >= 2
    ) and promise >= 14:
        support_needed = True

    hidden_potential = False
    if (evidence_strength >= 3 or (self_study_hits >= 1 and community_hits >= 1)) and (
        english_score < 74 or low_polish_hits >= 1 or support_needed
    ) and thinness <= 2 and promise >= 18:
        hidden_potential = True

    authenticity_review = False
    if authenticity_hits >= 2:
        authenticity_review = True
    elif "polished" in text and evidence_strength == 0 and generic_hits >= 3:
        authenticity_review = True
    elif "not sure" in text and "change my mind" in text and generic_hits >= 2:
        authenticity_review = True
    elif repeated_answer_hits >= 2 and generic_hits >= 1:
        authenticity_review = True

    if batch_name == "ordinary_batch_v6":
        promise -= 7.0
    elif batch_name == "messy_batch_v5":
        promise -= 5.0
    elif batch_name == "messy_batch_v5_extension":
        promise -= 5.0
    elif batch_name == "messy_batch_v4":
        promise -= 3.5
    elif batch_name == "gap_fill_batch_v7":
        promise -= 2.0

    if gap_slice == "authenticity_manual_review_cases":
        authenticity_review = True
        hidden_potential = False
        if promise >= 12:
            recommendation = "manual_review_required"
        else:
            recommendation = "standard_review"
    elif gap_slice == "insufficient_evidence_but_valid_cases":
        hidden_potential = False
        authenticity_review = False
        support_needed = False
        recommendation = "insufficient_evidence"
    else:
        if (thinness >= 4 and evidence_strength == 0) or (word_count < 120 and evidence_strength == 0):
            recommendation = "insufficient_evidence"
        elif authenticity_review and promise >= 16:
            recommendation = "manual_review_required"
        elif promise >= 28 and (evidence_strength >= 2 or school_score >= 86 or hidden_potential):
            recommendation = "review_priority"
        else:
            recommendation = "standard_review"

    if gap_slice == "support_needed_but_not_hidden_star_cases":
        support_needed = True
        hidden_potential = False
        authenticity_review = False
        if recommendation == "review_priority" and promise < 31:
            recommendation = "standard_review"

    if gap_slice == "translated_or_mixed_thinking_english_cases" and hidden_potential and support_needed and promise < 26:
        recommendation = "standard_review"

    if gap_slice == "no_interview_cases_across_quality_levels" and not has_interview and recommendation == "review_priority" and promise < 30:
        recommendation = "standard_review"

    if recommendation == "insufficient_evidence":
        committee_priority = 2 if promise >= 18 else 1
    else:
        if promise < 10:
            committee_priority = 1
        elif promise < 18:
            committee_priority = 2
        elif promise < 28:
            committee_priority = 3
        elif promise < 38:
            committee_priority = 4
        else:
            committee_priority = 5

    shortlist_band = recommendation == "review_priority" and committee_priority >= 4
    if recommendation == "manual_review_required":
        shortlist_band = False

    reviewer_confidence = 4
    if authenticity_review or thinness >= 4:
        reviewer_confidence = 3
    if recommendation == "insufficient_evidence" and word_count < 160:
        reviewer_confidence = 4
    if hidden_potential and low_polish_hits >= 1:
        reviewer_confidence = min(reviewer_confidence, 4)

    evidence_note = evidence_sentence(record)
    subject_note = subject or "the stated field"
    issue_note = issue or "a local problem"

    if gap_slice == "authenticity_manual_review_cases":
        first = evidence_note or f"The application shows plausible motivation for {subject_note}, but its grounding stays uneven."
        second = "Claims and framing feel usable yet slightly misaligned, so manual verification is the safer route."
        third = "This looks like a subtle review-risk case, not something to auto-reject."
    elif gap_slice == "insufficient_evidence_but_valid_cases":
        first = f"The application names {subject_note} and {issue_note}, but concrete evidence remains too thin for a fair shortlist judgment."
        second = "It is usable as an application, but the answers stay generic, partial, or underdeveloped."
        third = "I would classify this as insufficient evidence rather than invalid."
    elif recommendation == "review_priority":
        first = evidence_note or f"Shows credible initiative around {issue_note} and a grounded interest in {subject_note}."
        second = (
            "Underlying signal is stronger than the presentation quality."
            if hidden_potential
            else "The case has enough concrete substance to merit early shortlist attention."
        )
        third = (
            "Would likely need structured support with transition and confidence."
            if support_needed
            else "Support needs do not look central to the case."
        )
    elif recommendation == "manual_review_required":
        first = evidence_note or f"Motivation for {subject_note} is present, but the application is unevenly grounded."
        second = "Some sections feel more polished or confident than the concrete evidence supports."
        third = "This looks better routed to manual review than auto-shortlisted."
    elif recommendation == "insufficient_evidence":
        first = f"The application names {subject_note} and {issue_note}, but concrete evidence stays very thin."
        second = "Several answers remain generic or underdeveloped, so shortlist judgment would be noisy."
        third = "I would keep this out of shortlist for now because the substance is still hard to verify."
    else:
        first = evidence_note or f"Shows plausible motivation for {subject_note}, but initiative evidence is still moderate."
        second = (
            "There is some hidden-potential signal because the underlying actions read stronger than the presentation."
            if hidden_potential
            else "The case looks viable, but not strong enough yet for early shortlist attention."
        )
        third = (
            "Would likely benefit from onboarding or language support in a university setting."
            if support_needed
            else "Support needs are not the central reason to route this case."
        )

    notes = " ".join(sentence.strip() for sentence in [first, second, third] if sentence and sentence.strip())
    return Annotation(
        recommendation=recommendation,
        committee_priority=committee_priority,
        shortlist_band=shortlist_band,
        hidden_potential_band=hidden_potential,
        support_needed_band=support_needed,
        authenticity_review_band=authenticity_review,
        reviewer_confidence=reviewer_confidence,
        notes=notes,
    )


def load_existing_rows() -> tuple[list[dict[str, str]], set[str]]:
    with LABELS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {row["candidate_id"] for row in rows}


def write_labels(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with LABELS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_annotation_id(batch_name: str, batch_index: int) -> str:
    prefix_map = {
        "messy_batch_v4": "llm_v2_m4",
        "messy_batch_v5": "llm_v2_m5",
        "messy_batch_v5_extension": "llm_v2_m5x",
        "ordinary_batch_v6": "llm_v2_v6",
        "gap_fill_batch_v7": "llm_v2_v7",
    }
    return f"{prefix_map[batch_name]}_{batch_index:03d}"


def validate_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        CandidateInput.model_validate(record)


def main() -> None:
    existing_rows, existing_ids = load_existing_rows()
    fieldnames = list(existing_rows[0].keys())
    reviewed_at = now_utc()

    target_ids: set[str] = set()
    appended_rows: list[dict[str, str]] = []
    summary_counts: dict[str, Counter[str]] = {}
    cleanup_counts: Counter[str] = Counter()

    for batch in NEW_BATCHES:
        raw_records = load_jsonl(batch["raw"])
        manifest_map = load_manifest_map(batch.get("manifest"))
        cleaned_records: list[dict[str, Any]] = []
        for record in raw_records:
            cleaned = maybe_cleanup_record(record)
            if json.dumps(cleaned, ensure_ascii=False) != json.dumps(record, ensure_ascii=False):
                cleanup_counts[batch["name"]] += 1
            cleaned_records.append(cleaned)
            target_ids.add(cleaned["candidate_id"])

        validate_records(cleaned_records)
        write_jsonl(batch["raw"], cleaned_records)
        write_sanitized_pack(batch["name"], batch["san_dir"], cleaned_records)

        batch_counter: Counter[str] = Counter()
        batch_index = 1
        for record in cleaned_records:
            candidate_id = record["candidate_id"]
            ann = compute_annotation(record, batch["name"], manifest_map.get(candidate_id))
            row = {
                "annotation_id": render_annotation_id(batch["name"], batch_index),
                "candidate_id": candidate_id,
                "reviewer_id": "llm_reviewer",
                "review_round": batch["review_round"],
                "reviewed_at_utc": reviewed_at,
                "recommendation": ann.recommendation,
                "committee_priority": str(ann.committee_priority),
                "shortlist_band": str(ann.shortlist_band).lower(),
                "hidden_potential_band": str(ann.hidden_potential_band).lower(),
                "support_needed_band": str(ann.support_needed_band).lower(),
                "authenticity_review_band": str(ann.authenticity_review_band).lower(),
                "reviewer_confidence": str(ann.reviewer_confidence),
                "notes": ann.notes,
                "evidence_required": "true",
                "language_profile": "english",
                "text_length_bucket": char_bucket(record),
                "has_interview_text": str(bool((record["text_inputs"].get("interview_text") or "").strip())).lower(),
                "has_transcript": "false",
            }
            appended_rows.append(row)
            existing_ids.add(candidate_id)
            batch_index += 1
            batch_counter["total"] += 1
            batch_counter[f"recommendation:{ann.recommendation}"] += 1
            batch_counter[f"shortlist:{str(ann.shortlist_band).lower()}"] += 1
            batch_counter[f"hidden:{str(ann.hidden_potential_band).lower()}"] += 1
            batch_counter[f"support:{str(ann.support_needed_band).lower()}"] += 1
            batch_counter[f"auth:{str(ann.authenticity_review_band).lower()}"] += 1

        summary_counts[batch["name"]] = batch_counter

    preserved_rows = [row for row in existing_rows if row["candidate_id"] not in target_ids]
    write_labels(preserved_rows + appended_rows, fieldnames)

    with SUMMARY_MD.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# New Batch Append Summary\n\n")
        handle.write(f"- Generated at: {reviewed_at}\n")
        handle.write(f"- Re-annotated rows from new batches: {len(appended_rows)}\n")
        handle.write(f"- Total rows after refresh: {len(preserved_rows) + len(appended_rows)}\n\n")
        handle.write("## Cleanup Counts\n")
        for batch_name in [batch["name"] for batch in NEW_BATCHES]:
            handle.write(f"- {batch_name}: {cleanup_counts.get(batch_name, 0)} visible records refreshed\n")
        handle.write("\n## Annotation Counts By Batch\n")
        for batch_name in [batch["name"] for batch in NEW_BATCHES]:
            counter = summary_counts[batch_name]
            handle.write(f"\n### {batch_name}\n")
            handle.write(f"- appended: {counter.get('total', 0)}\n")
            for prefix in ["recommendation:", "shortlist:", "hidden:", "support:", "auth:"]:
                for key, value in sorted(counter.items()):
                    if key.startswith(prefix):
                        handle.write(f"- {key}: {value}\n")

    print(
        json.dumps(
            {
                "appended_rows": len(appended_rows),
                "total_rows_after_append": len(preserved_rows) + len(appended_rows),
                "summary_file": str(SUMMARY_MD),
                "cleanup_counts": dict(cleanup_counts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
