from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import sentencepiece  # noqa: F401
import torch
from transformers import MarianMTModel, MarianTokenizer

from app.schemas.input import CandidateInput

SOURCE_JSON = ROOT / "data" / "candidates.json"

RAW_DIR = ROOT / "data" / "ml_workbench" / "raw" / "generated" / "translated_batch_v3"
PACK_DIR = ROOT / "data" / "ml_workbench" / "processed" / "annotation_packs" / "translated_batch_v3"

RAW_JSONL = RAW_DIR / "translated_batch_v3_api_input.jsonl"
GEN_MANIFEST_JSONL = RAW_DIR / "translated_batch_v3_generation_manifest.jsonl"
SUMMARY_JSON = RAW_DIR / "translated_batch_v3_summary.json"

PACK_JSONL = PACK_DIR / "translated_batch_v3_annotation_pack.jsonl"
PACK_JSON = PACK_DIR / "translated_batch_v3_annotation_pack.json"
PACK_TABLE_CSV = PACK_DIR / "translated_batch_v3_annotation_pack_table.csv"
PACK_MANIFEST_JSON = PACK_DIR / "translated_batch_v3_annotation_pack_manifest.json"

MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
WORD_RE = re.compile(r"\b\w+\b")
ALLOWED_BEHAVIORAL_SIGNAL_KEYS = (
    "completion_rate",
    "returned_to_edit",
    "skipped_optional_questions",
)


@dataclass(frozen=True)
class SourceCandidate:
    source_candidate_id: str
    new_candidate_id: str
    raw: dict[str, Any]


def resolve_device() -> tuple[torch.device, str]:
    requested = os.environ.get("TRANSLATION_DEVICE", "cuda").strip().lower()
    if requested == "cpu":
        return torch.device("cpu"), "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("TRANSLATION_DEVICE=cuda was requested but CUDA is not available.")
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    return torch.device("cpu"), "cpu"


class RuToEnTranslator:
    def __init__(self) -> None:
        self.device, self.device_label = resolve_device()
        self.batch_size = int(os.environ.get("TRANSLATION_BATCH_SIZE", "24" if self.device.type == "cuda" else "8"))
        self.max_length = int(os.environ.get("TRANSLATION_MAX_LENGTH", "512"))
        self.num_beams = int(os.environ.get("TRANSLATION_NUM_BEAMS", "4"))
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(MODEL_NAME)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.cache: dict[str, str] = {}

    def translate_segments(self, segments: list[str]) -> list[str]:
        pending = [segment for segment in segments if segment not in self.cache]
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = self.model.generate(**encoded, max_length=self.max_length, num_beams=self.num_beams)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for source, translated in zip(batch, decoded):
                self.cache[source] = clean_spacing(translated)
        return [self.cache[segment] for segment in segments]


def clean_spacing(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def contains_cyrillic(text: str | None) -> bool:
    return bool(text and CYRILLIC_RE.search(text))


def sentence_chunks(paragraph: str, max_chars: int = 900) -> list[str]:
    paragraph = clean_spacing(paragraph)
    if not paragraph:
        return []
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", paragraph) if item.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            words = sentence.split()
            current_words: list[str] = []
            current_len = 0
            for word in words:
                projected = current_len + len(word) + (1 if current_words else 0)
                if current_words and projected > max_chars:
                    chunks.append(" ".join(current_words))
                    current_words = [word]
                    current_len = len(word)
                else:
                    current_words.append(word)
                    current_len = projected
            if current_words:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(" ".join(current_words))
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def translate_text(text: str | None, translator: RuToEnTranslator) -> str:
    if not text:
        return ""
    if not contains_cyrillic(text):
        return clean_spacing(text)

    paragraphs = [item for item in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")) if item.strip()]
    translated_paragraphs: list[str] = []
    for paragraph in paragraphs:
        chunks = sentence_chunks(paragraph)
        translated_chunks: list[str] = []
        chunk_batch: list[str] = []
        chunk_index_map: list[int] = []
        for idx, chunk in enumerate(chunks):
            if contains_cyrillic(chunk):
                chunk_batch.append(chunk)
                chunk_index_map.append(idx)
                translated_chunks.append("")
            else:
                translated_chunks.append(clean_spacing(chunk))
        if chunk_batch:
            translated_batch = translator.translate_segments(chunk_batch)
            for idx, translated in zip(chunk_index_map, translated_batch):
                translated_chunks[idx] = translated
        translated_paragraphs.append(clean_spacing(" ".join(translated_chunks)))
    return "\n\n".join(item for item in translated_paragraphs if item)


def normalize_english_type(value: Any) -> str | None:
    if value is None:
        return None
    text = clean_spacing(str(value))
    lower = text.lower()
    mapping = {
        "school curriculum": "school curriculum",
        "school classes": "school classes",
        "school english": "school English",
        "school_mark": "school grade",
        "ielts": "IELTS",
        "duolingo": "Duolingo",
        "duolingo english test": "Duolingo English Test",
    }
    return mapping.get(lower, text)


def normalize_school_type(value: Any, translator: RuToEnTranslator) -> str | None:
    if value is None:
        return None
    text = clean_spacing(str(value))
    lower = text.lower()
    mapping = {
        "kazakhstan certificate": "Kazakhstan certificate",
        "kazakhstani attestat": "Kazakhstan certificate",
        "attestat": "Kazakhstan certificate",
        "atestat": "Kazakhstan certificate",
        "аттестат": "Kazakhstan certificate",
        "обычный аттестат": "standard certificate",
        "attestat s otlichiem": "certificate with distinction",
        "almaty lyceum diploma": "Almaty Lyceum Diploma",
        "zertteu lyceum diploma": "Zertteu Lyceum Diploma",
        "nis diploma": "NIS Diploma",
    }
    if lower in mapping:
        return mapping[lower]
    if contains_cyrillic(text):
        return translate_text(text, translator)
    return text


def total_word_count(record: dict[str, Any]) -> int:
    text_inputs = record["text_inputs"]
    parts: list[str] = [text_inputs.get("motivation_letter_text") or ""]
    for item in text_inputs.get("motivation_questions") or []:
        parts.append(item.get("answer") or "")
    parts.append(text_inputs.get("interview_text") or "")
    return len(WORD_RE.findall(" ".join(parts)))


def length_bucket_from_words(words: int) -> str:
    if words < 400:
        return "short"
    if words <= 550:
        return "medium"
    return "long"


def reviewer_table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        education = record["structured_data"]["education"]
        text_inputs = record["text_inputs"]
        answers = text_inputs["motivation_questions"]
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "english_type": education["english_proficiency"]["type"],
                "english_score": education["english_proficiency"]["score"],
                "school_certificate_type": education["school_certificate"]["type"],
                "school_certificate_score": education["school_certificate"]["score"],
                "completion_rate": record["behavioral_signals"]["completion_rate"],
                "returned_to_edit": record["behavioral_signals"]["returned_to_edit"],
                "skipped_optional_questions": record["behavioral_signals"]["skipped_optional_questions"],
                "has_interview_text": bool(text_inputs.get("interview_text")),
                "motivation_question_count": len(answers),
                "total_text_word_count": total_word_count(record),
                "motivation_letter_text": text_inputs.get("motivation_letter_text") or "",
                "motivation_questions_text": "\n\n".join(
                    f"Q: {item['question']}\nA: {item['answer']}" for item in answers
                ),
                "interview_text": text_inputs.get("interview_text") or "",
            }
        )
    return rows


def visible_text_fields(record: dict[str, Any]) -> list[str]:
    text_inputs = record["text_inputs"]
    fields = [text_inputs.get("motivation_letter_text") or "", text_inputs.get("interview_text") or ""]
    fields.extend((item.get("answer") or "") for item in text_inputs.get("motivation_questions") or [])
    return [field for field in fields if field]


def count_source_id_leaks(records: list[dict[str, Any]], source_ids: list[str]) -> list[str]:
    hits: list[str] = []
    for record in records:
        joined = "\n".join(visible_text_fields(record))
        if any(source_id in joined for source_id in source_ids):
            hits.append(record["candidate_id"])
    return hits


def repeated_openings(records: list[dict[str, Any]], field: str, threshold: int = 3) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for record in records:
        text = (record["text_inputs"].get(field) or "").strip()
        if not text:
            continue
        opening = re.split(r"(?<=[.!?])\s+", text)[0].strip()
        if opening:
            buckets.setdefault(opening, []).append(record["candidate_id"])
    return {opening: ids for opening, ids in buckets.items() if len(ids) > threshold}


def load_russian_candidates() -> tuple[list[SourceCandidate], list[dict[str, str]]]:
    with SOURCE_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    candidates = payload["candidates"]
    russian_only = [item for item in candidates if item.get("content_profile", {}).get("language_profile") == "russian"]

    usable: list[SourceCandidate] = []
    skipped: list[dict[str, str]] = []
    for idx, candidate in enumerate(russian_only, start=1):
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        text_inputs = candidate.get("text_inputs")
        structured_data = candidate.get("structured_data")
        behavioral_signals = candidate.get("behavioral_signals")
        if not candidate_id:
            skipped.append({"source_candidate_id": "(missing)", "reason": "missing candidate_id"})
            continue
        if not isinstance(text_inputs, dict) or not isinstance(structured_data, dict) or not isinstance(behavioral_signals, dict):
            skipped.append({"source_candidate_id": candidate_id, "reason": "missing required public sections"})
            continue
        usable.append(
            SourceCandidate(
                source_candidate_id=candidate_id,
                new_candidate_id=f"tr_ru_v3_{idx:03d}",
                raw=candidate,
            )
        )
    return usable, skipped


def build_translated_record(candidate: SourceCandidate, translator: RuToEnTranslator) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = candidate.raw
    education = raw.get("structured_data", {}).get("education", {})
    text_inputs = raw.get("text_inputs", {})
    behavioral = raw.get("behavioral_signals", {})
    metadata = raw.get("metadata", {})

    motivation_questions = []
    for item in text_inputs.get("motivation_questions") or []:
        question = clean_spacing(str(item.get("question") or ""))
        answer = translate_text(item.get("answer") or "", translator)
        motivation_questions.append({"question": question, "answer": answer})

    translated = {
        "candidate_id": candidate.new_candidate_id,
        "structured_data": {
            "education": {
                "english_proficiency": {
                    "type": normalize_english_type(education.get("english_proficiency", {}).get("type")),
                    "score": education.get("english_proficiency", {}).get("score"),
                },
                "school_certificate": {
                    "type": normalize_school_type(education.get("school_certificate", {}).get("type"), translator),
                    "score": education.get("school_certificate", {}).get("score"),
                },
            }
        },
        "text_inputs": {
            "motivation_letter_text": translate_text(text_inputs.get("motivation_letter_text") or "", translator),
            "motivation_questions": motivation_questions,
            "interview_text": translate_text(text_inputs.get("interview_text") or "", translator),
        },
        "behavioral_signals": {
            "completion_rate": behavioral.get("completion_rate"),
            "returned_to_edit": behavioral.get("returned_to_edit"),
            "skipped_optional_questions": behavioral.get("skipped_optional_questions"),
        },
        "metadata": {
            "source": "translated_batch_v3_from_russian",
            "submitted_at": metadata.get("submitted_at"),
            "scoring_version": None,
        },
    }
    translated = CandidateInput.model_validate(translated).model_dump(mode="json", exclude_none=False)
    translated["behavioral_signals"] = {
        key: translated["behavioral_signals"].get(key)
        for key in ALLOWED_BEHAVIORAL_SIGNAL_KEYS
        if key in translated["behavioral_signals"]
    }
    sanitized = {
        "candidate_id": translated["candidate_id"],
        "structured_data": translated["structured_data"],
        "text_inputs": translated["text_inputs"],
        "behavioral_signals": translated["behavioral_signals"],
    }
    CandidateInput.model_validate(sanitized)
    manifest = {
        "candidate_id": candidate.new_candidate_id,
        "source_candidate_id": candidate.source_candidate_id,
        "source_language_profile": "russian",
        "translation_type": "ru_to_en_preserve_signal",
        "translation_notes": "Paragraph-level ru-to-en translation with no content expansion; preserved original question prompts and visible evidence density while normalizing structured labels into English-only public contract fields.",
        "visible_payload_sanitized": True,
    }
    return translated, sanitized, manifest


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)

    translator = RuToEnTranslator()
    print(
        json.dumps(
            {
                "translation_device": translator.device.type,
                "translation_device_label": translator.device_label,
                "translation_batch_size": translator.batch_size,
                "translation_num_beams": translator.num_beams,
            },
            ensure_ascii=False,
        )
    )
    candidates, skipped_records = load_russian_candidates()

    raw_records: list[dict[str, Any]] = []
    sanitized_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []

    for candidate in candidates:
        raw_record, sanitized_record, manifest_record = build_translated_record(candidate, translator)
        raw_records.append(raw_record)
        sanitized_records.append(sanitized_record)
        manifest_records.append(manifest_record)

    source_ids = [candidate.source_candidate_id for candidate in candidates]
    translated_ids = [record["candidate_id"] for record in raw_records]
    sanitized_ids = [record["candidate_id"] for record in sanitized_records]
    sanitized_payload_text = json.dumps(sanitized_records, ensure_ascii=False)
    text_length_counts = Counter(length_bucket_from_words(total_word_count(record)) for record in sanitized_records)
    with_interview_count = sum(1 for record in sanitized_records if record["text_inputs"].get("interview_text"))
    without_interview_count = len(sanitized_records) - with_interview_count
    source_id_leak_hits = count_source_id_leaks(sanitized_records, source_ids)
    repeated_letter_openings = repeated_openings(sanitized_records, "motivation_letter_text")
    repeated_interview_openings = repeated_openings(sanitized_records, "interview_text")
    english_only_heuristic = not CYRILLIC_RE.search(json.dumps(raw_records, ensure_ascii=False))

    validation_status = {
        "candidate_input_schema_raw": True,
        "candidate_input_schema_sanitized": True,
        "unique_candidate_ids": len(translated_ids) == len(set(translated_ids)),
        "raw_sanitized_one_to_one": translated_ids == sanitized_ids,
        "sanitized_has_no_metadata": all("metadata" not in item for item in sanitized_records),
        "source_id_leakage_absent": not source_id_leak_hits,
        "source_id_leakage_hits": source_id_leak_hits,
        "english_only_heuristic": english_only_heuristic,
        "repeated_letter_openings_over_threshold": repeated_letter_openings,
        "repeated_interview_openings_over_threshold": repeated_interview_openings,
        "translation_pipeline": "marian_ru_en_literal_chunked",
        "translation_device": translator.device.type,
        "translation_device_label": translator.device_label,
        "translation_batch_size": translator.batch_size,
        "question_counts_preserved": all(
            len(candidate.raw.get("text_inputs", {}).get("motivation_questions") or [])
            == len(record["text_inputs"].get("motivation_questions") or [])
            for candidate, record in zip(candidates, raw_records)
        ),
        "interview_presence_preserved": all(
            bool(candidate.raw.get("text_inputs", {}).get("interview_text"))
            == bool(record["text_inputs"].get("interview_text"))
            for candidate, record in zip(candidates, raw_records)
        ),
    }

    summary = {
        "candidate_count": len(raw_records),
        "source_candidate_count": len(candidates),
        "language_filter_used": 'content_profile.language_profile == "russian"',
        "with_interview_count": with_interview_count,
        "without_interview_count": without_interview_count,
        "text_length_counts": dict(text_length_counts),
        "validation_status": validation_status,
        "notes": [
            "Used all eligible Russian-only candidates from data/candidates.json unless the record was structurally unusable.",
            "Visible payloads stay inside the frozen public CandidateInput contract.",
            "Reviewer pack removes metadata and keeps only candidate_id, structured_data, text_inputs, and behavioral_signals.",
            "Translation was run paragraph-by-paragraph with chunked literal ru-to-en decoding to avoid summarization or style upgrade.",
            f"Skipped records: {len(skipped_records)}.",
        ],
    }

    with RAW_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in raw_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with GEN_MANIFEST_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with SUMMARY_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with PACK_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sanitized_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with PACK_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(sanitized_records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with PACK_TABLE_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "english_type",
                "english_score",
                "school_certificate_type",
                "school_certificate_score",
                "completion_rate",
                "returned_to_edit",
                "skipped_optional_questions",
                "has_interview_text",
                "motivation_question_count",
                "total_text_word_count",
                "motivation_letter_text",
                "motivation_questions_text",
                "interview_text",
            ],
        )
        writer.writeheader()
        writer.writerows(reviewer_table_rows(sanitized_records))

    pack_manifest = {
        "source_file": str(RAW_JSONL),
        "output_file": str(PACK_JSONL),
        "pretty_output_file": str(PACK_JSON),
        "table_output_file": str(PACK_TABLE_CSV),
        "candidate_count": len(sanitized_records),
        "sanitization_rule": "remove metadata to reduce source leakage; keep only candidate_id, structured_data, text_inputs, behavioral_signals",
        "intended_use": "translated Russian-only candidate batch v3 for English-only annotation research",
    }
    with PACK_MANIFEST_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(pack_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if skipped_records:
        print(json.dumps({"skipped_records": skipped_records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
