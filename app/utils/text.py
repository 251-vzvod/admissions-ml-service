"""Text utility helpers used by preprocessing and heuristics modules."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable


WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+", flags=re.UNICODE)
MULTI_SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]", flags=re.UNICODE)

UNICODE_REPLACEMENTS = {
    "\r\n": "\n",
    "\r": "\n",
    "\u00a0": " ",
    "\u202f": " ",
    "\u2009": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2026": "...",
}

COMMON_MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "вЂ™": "'",
    "вЂ": "'",
    "вЂњ": '"',
    "вЂќ": '"',
    "вЂ“": "-",
    "вЂ”": "-",
    "вЂ¦": "...",
}


def normalize_unicode_text(text: str) -> str:
    """Canonicalize unicode punctuation and repair common mojibake tokens."""
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFC", text)
    for src, dst in COMMON_MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(src, dst)
    for src, dst in UNICODE_REPLACEMENTS.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = ZERO_WIDTH_RE.sub("", cleaned)
    return cleaned


def normalize_multiline_text(text: str) -> str:
    """Normalize unicode and keep paragraph breaks while collapsing inner spacing."""
    cleaned = normalize_unicode_text(text)
    paragraphs = [normalize_whitespace(part) for part in cleaned.split("\n\n")]
    paragraphs = [part for part in paragraphs if part]
    return "\n\n".join(paragraphs)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace chars and strip ends."""
    return MULTI_SPACE_RE.sub(" ", text).strip()


def maybe_text(value: object) -> str:
    """Normalize unknown input into safe text value."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return normalize_whitespace(normalize_unicode_text(value))


def to_lower(text: str) -> str:
    """Lowercase helper for lexical rules."""
    return text.lower()


def word_count(text: str) -> int:
    """Unicode-aware word count."""
    return len(WORD_RE.findall(text))


def char_count(text: str) -> int:
    """Character count with whitespace included."""
    return len(text)


def sentence_count(text: str) -> int:
    """Approximate sentence count via punctuation split."""
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return len(parts)


def contains_any(text: str, terms: Iterable[str]) -> bool:
    """Check if text contains any literal term (case-insensitive)."""
    lowered = to_lower(text)
    return any(term in lowered for term in terms)


def count_occurrences(text: str, terms: Iterable[str]) -> int:
    """Count term occurrences in text (case-insensitive)."""
    lowered = to_lower(text)
    return sum(lowered.count(term) for term in terms)


def split_sections(text: str) -> list[str]:
    """Split text into rough paragraph-like sections."""
    cleaned = text.replace("\r\n", "\n")
    sections = [normalize_whitespace(p) for p in cleaned.split("\n\n") if normalize_whitespace(p)]
    if not sections and normalize_whitespace(cleaned):
        sections = [normalize_whitespace(cleaned)]
    return sections
