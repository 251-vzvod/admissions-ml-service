"""Normalize candidate JSON text fields for stable NLP preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.text import normalize_multiline_text, normalize_unicode_text


TEXT_FIELD_NAMES = {
    "motivation_letter_text",
    "interview_text",
    "video_interview_transcript_text",
    "video_presentation_transcript_text",
    "video_transcript_text",
    "question",
    "answer",
    "review_notes",
    "snippet",
    "summary",
}


def _normalize_string(value: str, key_hint: str | None) -> str:
    if key_hint in TEXT_FIELD_NAMES:
        return normalize_multiline_text(value)
    return normalize_unicode_text(value)


def _walk_and_normalize(obj: Any, stats: Counter[str], key_hint: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {key: _walk_and_normalize(value, stats, key) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_normalize(item, stats, key_hint) for item in obj]
    if not isinstance(obj, str):
        return obj

    normalized = _normalize_string(obj, key_hint)
    if normalized != obj:
        stats["changed_string_fields"] += 1
        stats[f"changed_field:{key_hint or 'unknown'}"] += 1

        for label, marker in [
            ("smart_apostrophe", "'"),
            ("smart_quote", '"'),
            ("dash", "-"),
            ("ellipsis", "..."),
        ]:
            if marker in normalized and marker not in obj:
                stats[f"normalized_token:{label}"] += 1
        if any(token in obj for token in ("â€™", "â€œ", "â€", "â€”", "вЂ™", "вЂњ", "вЂќ", "вЂ”")):
            stats["mojibake_tokens_repaired"] += 1
    return normalized


def normalize_candidate_file(path: Path) -> Counter[str]:
    original = json.loads(path.read_text(encoding="utf-8"))
    normalized = deepcopy(original)
    stats: Counter[str] = Counter()
    normalized = _walk_and_normalize(normalized, stats)

    if normalized != original:
        path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stats["files_changed"] += 1
    stats["candidate_count"] = len(normalized.get("candidates", []))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        default=["data/candidates.json", "data/candidates_expanded_v1.json"],
        help="Candidate JSON files to normalize in place.",
    )
    args = parser.parse_args()

    aggregate: Counter[str] = Counter()
    for raw_path in args.files:
        path = Path(raw_path)
        stats = normalize_candidate_file(path)
        aggregate.update(stats)
        print(f"{path}: changed_files={stats['files_changed']} changed_string_fields={stats['changed_string_fields']}")

    print(json.dumps(dict(aggregate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
