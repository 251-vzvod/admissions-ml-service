"""Privacy and merit-safe projection layer.

This layer removes sensitive attributes from scoring inputs and ensures that
candidate scoring relies only on merit-safe signals.
"""

from __future__ import annotations

from typing import Any

from app.config import CONFIG


def _scrub_dict(data: dict[str, Any], excluded_fields: set[str], hits: set[str]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()
        if key_lower in excluded_fields:
            hits.add(key)
            continue
        if isinstance(value, dict):
            cleaned[key] = _scrub_dict(value, excluded_fields, hits)
        elif isinstance(value, list):
            cleaned[key] = [
                _scrub_dict(item, excluded_fields, hits) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def merit_safe_projection(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Remove excluded sensitive fields and return filtered payload + hit list."""
    hits: set[str] = set()
    excluded_fields = {field.lower() for field in CONFIG.excluded_fields}
    projected = _scrub_dict(payload, excluded_fields=excluded_fields, hits=hits)
    return projected, sorted(hits)
