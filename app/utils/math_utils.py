"""Math helpers for normalized scoring scales."""

from __future__ import annotations

from typing import Iterable


def clamp01(value: float) -> float:
    """Clamp value to [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide numbers with default fallback."""
    if denominator == 0:
        return default
    return numerator / denominator


def to_display_score(raw: float) -> int:
    """Convert raw normalized score [0..1] to display integer [0..100]."""
    return int(round(clamp01(raw) * 100))


def weighted_average_normalized(values: Iterable[tuple[float, float]], default: float = 0.0) -> float:
    """Weighted average for normalized signals where each tuple is (value, weight)."""
    weighted_sum = 0.0
    total_weight = 0.0
    for value, weight in values:
        if weight <= 0:
            continue
        weighted_sum += clamp01(value) * weight
        total_weight += weight
    if total_weight == 0:
        return clamp01(default)
    return clamp01(weighted_sum / total_weight)
