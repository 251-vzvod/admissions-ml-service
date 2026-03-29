"""Shared LLM extraction type aliases."""

from __future__ import annotations

from typing import TypedDict


class LLMMetadata(TypedDict, total=False):
    provider: str
    model: str
    latency_ms: int
    fallback_reason: str
