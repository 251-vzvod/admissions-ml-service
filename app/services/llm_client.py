"""LLM client abstraction with OpenAI-compatible and mock providers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float
    timeout_seconds: float
    max_retries: int


@dataclass(slots=True)
class LLMResponse:
    content: str
    provider: str
    model: str
    latency_ms: int


class LLMClientError(RuntimeError):
    """Raised when LLM provider call fails."""


class BaseLLMClient:
    """Provider-agnostic interface for structured extraction calls."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI-compatible chat completion client."""

    def __init__(self, base_url: str, api_key: str, provider: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider

    def complete(self, request: LLMRequest) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": request.model,
            "temperature": request.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }

        last_exc: Exception | None = None
        for attempt in range(request.max_retries + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=request.timeout_seconds) as client:
                    response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                latency_ms = int((time.perf_counter() - started) * 1000)
                return LLMResponse(content=content, provider=self.provider, model=request.model, latency_ms=latency_ms)
            except Exception as exc:  # pragma: no cover - network branch hard to unit test reliably
                last_exc = exc
                if attempt >= request.max_retries:
                    break

        raise LLMClientError("llm_provider_request_failed") from last_exc


class MockLLMClient(BaseLLMClient):
    """Mock provider for deterministic local tests."""

    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.mode == "invalid":
            content = "{invalid_json"
        else:
            content = json.dumps(
                {
                    "motivation_clarity": 0.72,
                    "initiative": 0.68,
                    "leadership_impact": 0.61,
                    "growth_trajectory": 0.74,
                    "resilience": 0.66,
                    "program_fit": 0.73,
                    "evidence_richness": 0.64,
                    "specificity_score": 0.62,
                    "evidence_count": 0.58,
                    "consistency_score": 0.70,
                    "completeness_score": 0.71,
                    "genericness_score": 0.29,
                    "contradiction_flag": False,
                    "polished_but_empty_score": 0.24,
                    "cross_section_mismatch_score": 0.19,
                    "top_strength_signals": ["self-started initiative", "clear growth reflection"],
                    "main_gap_signals": ["limited quantified outcomes"],
                    "uncertainties": ["some claims need deeper verification"],
                    "evidence_spans": [
                        {
                            "dimension": "initiative",
                            "source": "motivation_questions",
                            "text": "I started a student club and organized weekly sessions",
                        }
                    ],
                    "extractor_rationale": "Signals are grounded in repeated action examples and reflective narrative.",
                }
            )

        return LLMResponse(content=content, provider="mock", model=request.model, latency_ms=1)


def build_llm_client(provider: str, base_url: str | None, api_key: str | None) -> BaseLLMClient:
    """Factory for configured LLM client implementation."""
    provider_normalized = provider.strip().lower()

    if provider_normalized == "mock":
        return MockLLMClient(mode="valid")
    if provider_normalized == "mock_invalid":
        return MockLLMClient(mode="invalid")

    if not base_url or not api_key:
        raise LLMClientError("missing_llm_credentials")

    return OpenAICompatibleClient(base_url=base_url, api_key=api_key, provider=provider_normalized)
