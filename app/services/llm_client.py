"""LLM client for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

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


class OpenAICompatibleClient:
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


def build_llm_client(provider: str, base_url: str | None, api_key: str | None) -> OpenAICompatibleClient:
    """Factory for OpenAI-compatible client implementation."""
    provider_normalized = provider.strip().lower()

    if provider_normalized not in {"openai", "openai-compatible", "openai_compatible"}:
        raise LLMClientError("unsupported_llm_provider")

    if not base_url or not api_key:
        raise LLMClientError("missing_llm_credentials")

    return OpenAICompatibleClient(base_url=base_url, api_key=api_key, provider=provider_normalized)
