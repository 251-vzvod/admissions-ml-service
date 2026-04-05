"""Optional auxiliary detector for AI-generated text via Hugging Face Inference API."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any

from app.config import CONFIG
from app.services.preprocessing import NormalizedTextBundle
from app.utils.math_utils import clamp01, weighted_average_normalized

try:  # pragma: no cover - optional dependency
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - optional dependency
    InferenceClient = None

try:  # pragma: no cover - optional dependency
    from langdetect import LangDetectException, detect
except Exception:  # pragma: no cover - optional dependency
    LangDetectException = Exception
    detect = None


TOKEN_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


@dataclass(slots=True)
class AIDetectorSourceResult:
    source_key: str
    source_label: str
    applicable: bool
    probability_ai_generated: float | None
    provider: str
    model: str
    note: str | None = None
    question: str | None = None


@dataclass(slots=True)
class AIDetectorResult:
    enabled: bool
    applicable: bool
    language: str | None
    probability_ai_generated: float | None
    provider: str
    model: str
    source_results: list[AIDetectorSourceResult]
    note: str | None = None


@dataclass(slots=True)
class _NamedTextUnit:
    source_key: str
    source_label: str
    text: str
    weight: float
    question: str | None = None


def _normalize_text(raw: str | None) -> str:
    return " ".join((raw or "").split()).strip()


def _runtime_model_name() -> str:
    raw = (CONFIG.ai_detector.model or "").strip()
    if ":" in raw:
        return raw.split(":", 1)[0].strip()
    return raw


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _detect_language(text: str) -> str | None:
    if len(text) < 40 or detect is None:  # pragma: no cover - optional dependency
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def _iter_named_text_units(bundle: NormalizedTextBundle) -> list[_NamedTextUnit]:
    units: list[_NamedTextUnit] = []

    def add(source_key: str, source_label: str, text: str, weight: float, *, question: str | None = None) -> None:
        cleaned = _normalize_text(text)
        if cleaned:
            units.append(
                _NamedTextUnit(
                    source_key=source_key,
                    source_label=source_label,
                    text=cleaned,
                    weight=weight,
                    question=_normalize_text(question) or None,
                )
            )

    add("motivation_letter_text", "Motivational letter", bundle.motivation_letter_original, 0.30)
    add("interview_text", "Interview text", bundle.interview_original, 0.26)
    add("video_interview_transcript_text", "Interview transcript", bundle.video_interview_transcript_original, 0.22)
    add(
        "video_presentation_transcript_text",
        "Video presentation transcript",
        bundle.video_presentation_transcript_original,
        0.18,
    )
    for idx, answer in enumerate(bundle.motivation_answers_original):
        add(
            f"motivation_questions[{idx}].answer",
            f"Motivational question answer #{idx + 1}",
            answer,
            0.16,
            question=bundle.qa_questions_original[idx] if idx < len(bundle.qa_questions_original) else None,
        )
    return units


@lru_cache(maxsize=1)
def _build_hf_client() -> Any:
    if InferenceClient is None:
        raise RuntimeError("huggingface_hub_unavailable")
    if not CONFIG.ai_detector.api_key:
        raise RuntimeError("missing_hf_token")
    return InferenceClient(provider="auto", api_key=CONFIG.ai_detector.api_key)


def _score_item(raw: Any) -> tuple[bool, float | None]:
    label = str(getattr(raw, "label", "") or (raw.get("label") if isinstance(raw, dict) else "")).strip().lower()
    score_raw = getattr(raw, "score", None)
    if score_raw is None and isinstance(raw, dict):
        score_raw = raw.get("score")
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        score = None

    positive_markers = ("ai", "generated", "fake", "machine", "bot")
    negative_markers = ("human", "real")
    is_positive = any(marker in label for marker in positive_markers) and not any(marker in label for marker in negative_markers)
    return is_positive, score


def _parse_classification_response(response: Any) -> float:
    if isinstance(response, list):
        positives: list[float] = []
        negatives: list[float] = []
        for item in response:
            is_positive, score = _score_item(item)
            if score is None:
                continue
            if is_positive:
                positives.append(score)
            else:
                negatives.append(score)
        if positives:
            return clamp01(max(positives))
        if negatives:
            return clamp01(1.0 - max(negatives))
    if isinstance(response, dict):
        if "score" in response:
            is_positive, score = _score_item(response)
            if score is not None:
                return clamp01(score if is_positive else 1.0 - score)
        for key in ("label", "generated", "ai"):
            if key in response:
                is_positive, score = _score_item(response)
                if score is not None:
                    return clamp01(score if is_positive else 1.0 - score)
    raise RuntimeError("invalid_hf_detector_response")


def _predict_probability(text: str) -> float:
    client = _build_hf_client()
    response = client.text_classification(text, model=_runtime_model_name())
    return _parse_classification_response(response)


def _build_source_result(unit: _NamedTextUnit, *, language: str | None) -> AIDetectorSourceResult:
    cfg = CONFIG.ai_detector
    provider = cfg.provider
    model = cfg.model

    if _token_count(unit.text) < cfg.min_words:
        return AIDetectorSourceResult(
            source_key=unit.source_key,
            source_label=unit.source_label,
            applicable=False,
            probability_ai_generated=None,
            provider=provider,
            model=model,
            note="not_enough_text",
            question=unit.question,
        )

    try:
        probability = _predict_probability(unit.text)
    except Exception as exc:
        return AIDetectorSourceResult(
            source_key=unit.source_key,
            source_label=unit.source_label,
            applicable=False,
            probability_ai_generated=None,
            provider=provider,
            model=model,
            note=f"detector_unavailable:{type(exc).__name__}",
            question=unit.question,
        )

    return AIDetectorSourceResult(
        source_key=unit.source_key,
        source_label=unit.source_label,
        applicable=True,
        probability_ai_generated=probability,
        provider=provider,
        model=model,
        note="ok",
        question=unit.question,
    )


def _aggregate_probability(units: list[_NamedTextUnit], source_results: list[AIDetectorSourceResult]) -> float | None:
    weighted_items: list[tuple[float, float]] = []
    for unit, result in zip(units, source_results, strict=False):
        if not result.applicable or result.probability_ai_generated is None:
            continue
        length_boost = min(1.25, max(0.7, math.log(max(16, _token_count(unit.text)), 16)))
        weighted_items.append((result.probability_ai_generated, unit.weight * length_boost))
    if not weighted_items:
        return None
    return clamp01(weighted_average_normalized(weighted_items, default=0.0))


def detect_ai_generated_text(bundle: NormalizedTextBundle) -> AIDetectorResult:
    """Return aggregate and source-level AI-generation probabilities for text inputs."""
    cfg = CONFIG.ai_detector
    provider = cfg.provider
    model = cfg.model
    units = _iter_named_text_units(bundle)
    full_text = _normalize_text(bundle.full_text_original)
    language = _detect_language(full_text)

    if not cfg.enabled:
        return AIDetectorResult(
            enabled=False,
            applicable=False,
            language=language,
            probability_ai_generated=None,
            provider=provider,
            model=model,
            source_results=[],
            note="disabled",
        )

    if not units:
        return AIDetectorResult(
            enabled=True,
            applicable=False,
            language=language,
            probability_ai_generated=None,
            provider=provider,
            model=model,
            source_results=[],
            note="no_text_inputs",
        )

    source_results = [_build_source_result(unit, language=language) for unit in units]
    aggregate_probability = _aggregate_probability(units, source_results)
    applicable = any(item.applicable for item in source_results)

    note = "ok" if applicable and aggregate_probability is not None else "no_applicable_text_sources"
    return AIDetectorResult(
        enabled=True,
        applicable=applicable,
        language=language,
        probability_ai_generated=aggregate_probability,
        provider=provider,
        model=model,
        source_results=source_results,
        note=note,
    )
