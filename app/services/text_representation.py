"""Offline-oriented source-aware text representation for ML/NLP experiments.

This module is intentionally not wired into the public runtime scoring path.
It provides chunk-level semantic features that can be cached and consumed by
research scripts without changing the service contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
import statistics
from typing import Protocol

from app.services.preprocessing import NormalizedTextBundle
from app.services.semantic_rubrics import RUBRIC_PROTOTYPES
from app.utils.math_utils import clamp01, safe_div, weighted_average_normalized
from app.utils.text import normalize_unicode_text, split_sections, word_count

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


TOKEN_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
EMBEDDING_DIM = 512
DEFAULT_TOP_K = 3
MAX_CHUNKS_PER_SOURCE = 24
MAX_WORDS_PER_CHUNK = 120

SOURCE_ALIASES = {
    "motivation_letter_text": "essay",
    "motivation_questions": "qa",
    "interview_text": "interview",
    "video_interview_transcript_text": "video_interview",
    "video_presentation_transcript_text": "video_presentation",
}

FAMILY_TO_PROTOTYPE_KEY = {
    "growth": "growth_trajectory",
    "service": "community_orientation",
    "motivation": "motivation_authenticity",
    "leadership": "leadership_potential",
    "groundedness": "authenticity_groundedness",
}


@dataclass(frozen=True, slots=True)
class TextRepresentationConfig:
    backend: str = "sentence_transformer"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None
    top_k_mean: int = DEFAULT_TOP_K
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE
    max_words_per_chunk: int = MAX_WORDS_PER_CHUNK


@dataclass(frozen=True, slots=True)
class SourceChunk:
    source_key: str
    source_alias: str
    chunk_id: str
    text: str
    word_count: int


@dataclass(frozen=True, slots=True)
class ChunkRepresentation:
    chunk: SourceChunk
    family_scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class TextRepresentationResult:
    feature_map: dict[str, float]
    chunk_records: list[ChunkRepresentation]
    diagnostics: dict[str, str | int | float]


class TextEncoder(Protocol):
    backend_name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token]


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    compact = " ".join(text.lower().split())
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[idx : idx + n] for idx in range(len(compact) - n + 1)]


def _hash_vectorize(text: str) -> list[float]:
    cleaned = normalize_unicode_text(text).lower()
    vector = [0.0] * EMBEDDING_DIM
    for token in _tokenize(cleaned):
        vector[hash(f"tok::{token}") % EMBEDDING_DIM] += 1.0
    for gram in _char_ngrams(cleaned):
        vector[hash(f"chr::{gram}") % EMBEDDING_DIM] += 0.35
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


class HashTextEncoder:
    backend_name = "hash"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vectorize(text) for text in texts]


class SentenceTransformerTextEncoder:
    backend_name = "sentence_transformer"

    def __init__(self, model_name: str, device: str | None = None) -> None:
        if SentenceTransformer is None:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence_transformers_unavailable")
        self.model = _load_sentence_transformer(model_name, device)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str | None = None):  # pragma: no cover - optional dependency
    kwargs = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def _build_encoder(config: TextRepresentationConfig) -> tuple[TextEncoder, str | None]:
    backend = config.backend.strip().lower()
    if backend in {"sentence_transformer", "sentence-transformers", "sentence_transformers"}:
        try:
            return SentenceTransformerTextEncoder(config.model_name, config.device), None
        except Exception as exc:  # pragma: no cover - optional dependency
            return HashTextEncoder(), f"sentence_transformer_fallback:{type(exc).__name__}"
    return HashTextEncoder(), None if backend == "hash" else "unknown_backend_fallback"


def _split_long_section(section: str, *, max_words_per_chunk: int) -> list[str]:
    cleaned = " ".join(section.split())
    if not cleaned:
        return []
    if word_count(cleaned) <= max_words_per_chunk:
        return [cleaned]
    sentences = [item.strip() for item in SENTENCE_RE.split(cleaned) if item.strip()]
    if not sentences:
        return [cleaned]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = word_count(sentence)
        if current and current_words + sentence_words > max_words_per_chunk:
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def _iter_source_chunks(bundle: NormalizedTextBundle, config: TextRepresentationConfig) -> list[SourceChunk]:
    source_sections: dict[str, list[str]] = {
        "motivation_letter_text": split_sections(bundle.motivation_letter_original),
        "motivation_questions": [text for text in bundle.motivation_answers_original if text.strip()],
        "interview_text": split_sections(bundle.interview_original),
        "video_interview_transcript_text": split_sections(bundle.video_interview_transcript_original),
        "video_presentation_transcript_text": split_sections(bundle.video_presentation_transcript_original),
    }

    chunks: list[SourceChunk] = []
    for source_key, sections in source_sections.items():
        source_alias = SOURCE_ALIASES[source_key]
        expanded_sections: list[str] = []
        for section in sections:
            expanded_sections.extend(
                _split_long_section(section, max_words_per_chunk=config.max_words_per_chunk)
            )
        for idx, text in enumerate(expanded_sections[: config.max_chunks_per_source]):
            cleaned = " ".join(text.split()).strip()
            if not cleaned:
                continue
            chunks.append(
                SourceChunk(
                    source_key=source_key,
                    source_alias=source_alias,
                    chunk_id=f"{source_alias}_{idx + 1}",
                    text=cleaned,
                    word_count=word_count(cleaned),
                )
            )
    return chunks


def _prototype_texts_for_family(family_name: str) -> tuple[list[str], list[str]]:
    rubric_key = FAMILY_TO_PROTOTYPE_KEY[family_name]
    rubric = RUBRIC_PROTOTYPES[rubric_key]
    return list(rubric["positive"]), list(rubric["negative"])


def _prototype_margin_score(chunk_vector: list[float], positive_vectors: list[list[float]], negative_vectors: list[list[float]]) -> float:
    best_positive = max((_cosine_similarity(chunk_vector, item) for item in positive_vectors), default=0.0)
    best_negative = max((_cosine_similarity(chunk_vector, item) for item in negative_vectors), default=0.0)
    return clamp01(0.5 + ((best_positive - best_negative) * 0.5))


def _source_word_distribution(chunks: list[ChunkRepresentation]) -> dict[str, float]:
    counts: dict[str, float] = {}
    total = 0.0
    for item in chunks:
        total += item.chunk.word_count
        counts[item.chunk.source_alias] = counts.get(item.chunk.source_alias, 0.0) + item.chunk.word_count
    if total <= 0:
        return {}
    return {source: value / total for source, value in counts.items()}


def _normalized_entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    if len(probabilities) == 1:
        return 0.0
    entropy = 0.0
    for value in probabilities:
        if value > 0:
            entropy -= value * math.log(value)
    return clamp01(safe_div(entropy, math.log(len(probabilities)), default=0.0))


def _top_mean(values: list[float], top_k: int) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(sorted(values, reverse=True)[: max(1, top_k)]))


def _alignment(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    pair_diffs: list[float] = []
    for idx in range(len(values)):
        for jdx in range(idx + 1, len(values)):
            pair_diffs.append(abs(values[idx] - values[jdx]))
    return clamp01(1.0 - float(statistics.fmean(pair_diffs)))


def build_text_representation(
    bundle: NormalizedTextBundle,
    config: TextRepresentationConfig | None = None,
) -> TextRepresentationResult:
    config = config or TextRepresentationConfig()
    encoder, fallback_note = _build_encoder(config)
    chunks = _iter_source_chunks(bundle, config)

    feature_map: dict[str, float] = {}
    chunk_records: list[ChunkRepresentation] = []
    diagnostics: dict[str, str | int | float] = {
        "requested_backend": config.backend,
        "requested_model_name": config.model_name,
        "active_backend": encoder.backend_name,
        "top_k_mean": config.top_k_mean,
        "max_chunks_per_source": config.max_chunks_per_source,
        "max_words_per_chunk": config.max_words_per_chunk,
        "chunk_count_total": len(chunks),
    }
    if fallback_note:
        diagnostics["backend_fallback_note"] = fallback_note

    for source_alias in SOURCE_ALIASES.values():
        for family_name in FAMILY_TO_PROTOTYPE_KEY:
            feature_map[f"repr_{source_alias}_{family_name}_top"] = 0.0
            feature_map[f"repr_{source_alias}_{family_name}_mean"] = 0.0

    feature_map.update(
        {
            "repr_source_coverage": 0.0,
            "repr_source_entropy": 0.0,
            "repr_single_source_reliance": 0.0,
            "repr_multi_source_evidence_balance": 0.0,
            "repr_non_empty_source_count": 0.0,
            "repr_chunk_count_total": 0.0,
            "repr_avg_chunk_words": 0.0,
            "repr_best_chunk_strength": 0.0,
            "repr_chunk_signal_spread": 0.0,
            "repr_strongest_evidence_density": 0.0,
            "repr_cross_source_growth_alignment": 0.0,
            "repr_cross_source_service_alignment": 0.0,
            "repr_cross_source_motivation_alignment": 0.0,
            "repr_interview_motivation_priority": 0.0,
            "repr_qa_service_priority": 0.0,
            "repr_essay_growth_priority": 0.0,
            "repr_leadership_for_others_priority": 0.0,
        }
    )

    if not chunks:
        return TextRepresentationResult(feature_map=feature_map, chunk_records=chunk_records, diagnostics=diagnostics)

    prototype_texts: list[str] = []
    prototype_specs: list[tuple[str, str]] = []
    for family_name in FAMILY_TO_PROTOTYPE_KEY:
        positives, negatives = _prototype_texts_for_family(family_name)
        for item in positives:
            prototype_specs.append((family_name, "positive"))
            prototype_texts.append(item)
        for item in negatives:
            prototype_specs.append((family_name, "negative"))
            prototype_texts.append(item)

    encoded = encoder.encode([chunk.text for chunk in chunks] + prototype_texts)
    chunk_vectors = encoded[: len(chunks)]
    prototype_vectors = encoded[len(chunks) :]

    positive_vectors_by_family: dict[str, list[list[float]]] = {name: [] for name in FAMILY_TO_PROTOTYPE_KEY}
    negative_vectors_by_family: dict[str, list[list[float]]] = {name: [] for name in FAMILY_TO_PROTOTYPE_KEY}
    for vector, (family_name, polarity) in zip(prototype_vectors, prototype_specs, strict=True):
        if polarity == "positive":
            positive_vectors_by_family[family_name].append(vector)
        else:
            negative_vectors_by_family[family_name].append(vector)

    scores_by_source_family: dict[str, dict[str, list[float]]] = {
        source_alias: {family_name: [] for family_name in FAMILY_TO_PROTOTYPE_KEY}
        for source_alias in SOURCE_ALIASES.values()
    }
    chunk_strengths: list[float] = []
    evidence_densities: list[float] = []

    for chunk, vector in zip(chunks, chunk_vectors, strict=True):
        family_scores: dict[str, float] = {}
        for family_name in FAMILY_TO_PROTOTYPE_KEY:
            family_score = _prototype_margin_score(
                vector,
                positive_vectors_by_family[family_name],
                negative_vectors_by_family[family_name],
            )
            family_scores[family_name] = family_score
            scores_by_source_family[chunk.source_alias][family_name].append(family_score)
        chunk_records.append(ChunkRepresentation(chunk=chunk, family_scores=family_scores))
        strength = max(family_scores.values(), default=0.0)
        chunk_strengths.append(strength)
        evidence_densities.append(
            weighted_average_normalized(
                [
                    (family_scores["groundedness"], 0.40),
                    (family_scores["growth"], 0.20),
                    (family_scores["service"], 0.20),
                    (family_scores["leadership"], 0.20),
                ]
            )
        )

    for source_alias, family_scores in scores_by_source_family.items():
        for family_name, values in family_scores.items():
            feature_map[f"repr_{source_alias}_{family_name}_top"] = max(values, default=0.0)
            feature_map[f"repr_{source_alias}_{family_name}_mean"] = _top_mean(values, config.top_k_mean)

    distribution = _source_word_distribution(chunk_records)
    distribution_values = list(distribution.values())
    feature_map["repr_source_coverage"] = safe_div(len(distribution_values), len(SOURCE_ALIASES), default=0.0)
    feature_map["repr_source_entropy"] = _normalized_entropy(distribution_values)
    feature_map["repr_single_source_reliance"] = max(distribution_values, default=0.0)
    feature_map["repr_multi_source_evidence_balance"] = weighted_average_normalized(
        [
            (feature_map["repr_source_entropy"], 0.60),
            (1.0 - feature_map["repr_single_source_reliance"], 0.40),
        ]
    )
    feature_map["repr_non_empty_source_count"] = safe_div(len(distribution_values), len(SOURCE_ALIASES), default=0.0)
    feature_map["repr_chunk_count_total"] = safe_div(len(chunks), len(chunks) + 8.0, default=0.0)
    feature_map["repr_avg_chunk_words"] = safe_div(
        statistics.fmean([chunk.chunk.word_count for chunk in chunk_records]),
        float(config.max_words_per_chunk),
        default=0.0,
    )
    feature_map["repr_best_chunk_strength"] = max(chunk_strengths, default=0.0)
    if len(chunk_strengths) > 1:
        feature_map["repr_chunk_signal_spread"] = clamp01(
            max(chunk_strengths) - float(statistics.median(chunk_strengths))
        )
    feature_map["repr_strongest_evidence_density"] = max(evidence_densities, default=0.0)

    def source_values(family_name: str) -> list[float]:
        values = [
            feature_map[f"repr_{source_alias}_{family_name}_top"]
            for source_alias in SOURCE_ALIASES.values()
            if distribution.get(source_alias, 0.0) > 0.0
        ]
        return values

    feature_map["repr_cross_source_growth_alignment"] = _alignment(source_values("growth"))
    feature_map["repr_cross_source_service_alignment"] = _alignment(source_values("service"))
    feature_map["repr_cross_source_motivation_alignment"] = _alignment(source_values("motivation"))

    feature_map["repr_interview_motivation_priority"] = weighted_average_normalized(
        [
            (feature_map["repr_interview_motivation_top"], 0.50),
            (feature_map["repr_video_interview_motivation_top"], 0.20),
            (feature_map["repr_qa_motivation_top"], 0.20),
            (feature_map["repr_interview_groundedness_top"], 0.10),
        ]
    )
    feature_map["repr_qa_service_priority"] = weighted_average_normalized(
        [
            (feature_map["repr_qa_service_top"], 0.55),
            (feature_map["repr_interview_service_top"], 0.25),
            (feature_map["repr_essay_service_top"], 0.10),
            (feature_map["repr_cross_source_service_alignment"], 0.10),
        ]
    )
    feature_map["repr_essay_growth_priority"] = weighted_average_normalized(
        [
            (feature_map["repr_essay_growth_top"], 0.45),
            (feature_map["repr_qa_growth_top"], 0.25),
            (feature_map["repr_interview_growth_top"], 0.20),
            (feature_map["repr_cross_source_growth_alignment"], 0.10),
        ]
    )
    feature_map["repr_leadership_for_others_priority"] = weighted_average_normalized(
        [
            (feature_map["repr_interview_leadership_top"], 0.35),
            (feature_map["repr_qa_service_top"], 0.25),
            (feature_map["repr_essay_service_top"], 0.15),
            (feature_map["repr_interview_service_top"], 0.15),
            (feature_map["repr_cross_source_service_alignment"], 0.10),
        ]
    )

    diagnostics["active_model_name"] = config.model_name if encoder.backend_name == "sentence_transformer" else "hash"
    diagnostics["non_empty_source_count"] = len(distribution_values)
    diagnostics["best_chunk_strength"] = round(feature_map["repr_best_chunk_strength"], 6)

    return TextRepresentationResult(feature_map=feature_map, chunk_records=chunk_records, diagnostics=diagnostics)
