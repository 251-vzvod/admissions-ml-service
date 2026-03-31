"""Semantic rubric prototypes and lightweight multilingual matching backends."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Iterable, Protocol

from app.config import CONFIG
from app.services.preprocessing import NormalizedTextBundle
from app.utils.math_utils import clamp01, weighted_average_normalized
from app.utils.text import normalize_unicode_text

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


TOKEN_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
EMBEDDING_DIM = 512


RUBRIC_PROTOTYPES: dict[str, dict[str, list[str]]] = {
    "leadership_potential": {
        "positive": [
            "organized people around a problem and took responsibility for the outcome",
            "started an initiative, coordinated others, and improved something for a group",
            "stood up for someone, influenced peers, and acted despite discomfort",
            "организовал людей вокруг проблемы и взял ответственность за результат",
            "запустил инициативу, координировал других и улучшил что-то для группы",
            "заступился за другого, повлиял на людей и действовал несмотря на дискомфорт",
        ],
        "negative": [
            "mostly talks about wanting to be a leader without concrete actions or impact",
            "focuses only on personal success and gives no evidence of initiative with others",
            "только говорит о лидерстве без конкретных действий и результата",
            "описывает только личный успех без инициативы и влияния на других",
        ],
    },
    "growth_trajectory": {
        "positive": [
            "describes a difficult period, specific setback, reflection, and what changed afterwards",
            "shows progress over time through repeated effort, learning, and adaptation",
            "описывает сложный период, конкретную неудачу, рефлексию и что изменилось потом",
            "показывает прогресс во времени через усилия, обучение и адаптацию",
        ],
        "negative": [
            "lists hardships without reflection or learning",
            "describes success as static talent rather than growth through effort",
            "перечисляет трудности без рефлексии и без того, чему научился",
            "описывает успех как врожденный талант, а не как рост через усилие",
        ],
    },
    "motivation_authenticity": {
        "positive": [
            "explains why this program matters with concrete reasons, values, and real future direction",
            "connects personal experience to mission, community, and learning goals",
            "объясняет зачем нужна программа через конкретные причины, ценности и реальное направление",
            "связывает личный опыт с миссией, сообществом и учебными целями",
        ],
        "negative": [
            "generic dream statement without grounded reasons or examples",
            "polished but empty motivation with vague impact claims",
            "общая мотивация без приземленных причин и примеров",
            "гладкий, но пустой текст с расплывчатыми заявлениями о влиянии",
        ],
    },
    "authenticity_groundedness": {
        "positive": [
            "uses concrete scenes, people, tradeoffs, and believable small details",
            "consistent voice across sections with specific examples and limits",
            "использует конкретные сцены, людей, компромиссы и правдоподобные детали",
            "сохраняет единый голос в разных секциях и приводит конкретные примеры с ограничениями",
        ],
        "negative": [
            "overly generic, polished, and abstract writing with little concrete evidence",
            "claims impact without names, situations, numbers, or real constraints",
            "слишком общий и гладкий текст без конкретных деталей",
            "заявляет о влиянии без ситуаций, чисел, ограничений и реального контекста",
        ],
    },
}


MULTILINGUAL_CONCEPTS: dict[str, list[str]] = {
    "leadership": [
        "leader",
        "leadership",
        "organize",
        "organized",
        "coordinate",
        "initiative",
        "лидер",
        "организ",
        "координ",
        "инициатив",
    ],
    "growth": [
        "growth",
        "adapt",
        "adapted",
        "improve",
        "improved",
        "learned",
        "reflection",
        "рост",
        "адапт",
        "улучш",
        "науч",
        "рефлекс",
    ],
    "motivation": [
        "mission",
        "purpose",
        "community",
        "future",
        "program",
        "мисси",
        "цель",
        "сообществ",
        "будущ",
        "програм",
    ],
    "groundedness": [
        "example",
        "details",
        "specific",
        "evidence",
        "outcome",
        "пример",
        "детал",
        "конкрет",
        "доказ",
        "результ",
    ],
    "responsibility": [
        "responsibility",
        "responsible",
        "accountable",
        "ответствен",
        "обязан",
    ],
    "challenge": [
        "difficulty",
        "setback",
        "failure",
        "problem",
        "challenge",
        "сложн",
        "неудач",
        "проблем",
        "трудн",
    ],
    "adaptation": [
        "changed",
        "change",
        "feedback",
        "adjusted",
        "adapted",
        "измен",
        "обратн",
        "адапт",
        "скоррект",
    ],
    "impact": [
        "impact",
        "improved",
        "helped",
        "result",
        "outcome",
        "влияни",
        "помог",
        "улучш",
        "результ",
    ],
}


@dataclass(slots=True)
class SemanticEvidence:
    source: str
    snippet: str
    similarity: float


@dataclass(slots=True)
class SemanticRubricResult:
    features: dict[str, float]
    evidence: dict[str, SemanticEvidence]
    diagnostics: dict[str, float | int | str]


class SemanticEncoder(Protocol):
    backend_name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token]


def _char_ngrams(text: str, n: int = 3) -> Iterable[str]:
    compact = " ".join(text.lower().split())
    if len(compact) < n:
        if compact:
            yield compact
        return
    for idx in range(len(compact) - n + 1):
        yield compact[idx : idx + n]


def _multilingual_semantic_expand(text: str) -> str:
    lowered = normalize_unicode_text(text).lower()
    concept_tokens: list[str] = []
    for concept, stems in MULTILINGUAL_CONCEPTS.items():
        if any(stem in lowered for stem in stems):
            concept_tokens.append(f"concept_{concept}")
    if not concept_tokens:
        return lowered
    return f"{lowered} {' '.join(concept_tokens)}"


def _hash_vectorize(text: str) -> list[float]:
    expanded = _multilingual_semantic_expand(text)
    vector = [0.0] * EMBEDDING_DIM
    for token in _tokenize(expanded):
        vector[hash(f"tok::{token}") % EMBEDDING_DIM] += 1.0
    for gram in _char_ngrams(expanded):
        vector[hash(f"chr::{gram}") % EMBEDDING_DIM] += 0.35

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    return max(-1.0, min(1.0, dot))


class HashSemanticEncoder:
    backend_name = "hash-embedding-multilingual-bridge"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vectorize(text) for text in texts]


class TfidfCharNgramEncoder:
    backend_name = "tfidf-char-ngram-multilingual-bridge"

    def __init__(self) -> None:
        if TfidfVectorizer is None:  # pragma: no cover - optional dependency
            raise RuntimeError("sklearn_unavailable")

    def encode(self, texts: list[str]) -> list[list[float]]:
        expanded_texts = [_multilingual_semantic_expand(text) for text in texts]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1,
            norm="l2",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(expanded_texts)
        return matrix.toarray().tolist()


class SentenceTransformerEncoder:
    backend_name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence_transformers_unavailable")
        self.model = _load_sentence_transformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        expanded_texts = [_multilingual_semantic_expand(text) for text in texts]
        vectors = self.model.encode(expanded_texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):  # pragma: no cover - optional dependency
    return SentenceTransformer(model_name)


def _build_encoder() -> tuple[SemanticEncoder, str | None]:
    backend = CONFIG.semantic.backend.strip().lower()
    if backend in {"sentence_transformer", "sentence-transformer", "sentence_transformers"}:
        try:
            return SentenceTransformerEncoder(CONFIG.semantic.model), None
        except Exception as exc:  # pragma: no cover - optional dependency
            return HashSemanticEncoder(), f"sentence_transformer_fallback:{type(exc).__name__}"
    if backend in {"tfidf", "tfidf_char_ngram", "tfidf-char-ngram"}:
        try:
            return TfidfCharNgramEncoder(), None
        except Exception as exc:  # pragma: no cover - optional dependency
            return HashSemanticEncoder(), f"tfidf_fallback:{type(exc).__name__}"
    return HashSemanticEncoder(), None if backend == "hash" else "unknown_backend_fallback"


def _prototype_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    merged = [0.0] * dim
    for vector in vectors:
        for idx, value in enumerate(vector):
            merged[idx] += value
    norm = math.sqrt(sum(value * value for value in merged))
    if norm == 0:
        return merged
    return [value / norm for value in merged]


def _iter_candidate_chunks(bundle: NormalizedTextBundle) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for source, sections in bundle.sections.items():
        for section in sections:
            cleaned = " ".join(section.split()).strip()
            if not cleaned:
                continue
            chunks.append((source, cleaned))
    if not chunks and bundle.full_text_original.strip():
        chunks.append(("full_text", " ".join(bundle.full_text_original.split())))
    return chunks


def extract_semantic_rubric_features(
    bundle: NormalizedTextBundle,
    heuristic_features: dict[str, float | bool] | None = None,
) -> SemanticRubricResult:
    """Score candidate text against rubric prototypes using configurable semantic backends."""
    heuristic_features = heuristic_features or {}
    chunks = _iter_candidate_chunks(bundle)
    encoder, backend_note = _build_encoder()

    if not chunks:
        empty = {
            "semantic_leadership_potential": 0.0,
            "semantic_growth_trajectory": 0.0,
            "semantic_motivation_authenticity": 0.0,
            "semantic_authenticity_groundedness": 0.0,
            "semantic_hidden_potential": 0.0,
        }
        diagnostics: dict[str, float | int | str] = {"chunk_count": 0, "backend": encoder.backend_name}
        if backend_note:
            diagnostics["backend_note"] = backend_note
        return SemanticRubricResult(features=empty, evidence={}, diagnostics=diagnostics)

    candidate_chunks = [text for _, text in chunks]
    all_texts = list(candidate_chunks)
    prototype_index: dict[str, dict[str, list[int]]] = {}
    for dimension, prototypes in RUBRIC_PROTOTYPES.items():
        prototype_index[dimension] = {"positive": [], "negative": []}
        for polarity in ("positive", "negative"):
            for text in prototypes.get(polarity, []):
                prototype_index[dimension][polarity].append(len(all_texts))
                all_texts.append(text)

    encoded = encoder.encode(all_texts)
    chunk_vectors = [(source, text, encoded[idx]) for idx, (source, text) in enumerate(chunks)]

    features: dict[str, float] = {}
    evidence: dict[str, SemanticEvidence] = {}

    for dimension in RUBRIC_PROTOTYPES:
        positive_vectors = [encoded[idx] for idx in prototype_index[dimension]["positive"]]
        negative_vectors = [encoded[idx] for idx in prototype_index[dimension]["negative"]]
        positive = _prototype_vector(positive_vectors)
        negative = _prototype_vector(negative_vectors)

        best_delta = -1.0
        best_source = chunk_vectors[0][0]
        best_text = chunk_vectors[0][1]
        chunk_scores: list[float] = []

        for source, text, vector in chunk_vectors:
            pos_similarity = _cosine_similarity(vector, positive)
            neg_similarity = _cosine_similarity(vector, negative)
            delta = pos_similarity - neg_similarity
            normalized = clamp01((delta + 1.0) / 2.0)
            chunk_scores.append(normalized)
            if delta > best_delta:
                best_delta = delta
                best_source = source
                best_text = text

        top_scores = sorted(chunk_scores, reverse=True)[:2]
        dimension_score = weighted_average_normalized(
            [(top_scores[0] if top_scores else 0.0, 0.65), (top_scores[1] if len(top_scores) > 1 else 0.0, 0.35)],
            default=0.0,
        )

        features[f"semantic_{dimension}"] = dimension_score
        evidence[dimension] = SemanticEvidence(
            source=best_source,
            snippet=best_text[:220] + ("..." if len(best_text) > 220 else ""),
            similarity=round(best_delta, 4),
        )

    hidden_potential = weighted_average_normalized(
        [
            (features.get("semantic_growth_trajectory", 0.0), 0.30),
            (features.get("semantic_leadership_potential", 0.0), 0.24),
            (features.get("semantic_motivation_authenticity", 0.0), 0.16),
            (features.get("semantic_authenticity_groundedness", 0.0), 0.10),
            (float(heuristic_features.get("resilience", 0.0)), 0.10),
            (float(heuristic_features.get("evidence_count", 0.0)), 0.05),
            (1.0 - float(heuristic_features.get("genericness_score", 0.0)), 0.05),
        ],
        default=0.0,
    )
    features["semantic_hidden_potential"] = hidden_potential

    diagnostics = {
        "chunk_count": len(chunks),
        "backend": encoder.backend_name,
        "prototype_count": len(RUBRIC_PROTOTYPES),
    }
    if backend_note:
        diagnostics["backend_note"] = backend_note
    if encoder.backend_name == "sentence-transformers":
        diagnostics["model"] = CONFIG.semantic.model

    return SemanticRubricResult(features=features, evidence=evidence, diagnostics=diagnostics)
