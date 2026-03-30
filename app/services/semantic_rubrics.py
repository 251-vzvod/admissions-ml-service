"""Semantic rubric prototypes and lightweight embedding-based matching."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

from app.services.preprocessing import NormalizedTextBundle
from app.utils.math_utils import clamp01, weighted_average_normalized


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


def _vectorize(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for token in _tokenize(text):
        index = hash(f"tok::{token}") % EMBEDDING_DIM
        vector[index] += 1.0
    for gram in _char_ngrams(text):
        index = hash(f"chr::{gram}") % EMBEDDING_DIM
        vector[index] += 0.35

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    return max(-1.0, min(1.0, dot))


def _prototype_vector(texts: list[str]) -> list[float]:
    if not texts:
        return [0.0] * EMBEDDING_DIM
    vectors = [_vectorize(text) for text in texts if text.strip()]
    if not vectors:
        return [0.0] * EMBEDDING_DIM
    merged = [0.0] * EMBEDDING_DIM
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
    """Score candidate text against rubric prototypes using lightweight embeddings."""
    heuristic_features = heuristic_features or {}
    chunks = _iter_candidate_chunks(bundle)
    if not chunks:
        empty = {
            "semantic_leadership_potential": 0.0,
            "semantic_growth_trajectory": 0.0,
            "semantic_motivation_authenticity": 0.0,
            "semantic_authenticity_groundedness": 0.0,
            "semantic_hidden_potential": 0.0,
        }
        return SemanticRubricResult(features=empty, evidence={}, diagnostics={"chunk_count": 0, "backend": "hash-embedding"})

    chunk_vectors = [(source, text, _vectorize(text)) for source, text in chunks]
    features: dict[str, float] = {}
    evidence: dict[str, SemanticEvidence] = {}

    for dimension, prototypes in RUBRIC_PROTOTYPES.items():
        positive = _prototype_vector(prototypes.get("positive", []))
        negative = _prototype_vector(prototypes.get("negative", []))

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

        feature_key = f"semantic_{dimension}"
        features[feature_key] = dimension_score
        evidence[dimension] = SemanticEvidence(
            source=best_source,
            snippet=best_text[:220] + ("..." if len(best_text) > 220 else ""),
            similarity=round(best_delta, 4),
        )

    hidden_potential = weighted_average_normalized(
        [
            (features.get("semantic_growth_trajectory", 0.0), 0.30),
            (features.get("semantic_leadership_potential", 0.0), 0.30),
            (features.get("semantic_motivation_authenticity", 0.0), 0.20),
            (float(heuristic_features.get("resilience", 0.0)), 0.10),
            (1.0 - float(heuristic_features.get("genericness_score", 0.0)), 0.10),
        ],
        default=0.0,
    )
    features["semantic_hidden_potential"] = hidden_potential

    return SemanticRubricResult(
        features=features,
        evidence=evidence,
        diagnostics={
            "chunk_count": len(chunks),
            "backend": "hash-embedding",
            "prototype_count": len(RUBRIC_PROTOTYPES),
        },
    )
