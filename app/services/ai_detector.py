"""Optional auxiliary detector for AI-generated English text."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import CONFIG
from app.services.preprocessing import NormalizedTextBundle

try:  # pragma: no cover - optional dependency
    from langdetect import LangDetectException, detect
except Exception:  # pragma: no cover - optional dependency
    LangDetectException = Exception
    detect = None

try:  # pragma: no cover - optional dependency
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    AutoConfig = None
    AutoModel = None
    AutoTokenizer = None
    PreTrainedModel = object
    torch = None
    nn = None


@dataclass(slots=True)
class AIDetectorResult:
    enabled: bool
    applicable: bool
    language: str | None
    probability_ai_generated: float | None
    provider: str
    model: str
    note: str | None = None


class DesklibAIDetectionModel(PreTrainedModel):
    """Local wrapper matching the model card architecture."""

    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        if AutoModel is None or nn is None:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers_unavailable")
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        return {"logits": logits}


def _detect_language(bundle: NormalizedTextBundle) -> str | None:
    text = bundle.full_text_original.strip()
    if len(text) < 40:
        return None
    if detect is None:  # pragma: no cover - optional dependency
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


@lru_cache(maxsize=1)
def _load_detector_components():
    if AutoTokenizer is None or AutoConfig is None or torch is None:  # pragma: no cover - optional dependency
        raise RuntimeError("detector_dependencies_unavailable")
    model_name = CONFIG.ai_detector.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DesklibAIDetectionModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _predict_probability(text: str) -> float:
    tokenizer, model = _load_detector_components()
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=768,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()
    return float(probability)


def detect_ai_generated_text(bundle: NormalizedTextBundle) -> AIDetectorResult:
    """Return optional auxiliary AI-generation signal for English text only."""
    cfg = CONFIG.ai_detector
    if not cfg.enabled:
        return AIDetectorResult(
            enabled=False,
            applicable=False,
            language=None,
            probability_ai_generated=None,
            provider="huggingface-local",
            model=cfg.model,
            note="disabled",
        )

    if int(bundle.stats.get("word_count", 0)) < cfg.min_words:
        return AIDetectorResult(
            enabled=True,
            applicable=False,
            language=None,
            probability_ai_generated=None,
            provider="huggingface-local",
            model=cfg.model,
            note="not_enough_text",
        )

    language = _detect_language(bundle)
    if cfg.english_only and language != "en":
        return AIDetectorResult(
            enabled=True,
            applicable=False,
            language=language,
            probability_ai_generated=None,
            provider="huggingface-local",
            model=cfg.model,
            note="english_only_detector_not_applicable",
        )

    try:
        probability = _predict_probability(bundle.full_text_original)
    except Exception as exc:
        return AIDetectorResult(
            enabled=True,
            applicable=False,
            language=language,
            probability_ai_generated=None,
            provider="huggingface-local",
            model=cfg.model,
            note=f"detector_unavailable:{type(exc).__name__}",
        )

    return AIDetectorResult(
        enabled=True,
        applicable=True,
        language=language,
        probability_ai_generated=probability,
        provider="huggingface-local",
        model=cfg.model,
        note="ok",
    )
