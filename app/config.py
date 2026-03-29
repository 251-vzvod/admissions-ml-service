"""Global scoring configuration for the MVP service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any


SCORING_VERSION = "v1.0.0"
SCORING_CONFIG_VERSION = "cfg-v1.1.0"
WEIGHT_EXPERIMENT_PROTOCOL_VERSION = "weights-protocol-v1"
PROMPT_VERSION: str | None = None


def _strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def _load_dotenv() -> None:
    """Load .env values into process environment when not already set."""
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value)
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

EXCLUDED_FIELDS = [
    "first_name",
    "last_name",
    "middle_name",
    "full_name",
    "iin",
    "id_number",
    "address",
    "phone",
    "email",
    "social_links",
    "family_details",
    "income",
    "social_background",
    "gender",
    "sex",
    "citizenship",
    "ethnicity",
    "race",
    "religion",
]


@dataclass(slots=True)
class ScoringWeights:
    """Weights for merit, confidence, and risk calculations."""

    merit_breakdown: dict[str, float] = field(
        default_factory=lambda: {
            "potential": 0.30,
            "motivation": 0.25,
            "leadership_agency": 0.20,
            "experience_skills": 0.15,
            "trust_completeness": 0.10,
        }
    )
    confidence_components: dict[str, float] = field(
        default_factory=lambda: {
            "specificity_score": 0.30,
            "evidence_count": 0.25,
            "consistency_score": 0.25,
            "completeness_score": 0.20,
        }
    )


@dataclass(slots=True)
class Thresholds:
    """Operational thresholds for recommendation routing."""

    min_words_meaningful_text: int = 40
    min_non_empty_sources: int = 1

    very_low_confidence: float = 0.35
    acceptable_confidence: float = 0.55

    high_merit: float = 0.72
    medium_merit: float = 0.55

    elevated_risk: float = 0.60
    high_risk: float = 0.75

    low_evidence: float = 0.35

    # Optional formal eligibility requirements (kept conservative by default).
    require_video_presentation: bool = False
    min_required_documents: int = 0
    min_portfolio_links: int = 0


@dataclass(slots=True)
class NormalizationConfig:
    """Scales and defaults for structured feature normalization."""

    english_scale_max: dict[str, float] = field(
        default_factory=lambda: {
            "ielts": 9.0,
            "toefl": 120.0,
            "toefl ibt": 120.0,
            "toefl pbt": 677.0,
            "cefr": 6.0,
        }
    )
    certificate_scale_max: dict[str, float] = field(
        default_factory=lambda: {
            "unt": 140.0,
            "nis graduation": 100.0,
            "kazakhstan high school diploma": 100.0,
            "kazakhstan school completion": 100.0,
        }
    )
    unknown_scale_default: float = 0.5


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return _strip_wrapping_quotes(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(_strip_wrapping_quotes(raw))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(_strip_wrapping_quotes(raw))
    except ValueError:
        return default


@dataclass(slots=True)
class LLMConfig:
    """Configuration for LLM-assisted extraction."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 20.0
    temperature: float = 0.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.6
    retry_jitter_seconds: float = 0.2
    fallback_to_baseline: bool = True
    extractor_version: str = "llm-extractor-v1"
    base_url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=_strip_wrapping_quotes(os.getenv("LLM_PROVIDER", "openai")),
            model=_strip_wrapping_quotes(os.getenv("LLM_MODEL", "gpt-4o-mini")),
            timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 20.0),
            temperature=_env_float("LLM_TEMPERATURE", 0.0),
            max_retries=_env_int("LLM_MAX_RETRIES", 1),
            retry_backoff_seconds=_env_float("LLM_RETRY_BACKOFF_SECONDS", 0.6),
            retry_jitter_seconds=_env_float("LLM_RETRY_JITTER_SECONDS", 0.2),
            fallback_to_baseline=_env_bool("LLM_FALLBACK_TO_BASELINE", True),
            base_url=_strip_wrapping_quotes(os.getenv("LLM_BASE_URL")) if os.getenv("LLM_BASE_URL") else None,
            api_key=_strip_wrapping_quotes(os.getenv("LLM_API_KEY")) if os.getenv("LLM_API_KEY") else None,
        )


@dataclass(slots=True)
class AppConfig:
    """Application configuration container."""

    scoring_version: str = SCORING_VERSION
    scoring_config_version: str = SCORING_CONFIG_VERSION
    weight_experiment_protocol_version: str = WEIGHT_EXPERIMENT_PROTOCOL_VERSION
    prompt_version: str | None = PROMPT_VERSION
    excluded_fields: list[str] = field(default_factory=lambda: EXCLUDED_FIELDS.copy())
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)


CONFIG = AppConfig()


def build_scoring_config_snapshot() -> dict[str, Any]:
    """Build a serializable snapshot for reproducible scoring/evaluation reports."""
    return {
        "scoring_version": CONFIG.scoring_version,
        "scoring_config_version": CONFIG.scoring_config_version,
        "weight_experiment_protocol_version": CONFIG.weight_experiment_protocol_version,
        "prompt_version": CONFIG.prompt_version,
        "excluded_fields": CONFIG.excluded_fields,
        "weights": {
            "merit_breakdown": CONFIG.weights.merit_breakdown,
            "confidence_components": CONFIG.weights.confidence_components,
        },
        "thresholds": asdict(CONFIG.thresholds),
        "normalization": {
            "english_scale_max": CONFIG.normalization.english_scale_max,
            "certificate_scale_max": CONFIG.normalization.certificate_scale_max,
            "unknown_scale_default": CONFIG.normalization.unknown_scale_default,
        },
        "llm": {
            "provider": CONFIG.llm.provider,
            "model": CONFIG.llm.model,
            "timeout_seconds": CONFIG.llm.timeout_seconds,
            "temperature": CONFIG.llm.temperature,
            "max_retries": CONFIG.llm.max_retries,
            "retry_backoff_seconds": CONFIG.llm.retry_backoff_seconds,
            "retry_jitter_seconds": CONFIG.llm.retry_jitter_seconds,
            "fallback_to_baseline": CONFIG.llm.fallback_to_baseline,
            "extractor_version": CONFIG.llm.extractor_version,
        },
    }
