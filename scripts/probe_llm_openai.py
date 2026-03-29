"""Quick probe to verify OpenAI LLM extraction is actually called."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CONFIG
from app.services.pipeline import ScoringPipeline


def main() -> None:
    data = json.loads(Path("data/candidates.json").read_text(encoding="utf-8"))
    candidate = data["candidates"][0]

    pipe = ScoringPipeline()
    result = pipe.score_candidate(candidate).model_dump()

    safe_summary = {
        "provider": CONFIG.llm.provider,
        "model": CONFIG.llm.model,
        "extraction_mode": result.get("extraction_mode"),
        "extractor_version": result.get("extractor_version"),
        "llm_metadata": result.get("llm_metadata"),
        "has_evidence_spans": bool(result.get("evidence_spans")),
        "top_strengths_count": len(result.get("top_strengths", [])),
        "main_gaps_count": len(result.get("main_gaps", [])),
        "uncertainties_count": len(result.get("uncertainties", [])),
        "recommendation": result.get("recommendation"),
        "merit_score": result.get("merit_score"),
        "confidence_score": result.get("confidence_score"),
        "authenticity_risk": result.get("authenticity_risk"),
    }

    print(json.dumps(safe_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
