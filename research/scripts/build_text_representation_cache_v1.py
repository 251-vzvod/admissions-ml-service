from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from research.scripts.offline_ml_common import (
    DEFAULT_REPR_CONFIG,
    REPRESENTATION_CACHE_JOBLIB,
    REPRESENTATION_CACHE_METADATA_JSON,
    build_or_load_text_representation_cache,
)


def main() -> None:
    print("[offline-ml] Building source-aware text representation cache v1")
    cache = build_or_load_text_representation_cache(
        repr_config=DEFAULT_REPR_CONFIG,
        rebuild=True,
    )
    payload = {
        "artifact": str(REPRESENTATION_CACHE_JOBLIB),
        "metadata": str(REPRESENTATION_CACHE_METADATA_JSON),
        "row_count": len(cache),
        "representation_config": {
            "backend": DEFAULT_REPR_CONFIG.backend,
            "model_name": DEFAULT_REPR_CONFIG.model_name,
            "device": DEFAULT_REPR_CONFIG.device,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
