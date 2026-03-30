"""FastAPI entrypoint for inVision U scoring MVP."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="inVision U Scoring Service",
    version="1.1.0",
    description=(
        "Explainable scoring service: deterministic feature extraction, semantic rubric matching, and optional LLM explainability. "
        "This service does not make autonomous admission decisions."
    ),
)

app.include_router(router)
