"""FastAPI entrypoint for inVision U scoring MVP."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="inVision U Scoring Service",
    version="1.0.0",
    description=(
        "Explainable, deterministic baseline scoring service for candidate decision support. "
        "This service does not make autonomous admission decisions."
    ),
)

app.include_router(router)
