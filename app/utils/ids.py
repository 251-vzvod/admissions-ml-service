"""ID helpers for reproducible scoring runs."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def generate_scoring_run_id(prefix: str = "run") -> str:
    """Generate a unique run id with timestamp and random suffix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{ts}_{uuid4().hex[:8]}"
