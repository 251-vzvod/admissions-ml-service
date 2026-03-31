# Research Layer

This directory contains offline-only artifacts and utilities.

It is intentionally outside the runtime service path.

Use this layer for:

- annotation analysis
- shortlist validation
- offline ranker training
- evaluation reports
- experiments that should not affect request-time behavior directly

Generated outputs should go to:

- `research/reports/`

Runtime rule:

- `app/services/*` should contain request-time logic only
- `research/*` should contain offline evaluation and training logic only
