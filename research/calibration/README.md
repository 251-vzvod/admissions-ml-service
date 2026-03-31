# Calibration Toolkit

This folder is for fast human-vs-model calibration on a small adjudicated set.

## Goal

Do not start with a large benchmark.

Start with `20-40` borderline cases and compare:

- committee recommendation
- shortlist surfacing
- hidden potential surfacing
- support-needed surfacing
- authenticity review surfacing

## Files

- `adjudication_template.json`: example schema for a small adjudicated calibration set
- `compare_to_human.py`: run model scoring + trace and compare against adjudicated decisions
- `llm_adjudication_prompt.md`: offline-only prompt for an LLM assistant that drafts preliminary calibration labels

## Typical Workflow

1. Copy `adjudication_template.json` and create a real calibration set.
2. Fill the `human_review` block for each case.
3. Run:

```bash
python research/calibration/compare_to_human.py ^
  --input research/calibration/adjudication_template.json ^
  --output-json research/reports/calibration_report.json ^
  --output-md research/reports/calibration_report.md
```

4. Review mismatches first.
5. Tune policy thresholds before touching scorer weights.

## Design Rule

This toolkit is for calibration, not for training a black-box model.

Keep the LLM adjudication prompt offline. The live runtime LLM path should stay explainability-only.
