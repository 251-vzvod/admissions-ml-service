from research.calibration.compare_to_human import build_markdown_report, compare_cases


def test_compare_cases_returns_summary_and_comparisons() -> None:
    report = compare_cases(
        [
            type(
                "CalibrationCase",
                (),
                {
                    "candidate_id": "calib_case_001",
                    "candidate_payload": {
                        "candidate_id": "calib_case_001",
                        "text_inputs": {
                            "motivation_letter_text": (
                                "I started a peer tutoring group, changed the format after the first month failed, "
                                "and kept improving it for younger students."
                            ),
                            "interview_text": "I can explain what changed and why it worked better later.",
                        },
                        "consent": True,
                    },
                    "human_review": {
                        "recommendation": "standard_review",
                        "shortlist_band": True,
                        "hidden_potential_band": True,
                        "support_needed_band": False,
                        "authenticity_review_band": False,
                        "notes": "Should surface as a growth-heavy candidate.",
                    },
                },
            )()
        ]
    )

    assert report["summary"]["case_count"] == 1
    assert len(report["comparisons"]) == 1
    markdown = build_markdown_report(report)
    assert "# Calibration Report" in markdown
    assert "calib_case_001" in markdown
