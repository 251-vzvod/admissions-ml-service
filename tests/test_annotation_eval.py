from app.services.annotation_eval import CandidateAnnotation, build_label_evaluation


def test_build_label_evaluation_reports_ranking_metrics() -> None:
    scored = [
        {
            "candidate_id": "cand_a",
            "merit_score": 92,
            "semantic_rubric_scores": {
                "leadership_potential": 95,
                "growth_trajectory": 91,
                "motivation_authenticity": 88,
                "authenticity_groundedness": 84,
                "hidden_potential": 90,
            },
        },
        {
            "candidate_id": "cand_b",
            "merit_score": 71,
            "semantic_rubric_scores": {
                "leadership_potential": 74,
                "growth_trajectory": 76,
                "motivation_authenticity": 72,
                "authenticity_groundedness": 70,
                "hidden_potential": 65,
            },
        },
        {
            "candidate_id": "cand_c",
            "merit_score": 44,
            "semantic_rubric_scores": {
                "leadership_potential": 38,
                "growth_trajectory": 35,
                "motivation_authenticity": 41,
                "authenticity_groundedness": 33,
                "hidden_potential": 30,
            },
        },
    ]
    annotations = {
        "cand_a": CandidateAnnotation(
            candidate_id="cand_a",
            leadership_potential=5,
            growth_trajectory=5,
            motivation_authenticity=4,
            evidence_strength=4,
            committee_priority=5,
            hidden_potential_flag=True,
        ),
        "cand_b": CandidateAnnotation(
            candidate_id="cand_b",
            leadership_potential=4,
            growth_trajectory=4,
            motivation_authenticity=4,
            evidence_strength=3,
            committee_priority=4,
            needs_support_flag=True,
        ),
        "cand_c": CandidateAnnotation(
            candidate_id="cand_c",
            leadership_potential=2,
            growth_trajectory=2,
            motivation_authenticity=2,
            evidence_strength=2,
            committee_priority=2,
            authenticity_review_flag=True,
        ),
    }

    report = build_label_evaluation(scored, annotations)

    assert report["annotated_candidate_count"] == 3
    assert report["spearman_merit_vs_labels"] == 1.0
    assert report["pairwise_accuracy"] == 1.0
    assert report["hidden_potential_recall_at_k"] == 1.0
    assert report["semantic_dimension_alignment"]["leadership_potential_spearman"] == 1.0
