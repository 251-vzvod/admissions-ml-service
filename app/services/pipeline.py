"""End-to-end scoring pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.config import CONFIG
from app.schemas.decision import Recommendation, ReviewFlag
from app.schemas.input import normalize_candidate_payload
from app.schemas.output import ScoreResponse
from app.services.ai_detector import detect_ai_generated_text
from app.services.authenticity import estimate_authenticity_risk
from app.services.claim_evidence import build_claim_evidence_map
from app.services.committee_guidance import build_committee_guidance
from app.services.eligibility import evaluate_eligibility
from app.services.explanations import build_explanation
from app.services.llm_extractor import extract_explainability_with_llm
from app.services.policy import build_policy_snapshot
from app.services.preprocessing import preprocess_text_inputs
from app.services.privacy import merit_safe_projection
from app.services.recommendation import map_recommendation
from app.services.reviewer_signals import build_reviewer_signals
from app.services.scoring import build_score_trace, compute_scores
from app.services.semantic_rubrics import extract_semantic_rubric_features
from app.services.shortlist import build_shortlist_signals
from app.services.structured_features import extract_structured_features
from app.services.text_features import extract_text_features
from app.utils.ids import generate_scoring_run_id
from app.utils.math_utils import to_display_score


class ScoringPipeline:
    """Deterministic scoring pipeline with optional LLM explainability."""

    def _prepare_scoring_context(self, candidate_payload: dict[str, Any]) -> dict[str, Any]:
        normalized_payload = normalize_candidate_payload(candidate_payload)
        projected, excluded_hits = merit_safe_projection(normalized_payload)

        text_inputs = projected.get("text_inputs") if isinstance(projected, dict) else {}
        if not isinstance(text_inputs, dict):
            text_inputs = {}

        bundle = preprocess_text_inputs(text_inputs=text_inputs)
        eligibility = evaluate_eligibility(
            candidate_id=str(projected.get("candidate_id", "")),
            consent=projected.get("consent"),
            bundle=bundle,
            profile=projected,
        )

        context: dict[str, Any] = {
            "projected": projected,
            "excluded_hits": excluded_hits,
            "bundle": bundle,
            "eligibility": eligibility,
        }

        if eligibility.status in {Recommendation.INVALID, Recommendation.INCOMPLETE_APPLICATION}:
            context["early_exit"] = True
            return context

        structured_result = extract_structured_features(
            structured_data=projected.get("structured_data") if isinstance(projected.get("structured_data"), dict) else None,
            behavioral_signals=projected.get("behavioral_signals")
            if isinstance(projected.get("behavioral_signals"), dict)
            else None,
            bundle=bundle,
        )

        text_result = extract_text_features(bundle=bundle, structured=structured_result.features)

        merged_features: dict[str, float | bool] = {}
        merged_features.update(structured_result.features)
        merged_features.update(text_result.features)
        semantic_result = extract_semantic_rubric_features(bundle=bundle, heuristic_features=merged_features)
        merged_features.update(semantic_result.features)

        ai_detector_result = detect_ai_generated_text(bundle=bundle)
        if ai_detector_result.probability_ai_generated is not None:
            merged_features["ai_detector_probability"] = ai_detector_result.probability_ai_generated
        merged_features["ai_detector_applicable"] = ai_detector_result.applicable

        auth_result = estimate_authenticity_risk(
            features=merged_features,
            diagnostics=text_result.diagnostics,
            ai_detector_result=ai_detector_result,
        )
        merged_features["authenticity_risk_raw"] = auth_result.authenticity_risk_raw

        scoring_result = compute_scores(
            feature_map=merged_features,
            authenticity_risk_raw=auth_result.authenticity_risk_raw,
            use_semantic_layer=True,
        )

        snapshot = {
            "motivation_clarity": round(float(merged_features.get("motivation_clarity", 0.0)), 4),
            "initiative": round(float(merged_features.get("initiative", 0.0)), 4),
            "growth_trajectory": round(float(merged_features.get("growth_trajectory", 0.0)), 4),
            "community_value_orientation": round(float(merged_features.get("community_value_orientation", 0.0)), 4),
            "evidence_count": round(float(merged_features.get("evidence_count", 0.0)), 4),
            "specificity_score": round(float(merged_features.get("specificity_score", 0.0)), 4),
            "consistency_score": round(float(merged_features.get("consistency_score", 0.0)), 4),
            "genericness_score": round(float(merged_features.get("genericness_score", 0.0)), 4),
            "polished_but_empty_score": round(float(merged_features.get("polished_but_empty_score", 0.0)), 4),
            "cross_section_mismatch_score": round(float(merged_features.get("cross_section_mismatch_score", 0.0)), 4),
            "authenticity_risk_raw": round(float(merged_features.get("authenticity_risk_raw", 0.0)), 4),
        }
        semantic_snapshot = {
            "leadership_potential": round(float(merged_features.get("semantic_leadership_potential", 0.0)), 4),
            "growth_trajectory": round(float(merged_features.get("semantic_growth_trajectory", 0.0)), 4),
            "motivation_authenticity": round(float(merged_features.get("semantic_motivation_authenticity", 0.0)), 4),
            "authenticity_groundedness": round(float(merged_features.get("semantic_authenticity_groundedness", 0.0)), 4),
            "community_orientation": round(float(merged_features.get("semantic_community_orientation", 0.0)), 4),
            "hidden_potential": round(float(merged_features.get("semantic_hidden_potential", 0.0)), 4),
        }
        reviewer_signals = build_reviewer_signals(merged_features)
        merit_breakdown = {k: to_display_score(v) for k, v in scoring_result.merit_breakdown_raw.items()}
        provisional_shortlist_signals = build_shortlist_signals(
            feature_map={**merged_features, **snapshot},
            semantic_scores={key: to_display_score(value) for key, value in semantic_snapshot.items()},
            merit_score=scoring_result.merit_score,
            confidence_score=scoring_result.confidence_score,
            authenticity_risk=scoring_result.authenticity_risk,
            recommendation=Recommendation.STANDARD_REVIEW,
        )
        policy = build_policy_snapshot(
            merit_score=scoring_result.merit_score,
            confidence_score=scoring_result.confidence_score,
            authenticity_risk=scoring_result.authenticity_risk,
            hidden_potential_score=provisional_shortlist_signals.hidden_potential_score,
            support_needed_score=provisional_shortlist_signals.support_needed_score,
            shortlist_priority_score=provisional_shortlist_signals.shortlist_priority_score,
            evidence_coverage_score=provisional_shortlist_signals.evidence_coverage_score,
            trajectory_score=provisional_shortlist_signals.trajectory_score,
        )
        recommendation_result = map_recommendation(
            eligibility_status=eligibility.status,
            merit_raw=scoring_result.merit_raw,
            confidence_raw=scoring_result.confidence_raw,
            authenticity_risk_raw=scoring_result.authenticity_risk_raw,
            feature_map=merged_features,
            prior_flags=auth_result.review_flags,
            policy=policy,
        )

        merged_flags = list(recommendation_result.review_flags)
        if any(reason.startswith("missing_required_materials") for reason in eligibility.reasons):
            merged_flags.append(ReviewFlag.MISSING_REQUIRED_MATERIALS)

        recommendation_flags = sorted(set(merged_flags))

        context.update(
            {
                "early_exit": False,
                "structured_result": structured_result,
                "text_result": text_result,
                "merged_features": merged_features,
                "auth_result": auth_result,
                "ai_detector_result": ai_detector_result,
                "semantic_result": semantic_result,
                "scoring_result": scoring_result,
                "recommendation_result": recommendation_result,
                "recommendation_flags": recommendation_flags,
                "merit_breakdown": merit_breakdown,
                "snapshot": snapshot,
                "semantic_snapshot": semantic_snapshot,
                "reviewer_signals": reviewer_signals,
                "policy": policy,
            }
        )
        return context

    def score_candidate(
        self,
        candidate_payload: dict[str, Any],
        scoring_run_id: str | None = None,
        enable_llm_explainability: bool | None = None,
    ) -> ScoreResponse:
        """Run full scoring flow for a single candidate payload."""
        scoring_run_id = scoring_run_id or generate_scoring_run_id()
        if enable_llm_explainability is None:
            enable_llm_explainability = CONFIG.llm.enabled
        context = self._prepare_scoring_context(candidate_payload)

        projected = context["projected"]
        excluded_hits = context["excluded_hits"]
        bundle = context["bundle"]
        eligibility = context["eligibility"]

        base_response = {
            "candidate_id": str(projected.get("candidate_id", "")),
            "scoring_run_id": scoring_run_id,
            "scoring_version": CONFIG.scoring_version,
            "extraction_mode": "deterministic_scoring",
            "llm_metadata": None,
            "eligibility_status": eligibility.status,
            "eligibility_reasons": list(eligibility.reasons),
        }

        if excluded_hits:
            base_response["eligibility_reasons"].append("sensitive_fields_excluded_from_scoring")

        if context["early_exit"]:
            recommendation = Recommendation.INVALID if eligibility.status == Recommendation.INVALID else Recommendation.INCOMPLETE_APPLICATION
            early_flags = [ReviewFlag.ELIGIBILITY_GATE]
            if any(reason.startswith("missing_required_materials") for reason in eligibility.reasons):
                early_flags.append(ReviewFlag.MISSING_REQUIRED_MATERIALS)

            explanation = build_explanation(
                review_signals={},
                merit_breakdown={
                    "potential": 0,
                    "motivation": 0,
                    "leadership_agency": 0,
                    "community_values": 0,
                    "experience_skills": 0,
                    "trust_completeness": 0,
                },
                recommendation=recommendation,
                review_flags=early_flags,
                sections=bundle.sections,
                extraction_mode="deterministic_scoring",
                merit_score=0,
                confidence_score=0,
                authenticity_risk=0,
            )

            return ScoreResponse(
                **base_response,
                merit_score=0,
                confidence_score=0,
                authenticity_risk=0,
                recommendation=recommendation,
                review_flags=early_flags,
                llm_rubric_assessment=None,
                merit_breakdown={
                    "potential": 0,
                    "motivation": 0,
                    "leadership_agency": 0,
                    "community_values": 0,
                    "experience_skills": 0,
                    "trust_completeness": 0,
                },
                semantic_rubric_scores={},
                hidden_potential_score=0,
                support_needed_score=0,
                shortlist_priority_score=0,
                evidence_coverage_score=0,
                trajectory_score=0,
                evidence_highlights=[],
                supported_claims=[],
                weakly_supported_claims=[],
                top_strengths=explanation.top_strengths,
                main_gaps=explanation.main_gaps,
                uncertainties=explanation.uncertainties,
                authenticity_review_reasons=[],
                ai_detector=None,
                committee_cohorts=["Eligibility gate"],
                why_candidate_surfaced=[],
                what_to_verify_manually=["Complete the missing required materials before substantive committee review."],
                suggested_follow_up_question="What prevented you from completing the required application materials, and can you provide them now?",
                evidence_spans=explanation.evidence_spans,
                explanation=explanation.explanation,
            )

        merged_features = context["merged_features"]
        scoring_result = context["scoring_result"]
        recommendation_result = context["recommendation_result"]
        recommendation_flags = context["recommendation_flags"]
        merit_breakdown = context["merit_breakdown"]
        snapshot = context["snapshot"]
        semantic_snapshot = context["semantic_snapshot"]
        reviewer_signals = context["reviewer_signals"]
        policy = context["policy"]
        text_result = context["text_result"]
        semantic_result = context["semantic_result"]
        auth_result = context["auth_result"]
        ai_detector_result = context["ai_detector_result"]

        llm_metadata: dict[str, str | int | float] | None = None
        llm_strength_claims: list[dict[str, str]] | None = None
        llm_gap_claims: list[dict[str, str]] | None = None
        llm_uncertainty_claims: list[dict[str, str]] | None = None
        llm_evidence_spans: list[dict[str, str]] | None = None
        extractor_rationale: str | None = None
        llm_rubric_assessment: dict[str, int | str] | None = None
        llm_follow_up_question: str | None = None

        if enable_llm_explainability:
            try:
                llm_result = extract_explainability_with_llm(
                    bundle=bundle,
                    deterministic_signals=text_result.features,
                )
                llm_metadata = llm_result.llm_metadata
                llm_strength_claims = llm_result.strength_claims
                llm_gap_claims = llm_result.gap_claims
                llm_uncertainty_claims = llm_result.uncertainty_claims
                llm_evidence_spans = llm_result.evidence_spans
                extractor_rationale = llm_result.rationale
                llm_rubric_assessment = llm_result.rubric_assessment or None
                llm_follow_up_question = llm_result.committee_follow_up_question or None
            except RuntimeError as exc:
                llm_metadata = {
                    "provider": CONFIG.llm.provider,
                    "model": CONFIG.llm.model,
                    "fallback_reason": str(exc),
                }

        explanation_result = build_explanation(
            review_signals=reviewer_signals,
            merit_breakdown=merit_breakdown,
            recommendation=recommendation_result.recommendation,
            review_flags=recommendation_result.review_flags,
            sections=bundle.sections,
            extraction_mode="deterministic_scoring",
            merit_score=scoring_result.merit_score,
            confidence_score=scoring_result.confidence_score,
            authenticity_risk=scoring_result.authenticity_risk,
            provided_strength_claims=llm_strength_claims,
            provided_gap_claims=llm_gap_claims,
            provided_uncertainty_claims=llm_uncertainty_claims,
            provided_evidence_spans=llm_evidence_spans
            or [
                {
                    "source": item.source,
                    "snippet": item.snippet,
                }
                for item in list(semantic_result.evidence.values())[:2]
            ],
            extractor_rationale=extractor_rationale,
        )
        shortlist_signals = build_shortlist_signals(
            feature_map={**merged_features, **snapshot},
            semantic_scores={key: to_display_score(value) for key, value in semantic_snapshot.items()},
            merit_score=scoring_result.merit_score,
            confidence_score=scoring_result.confidence_score,
            authenticity_risk=scoring_result.authenticity_risk,
            recommendation=recommendation_result.recommendation,
        )
        committee_guidance = build_committee_guidance(
            review_signals=reviewer_signals,
            policy=policy,
            hidden_potential_score=shortlist_signals.hidden_potential_score,
            support_needed_score=shortlist_signals.support_needed_score,
            trajectory_score=shortlist_signals.trajectory_score,
            evidence_coverage_score=shortlist_signals.evidence_coverage_score,
            merit_score=scoring_result.merit_score,
            authenticity_risk=scoring_result.authenticity_risk,
            recommendation=recommendation_result.recommendation,
            review_flags=recommendation_flags,
        )
        claim_evidence = build_claim_evidence_map(
            bundle=bundle,
            review_signals=reviewer_signals,
            semantic_result=semantic_result,
            hidden_potential_score=shortlist_signals.hidden_potential_score,
            evidence_coverage_score=shortlist_signals.evidence_coverage_score,
            trajectory_score=shortlist_signals.trajectory_score,
        )
        supported_claim_dicts = [asdict(item) for item in claim_evidence.supported_claims]
        weak_claim_dicts = [asdict(item) for item in claim_evidence.weakly_supported_claims]
        evidence_highlights = supported_claim_dicts[:2]
        if len(evidence_highlights) < 3:
            evidence_highlights.extend(weak_claim_dicts[: 3 - len(evidence_highlights)])

        base_response["llm_metadata"] = llm_metadata

        return ScoreResponse(
            **base_response,
            merit_score=scoring_result.merit_score,
            confidence_score=scoring_result.confidence_score,
            authenticity_risk=scoring_result.authenticity_risk,
            recommendation=recommendation_result.recommendation,
            review_flags=recommendation_flags,
            llm_rubric_assessment=llm_rubric_assessment,
            merit_breakdown=merit_breakdown,
            semantic_rubric_scores={
                key: to_display_score(value) for key, value in semantic_snapshot.items()
            },
            hidden_potential_score=shortlist_signals.hidden_potential_score,
            support_needed_score=shortlist_signals.support_needed_score,
            shortlist_priority_score=shortlist_signals.shortlist_priority_score,
            evidence_coverage_score=shortlist_signals.evidence_coverage_score,
            trajectory_score=shortlist_signals.trajectory_score,
            evidence_highlights=evidence_highlights,
            supported_claims=supported_claim_dicts,
            weakly_supported_claims=weak_claim_dicts,
            top_strengths=explanation_result.top_strengths,
            main_gaps=explanation_result.main_gaps,
            uncertainties=explanation_result.uncertainties,
            authenticity_review_reasons=auth_result.review_reasons,
            ai_detector={
                "enabled": ai_detector_result.enabled,
                "applicable": ai_detector_result.applicable,
                "language": ai_detector_result.language,
                "probability_ai_generated": round(ai_detector_result.probability_ai_generated, 4)
                if ai_detector_result.probability_ai_generated is not None
                else None,
                "provider": ai_detector_result.provider,
                "model": ai_detector_result.model,
                "note": ai_detector_result.note,
            },
            committee_cohorts=committee_guidance.cohorts,
            why_candidate_surfaced=committee_guidance.why_candidate_surfaced,
            what_to_verify_manually=committee_guidance.what_to_verify_manually,
            suggested_follow_up_question=llm_follow_up_question or committee_guidance.suggested_follow_up_question,
            evidence_spans=explanation_result.evidence_spans,
            explanation=explanation_result.explanation,
        )

    def score_candidate_trace(self, candidate_payload: dict[str, Any]) -> dict[str, Any]:
        """Return full deterministic score trace for auditing/debugging."""
        context = self._prepare_scoring_context(candidate_payload)

        projected = context["projected"]
        eligibility = context["eligibility"]

        payload: dict[str, Any] = {
            "candidate_id": str(projected.get("candidate_id", "")),
            "eligibility_status": eligibility.status,
            "eligibility_reasons": list(eligibility.reasons),
            "scoring_version": CONFIG.scoring_version,
            "extraction_mode": "deterministic_scoring",
        }

        if context["early_exit"]:
            payload["note"] = "score_trace_unavailable_for_invalid_or_incomplete_application"
            return payload

        merged_features = context["merged_features"]
        auth_result = context["auth_result"]
        scoring_trace = build_score_trace(merged_features, auth_result.authenticity_risk_raw, use_semantic_layer=True)

        payload.update(
            {
                "review_flags": context["recommendation_flags"],
                "recommendation": context["recommendation_result"].recommendation,
                "structured_features": context["structured_result"].features,
                "text_features": context["text_result"].features,
                "authenticity": {
                    "authenticity_risk_raw": round(auth_result.authenticity_risk_raw, 6),
                    "review_flags": context["auth_result"].review_flags,
                    "review_reasons": context["auth_result"].review_reasons,
                    "ai_detector": {
                        "enabled": context["ai_detector_result"].enabled,
                        "applicable": context["ai_detector_result"].applicable,
                        "language": context["ai_detector_result"].language,
                        "probability_ai_generated": round(context["ai_detector_result"].probability_ai_generated, 6)
                        if context["ai_detector_result"].probability_ai_generated is not None
                        else None,
                        "provider": context["ai_detector_result"].provider,
                        "model": context["ai_detector_result"].model,
                        "note": context["ai_detector_result"].note,
                    },
                },
                "semantic_features": context["semantic_result"].features,
                "reviewer_signals": context["reviewer_signals"],
                "policy": {
                    "priority_band": context["policy"].priority_band,
                    "shortlist_band": context["policy"].shortlist_band,
                    "hidden_potential_band": context["policy"].hidden_potential_band,
                    "support_needed_band": context["policy"].support_needed_band,
                    "authenticity_review_band": context["policy"].authenticity_review_band,
                    "insufficient_evidence_band": context["policy"].insufficient_evidence_band,
                },
                "semantic_evidence": {
                    key: {
                        "source": value.source,
                        "snippet": value.snippet,
                        "similarity": value.similarity,
                    }
                    for key, value in context["semantic_result"].evidence.items()
                },
                "score_trace": scoring_trace,
            }
        )
        return payload

    def score_candidate_model(
        self,
        model: Any,
        scoring_run_id: str | None = None,
        enable_llm_explainability: bool | None = None,
    ) -> ScoreResponse:
        """Helper for pydantic models accepted by API layer."""
        payload = model.model_dump(mode="python") if hasattr(model, "model_dump") else asdict(model)
        return self.score_candidate(
            payload,
            scoring_run_id=scoring_run_id,
            enable_llm_explainability=enable_llm_explainability,
        )

    def score_candidate_trace_model(self, model: Any) -> dict[str, Any]:
        """Helper to build score trace from pydantic models in API layer."""
        payload = model.model_dump(mode="python") if hasattr(model, "model_dump") else asdict(model)
        return self.score_candidate_trace(payload)
