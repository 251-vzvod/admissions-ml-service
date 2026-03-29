"""End-to-end scoring pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.config import CONFIG
from app.schemas.decision import Recommendation, ReviewFlag
from app.schemas.output import ScoreResponse
from app.services.authenticity import estimate_authenticity_risk
from app.services.eligibility import evaluate_eligibility
from app.services.explanations import build_explanation
from app.services.llm_extractor import extract_explainability_with_llm
from app.services.preprocessing import preprocess_text_inputs
from app.services.privacy import merit_safe_projection
from app.services.recommendation import map_recommendation
from app.services.scoring import build_score_trace, compute_scores
from app.services.structured_features import extract_structured_features
from app.services.text_features import extract_text_features
from app.utils.ids import generate_scoring_run_id
from app.utils.math_utils import to_display_score


class ScoringPipeline:
    """Deterministic scoring pipeline with optional LLM explainability."""

    def _prepare_scoring_context(self, candidate_payload: dict[str, Any]) -> dict[str, Any]:
        projected, excluded_hits = merit_safe_projection(candidate_payload)

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

        auth_result = estimate_authenticity_risk(features=merged_features, diagnostics=text_result.diagnostics)
        merged_features["authenticity_risk_raw"] = auth_result.authenticity_risk_raw

        scoring_result = compute_scores(feature_map=merged_features, authenticity_risk_raw=auth_result.authenticity_risk_raw)

        recommendation_result = map_recommendation(
            eligibility_status=eligibility.status,
            merit_raw=scoring_result.merit_raw,
            confidence_raw=scoring_result.confidence_raw,
            authenticity_risk_raw=scoring_result.authenticity_risk_raw,
            feature_map=merged_features,
            prior_flags=auth_result.review_flags,
        )

        merged_flags = list(recommendation_result.review_flags)
        if any(reason.startswith("missing_required_materials") for reason in eligibility.reasons):
            merged_flags.append(ReviewFlag.MISSING_REQUIRED_MATERIALS)

        recommendation_flags = sorted(set(merged_flags))
        merit_breakdown = {k: to_display_score(v) for k, v in scoring_result.merit_breakdown_raw.items()}

        snapshot = {
            "motivation_clarity": round(float(merged_features.get("motivation_clarity", 0.0)), 4),
            "initiative": round(float(merged_features.get("initiative", 0.0)), 4),
            "leadership_impact": round(float(merged_features.get("leadership_impact", 0.0)), 4),
            "growth_trajectory": round(float(merged_features.get("growth_trajectory", 0.0)), 4),
            "resilience": round(float(merged_features.get("resilience", 0.0)), 4),
            "program_fit": round(float(merged_features.get("program_fit", 0.0)), 4),
            "evidence_richness": round(float(merged_features.get("evidence_richness", 0.0)), 4),
            "specificity_score": round(float(merged_features.get("specificity_score", 0.0)), 4),
            "evidence_count": round(float(merged_features.get("evidence_count", 0.0)), 4),
            "consistency_score": round(float(merged_features.get("consistency_score", 0.0)), 4),
            "completeness_score": round(float(merged_features.get("completeness_score", 0.0)), 4),
            "docs_count_score": round(float(merged_features.get("docs_count_score", 0.0)), 4),
            "portfolio_links_score": round(float(merged_features.get("portfolio_links_score", 0.0)), 4),
            "has_video_presentation": bool(merged_features.get("has_video_presentation", False)),
            "genericness_score": round(float(merged_features.get("genericness_score", 0.0)), 4),
            "contradiction_flag": bool(merged_features.get("contradiction_flag", False)),
            "polished_but_empty_score": round(float(merged_features.get("polished_but_empty_score", 0.0)), 4),
            "cross_section_mismatch_score": round(float(merged_features.get("cross_section_mismatch_score", 0.0)), 4),
            "authenticity_risk_raw": round(float(merged_features.get("authenticity_risk_raw", 0.0)), 4),
            "excluded_sensitive_fields_count": len(excluded_hits),
        }

        context.update(
            {
                "early_exit": False,
                "structured_result": structured_result,
                "text_result": text_result,
                "merged_features": merged_features,
                "auth_result": auth_result,
                "scoring_result": scoring_result,
                "recommendation_result": recommendation_result,
                "recommendation_flags": recommendation_flags,
                "merit_breakdown": merit_breakdown,
                "snapshot": snapshot,
            }
        )
        return context

    def score_candidate(self, candidate_payload: dict[str, Any], scoring_run_id: str | None = None) -> ScoreResponse:
        """Run full scoring flow for a single candidate payload."""
        scoring_run_id = scoring_run_id or generate_scoring_run_id()
        context = self._prepare_scoring_context(candidate_payload)

        projected = context["projected"]
        excluded_hits = context["excluded_hits"]
        bundle = context["bundle"]
        eligibility = context["eligibility"]

        base_response = {
            "candidate_id": str(projected.get("candidate_id", "")),
            "scoring_run_id": scoring_run_id,
            "scoring_version": CONFIG.scoring_version,
            "prompt_version": CONFIG.prompt_version,
            "extraction_mode": "deterministic_scoring",
            "extractor_version": CONFIG.llm.extractor_version,
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
                feature_map={},
                merit_breakdown={
                    "potential": 0,
                    "motivation": 0,
                    "leadership_agency": 0,
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
                merit_breakdown={
                    "potential": 0,
                    "motivation": 0,
                    "leadership_agency": 0,
                    "experience_skills": 0,
                    "trust_completeness": 0,
                },
                feature_snapshot={
                    "completeness_score": bundle.stats.get("non_empty_text_sources", 0) / 3.0,
                    "excluded_sensitive_fields_count": len(excluded_hits),
                },
                top_strengths=explanation.top_strengths,
                main_gaps=explanation.main_gaps,
                uncertainties=explanation.uncertainties,
                evidence_spans=explanation.evidence_spans,
                explanation=explanation.explanation,
            )

        merged_features = context["merged_features"]
        scoring_result = context["scoring_result"]
        recommendation_result = context["recommendation_result"]
        recommendation_flags = context["recommendation_flags"]
        merit_breakdown = context["merit_breakdown"]
        snapshot = context["snapshot"]
        text_result = context["text_result"]

        llm_metadata: dict[str, str | int | float] | None = None
        llm_strength_claims: list[dict[str, str]] | None = None
        llm_gap_claims: list[dict[str, str]] | None = None
        llm_uncertainty_claims: list[dict[str, str]] | None = None
        llm_evidence_spans: list[dict[str, str]] | None = None
        extractor_rationale: str | None = None

        try:
            llm_result = extract_explainability_with_llm(bundle=bundle, deterministic_signals=text_result.features)
            llm_metadata = llm_result.llm_metadata
            llm_strength_claims = llm_result.strength_claims
            llm_gap_claims = llm_result.gap_claims
            llm_uncertainty_claims = llm_result.uncertainty_claims
            llm_evidence_spans = llm_result.evidence_spans
            extractor_rationale = llm_result.rationale
        except RuntimeError as exc:
            llm_metadata = {
                "provider": CONFIG.llm.provider,
                "model": CONFIG.llm.model,
                "fallback_reason": str(exc),
            }

        explanation_result = build_explanation(
            feature_map=snapshot,
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
            provided_evidence_spans=llm_evidence_spans,
            extractor_rationale=extractor_rationale,
        )

        base_response["llm_metadata"] = llm_metadata

        return ScoreResponse(
            **base_response,
            merit_score=scoring_result.merit_score,
            confidence_score=scoring_result.confidence_score,
            authenticity_risk=scoring_result.authenticity_risk,
            recommendation=recommendation_result.recommendation,
            review_flags=recommendation_flags,
            merit_breakdown=merit_breakdown,
            feature_snapshot=snapshot,
            top_strengths=explanation_result.top_strengths,
            main_gaps=explanation_result.main_gaps,
            uncertainties=explanation_result.uncertainties,
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
        scoring_trace = build_score_trace(merged_features, auth_result.authenticity_risk_raw)

        payload.update(
            {
                "review_flags": context["recommendation_flags"],
                "recommendation": context["recommendation_result"].recommendation,
                "structured_features": context["structured_result"].features,
                "text_features": context["text_result"].features,
                "authenticity": {
                    "authenticity_risk_raw": round(auth_result.authenticity_risk_raw, 6),
                    "review_flags": context["auth_result"].review_flags,
                },
                "score_trace": scoring_trace,
            }
        )
        return payload

    def score_candidate_model(self, model: Any, scoring_run_id: str | None = None) -> ScoreResponse:
        """Helper for pydantic models accepted by API layer."""
        payload = model.model_dump(mode="python") if hasattr(model, "model_dump") else asdict(model)
        return self.score_candidate(payload, scoring_run_id=scoring_run_id)

    def score_candidate_trace_model(self, model: Any) -> dict[str, Any]:
        """Helper to build score trace from pydantic models in API layer."""
        payload = model.model_dump(mode="python") if hasattr(model, "model_dump") else asdict(model)
        return self.score_candidate_trace(payload)
