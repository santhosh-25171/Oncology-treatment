"""
Scenario Adapter for Stage 5 Synthetic Oncology Testing Dashboard.
Transforms raw scenario data structures into standardized dashboard-ready views,
ensuring prominent synthetic labeling, safety badges, and complete source traceability.
"""

from typing import Dict, Any, List, Optional


SYNTHETIC_BADGE = "[SYNTHETIC TEST SCENARIO - RESEARCH ONLY]"


class ScenarioAdapter:
    """Adapts raw synthetic scenario records for UI rendering and dashboard APIs."""

    @staticmethod
    def to_dashboard_item(scenario: Dict[str, Any], evaluation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Creates a high-level summary item suitable for the scenario table/list view.
        """
        sid = scenario.get("scenario_id", "NOT AVAILABLE")
        category = scenario.get("scenario_category", "NOT AVAILABLE")
        method = scenario.get("generation_method", "NOT AVAILABLE")

        # Blind spot info
        bs_info = scenario.get("target_blind_spot", {})
        if isinstance(bs_info, dict):
            bs_id = bs_info.get("blind_spot_id", "NOT AVAILABLE")
            bs_reason = bs_info.get("reason", "NOT AVAILABLE")
        else:
            bs_id = str(bs_info)
            bs_reason = "NOT AVAILABLE"

        # Uncertainty
        uncertainty = scenario.get("uncertainty", {})
        unc_level = uncertainty.get("level", "NOT AVAILABLE") if isinstance(uncertainty, dict) else "NOT AVAILABLE"

        # Evaluation metrics (if provided)
        if evaluation:
            status = evaluation.get("evaluation_status", "NOT EVALUATED")
            stress_score_obj = evaluation.get("decision_stress_score", {})
            realism_score_obj = evaluation.get("realism_score", {})
            stress_score = stress_score_obj.get("overall_stress_score", "NOT AVAILABLE") if isinstance(stress_score_obj, dict) else "NOT AVAILABLE"
            realism_score = realism_score_obj.get("overall_realism_score", "NOT AVAILABLE") if isinstance(realism_score_obj, dict) else "NOT AVAILABLE"
            stress_level = stress_score_obj.get("stress_level", "NOT AVAILABLE") if isinstance(stress_score_obj, dict) else "NOT AVAILABLE"
        else:
            status = "NOT EVALUATED"
            stress_score = "NOT EVALUATED"
            realism_score = "NOT EVALUATED"
            stress_level = "NOT EVALUATED"

        return {
            "scenario_id": sid,
            "synthetic": True,
            "synthetic_badge": SYNTHETIC_BADGE,
            "scenario_category": category,
            "target_blind_spot": bs_id,
            "target_blind_spot_reason": bs_reason,
            "uncertainty_level": unc_level,
            "decision_stress_score": stress_score,
            "stress_level": stress_level,
            "realism_score": realism_score,
            "evaluation_status": status,
            "generation_method": method
        }

    @staticmethod
    def to_detailed_view(scenario: Dict[str, Any], evaluation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Creates a complete detailed representation of a synthetic scenario for the inspection panel.
        """
        sid = scenario.get("scenario_id", "NOT AVAILABLE")
        category = scenario.get("scenario_category", "NOT AVAILABLE")
        method = scenario.get("generation_method", "NOT AVAILABLE")

        # Patient Context
        p_ctx = scenario.get("patient_context", {})
        patient_context = {
            "age_group": p_ctx.get("age_group", "NOT AVAILABLE"),
            "sex": p_ctx.get("sex", "NOT AVAILABLE"),
            "cancer_type": p_ctx.get("cancer_type", "NOT AVAILABLE"),
            "histology": p_ctx.get("histology", "NOT AVAILABLE"),
            "stage": p_ctx.get("stage", "NOT AVAILABLE"),
            "prior_treatment_context": p_ctx.get("prior_treatment_context", "NOT AVAILABLE")
        }

        # Genomic Profile
        g_prof = scenario.get("genomic_profile", {})
        alterations = []
        for alt in g_prof.get("alterations", []):
            alterations.append({
                "gene": alt.get("gene", "NOT AVAILABLE"),
                "variant": alt.get("variant", "NOT AVAILABLE"),
                "alteration_type": alt.get("alteration_type", "NOT AVAILABLE"),
                "status": alt.get("status", "NOT AVAILABLE")
            })

        cooccurring = []
        for alt in g_prof.get("cooccurring_alterations", []):
            cooccurring.append({
                "gene": alt.get("gene", "NOT AVAILABLE"),
                "variant": alt.get("variant", "NOT AVAILABLE"),
                "status": alt.get("status", "NOT AVAILABLE")
            })

        resistance_features = g_prof.get("resistance_related_features", [])

        # Biomarkers
        b_marks = scenario.get("biomarkers", {})
        biomarkers = {
            "tmb": b_marks.get("tmb", "NOT AVAILABLE"),
            "tmb_status": b_marks.get("tmb_status", "NOT AVAILABLE"),
            "pdl1_tps": b_marks.get("pdl1_tps", "NOT AVAILABLE"),
            "msi_status": b_marks.get("msi_status", "NOT AVAILABLE")
        }

        # Clinical Context
        c_ctx = scenario.get("clinical_context", {})
        clinical_context = {
            "disease_status": c_ctx.get("disease_status", "NOT AVAILABLE"),
            "progression_context": c_ctx.get("progression_context", "NOT AVAILABLE")
        }

        # Assumptions & Evidence
        assumptions = scenario.get("synthetic_assumptions", ["NOT AVAILABLE"])
        evidence = scenario.get("reference_evidence", ["NOT AVAILABLE"])

        # Target Blind Spot
        bs = scenario.get("target_blind_spot", {})
        target_blind_spot = {
            "blind_spot_id": bs.get("blind_spot_id", "NOT AVAILABLE") if isinstance(bs, dict) else str(bs),
            "category": bs.get("category", "NOT AVAILABLE") if isinstance(bs, dict) else "NOT AVAILABLE",
            "coverage_status": bs.get("coverage_status", "NOT AVAILABLE") if isinstance(bs, dict) else "NOT AVAILABLE",
            "reason": bs.get("reason", "NOT AVAILABLE") if isinstance(bs, dict) else "NOT AVAILABLE"
        }

        # Uncertainty
        unc = scenario.get("uncertainty", {})
        uncertainty = {
            "level": unc.get("level", "NOT AVAILABLE") if isinstance(unc, dict) else "NOT AVAILABLE",
            "reason": unc.get("reason", "NOT AVAILABLE") if isinstance(unc, dict) else "NOT AVAILABLE"
        }

        # Provenance
        prov = scenario.get("provenance", {})
        provenance = {
            "reference_sources": prov.get("reference_sources", ["NOT AVAILABLE"]),
            "blind_spot_source": prov.get("blind_spot_source", "NOT AVAILABLE"),
            "generation_method": prov.get("generation_method", method),
            "generation_timestamp": prov.get("generation_timestamp", "NOT AVAILABLE"),
            "random_seed": prov.get("random_seed", "NOT AVAILABLE")
        }

        # Evaluation Details
        eval_view = None
        if evaluation:
            eval_view = {
                "evaluation_status": evaluation.get("evaluation_status", "NOT EVALUATED"),
                "audit_timestamp": evaluation.get("audit_timestamp", evaluation.get("evaluation_timestamp", "NOT AVAILABLE")),
                "flags": evaluation.get("flags", []),
                "schema_validation": evaluation.get("schema_validation", {"valid": "NOT EVALUATED"}),
                "provenance_validation": evaluation.get("provenance_validation", {"valid": "NOT EVALUATED"}),
                "blind_spot_audit": evaluation.get("blind_spot_audit", {"blind_spot_supported": "NOT EVALUATED"}),
                "genomic_consistency": evaluation.get("genomic_consistency", {"valid": "NOT EVALUATED"}),
                "clinical_consistency": evaluation.get("clinical_consistency", {"valid": "NOT EVALUATED"}),
                "resistance_stress_audit": evaluation.get("resistance_stress_audit", {"identified_dimensions": []}),
                "decision_stress_score": evaluation.get("decision_stress_score", {"overall_stress_score": "NOT EVALUATED"}),
                "realism_score": evaluation.get("realism_score", {"overall_realism_score": "NOT EVALUATED"})
            }

        return {
            "scenario_id": sid,
            "synthetic": True,
            "synthetic_badge": SYNTHETIC_BADGE,
            "safety_warning": "CRITICAL: This is a synthetic stress-test scenario. Never interpret as a real patient or clinical outcome.",
            "scenario_category": category,
            "generation_method": method,
            "target_blind_spot": target_blind_spot,
            "patient_context": patient_context,
            "genomic_profile": {
                "alterations": alterations,
                "cooccurring_alterations": cooccurring,
                "resistance_related_features": resistance_features
            },
            "biomarkers": biomarkers,
            "clinical_context": clinical_context,
            "synthetic_assumptions": assumptions,
            "reference_evidence": evidence,
            "uncertainty": uncertainty,
            "provenance": provenance,
            "evaluation": eval_view
        }
