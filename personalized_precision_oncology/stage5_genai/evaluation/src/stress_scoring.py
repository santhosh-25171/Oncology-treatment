"""
Decision-Stress and Evidence-Constrained Realism Scoring.
Implements the 5-dimension 0-5 Decision-Stress score and 0-5 Realism score.
"""

from typing import Dict, Any, Tuple


class StressScoringEngine:
    """Calculates Decision-Stress Score and Evidence-Constrained Scenario Realism Score."""

    @staticmethod
    def calculate_decision_stress_score(scenario: Dict[str, Any], resistance_audit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes 0-5 scores across 5 core decision stress dimensions:
        1. genomic_complexity: single driver (1-2), co-mutations (3-4), triple/multiplex (4-5)
        2. evidence_uncertainty: based on explicit uncertainty level (low=1.5, medium=3.0, high=4.5)
        3. resistance_complexity: naive (1), on-target (3), bypass/lineage switch (4.5), composite (5.0)
        4. conflicting_signals: concordant (1.0), subtle mismatch (2.5), diametric opposition (4.5-5.0)
        5. decision_ambiguity: standard of care available (1.5), off-label/basket trial (3.5), no guidance/multimodal dilemma (4.5-5.0)
        """
        category = scenario.get("scenario_category", "")
        genomic_profile = scenario.get("genomic_profile", {})
        alterations = genomic_profile.get("alterations", [])
        cooccurrences = genomic_profile.get("cooccurring_alterations", [])
        resistance = genomic_profile.get("resistance_related_features", [])
        uncertainty = scenario.get("uncertainty", {})
        u_level = uncertainty.get("level", "medium")

        # 1. Genomic Complexity
        num_alts = len(alterations) + len(cooccurrences) + len(resistance)
        if num_alts <= 1:
            genomic_complexity = 2.0
        elif num_alts == 2:
            genomic_complexity = 3.0
        elif num_alts == 3:
            genomic_complexity = 4.0
        else:
            genomic_complexity = 4.8

        if category == "compound_mutation" or "rare_mutation_combination" in category:
            genomic_complexity = max(genomic_complexity, 4.2)
        if category == "wildcard":
            genomic_complexity = 5.0

        # 2. Evidence Uncertainty
        if u_level == "low":
            evidence_uncertainty = 2.0
        elif u_level == "medium":
            evidence_uncertainty = 3.5
        else:
            evidence_uncertainty = 4.8

        # 3. Resistance Complexity
        if not resistance or len(resistance) == 0:
            resistance_complexity = 1.5 if "naive" in str(scenario.get("patient_context")).lower() else 2.5
        elif len(resistance) == 1:
            variant = str(resistance[0].get("variant", "")).lower()
            if "switch" in variant or "amplification" in variant:
                resistance_complexity = 4.2
            elif "c797s" in variant or "g1202r" in variant:
                resistance_complexity = 4.0
            else:
                resistance_complexity = 3.5
        else:
            resistance_complexity = 4.8

        if category == "wildcard":
            resistance_complexity = 5.0

        # 4. Conflicting Signals
        if "conflict" in category or "BS023" in str(scenario.get("target_blind_spot")):
            conflicting_signals = 4.8
        elif "rare_mutation_combination" in category:
            conflicting_signals = 4.0
        elif category == "wildcard":
            conflicting_signals = 4.8
        elif len(cooccurrences) > 1:
            conflicting_signals = 3.5
        else:
            conflicting_signals = 1.8

        # 5. Decision Ambiguity
        if category == "wildcard":
            decision_ambiguity = 5.0
        elif "conflict" in category or "compound_resistance" in category or "prior_treatment_resistance" in category:
            decision_ambiguity = 4.6
        elif "sparse_evidence" in category or "multi_factor_resistance" in category:
            decision_ambiguity = 4.2
        elif "rare_mutation" in category:
            decision_ambiguity = 2.4
        else:
            decision_ambiguity = 3.2

        overall_stress = round((genomic_complexity + evidence_uncertainty + resistance_complexity + conflicting_signals + decision_ambiguity) / 5.0, 2)

        if overall_stress < 1.0:
            stress_level = "weak"
        elif overall_stress < 2.0:
            stress_level = "low"
        elif overall_stress < 3.0:
            stress_level = "moderate"
        elif overall_stress < 4.0:
            stress_level = "strong"
        else:
            stress_level = "extreme"

        return {
            "genomic_complexity": round(genomic_complexity, 2),
            "evidence_uncertainty": round(evidence_uncertainty, 2),
            "resistance_complexity": round(resistance_complexity, 2),
            "conflicting_signals": round(conflicting_signals, 2),
            "decision_ambiguity": round(decision_ambiguity, 2),
            "overall_stress_score": overall_stress,
            "stress_level": stress_level
        }

    @staticmethod
    def calculate_realism_score(scenario: Dict[str, Any], gen_audit: Dict[str, Any], clin_audit_passed: bool, prov_audit_passed: bool) -> Dict[str, Any]:
        """
        Computes 0-5 Evidence-Constrained Scenario Realism Score:
        - reference_data_consistency (0-5): supported mutations and distributions
        - evidence_consistency (0-5): non-fabrication of historical claims
        - clinical_consistency (0-5): internal clinical logic
        - provenance_quality (0-5): complete source provenance
        - uncertainty_handling (0-5): explicit isolation of assumptions and uncertainty
        """
        # Reference Data Consistency
        ref_consistency = 4.5 if gen_audit.get("valid", False) else 2.0

        # Evidence Consistency (check separation)
        ref_ev = scenario.get("reference_evidence", [])
        syn_ass = scenario.get("synthetic_assumptions", [])
        if len(ref_ev) > 0 and len(syn_ass) > 0:
            ev_consistency = 4.8
        else:
            ev_consistency = 2.0

        # Clinical Consistency
        clin_consistency = 4.8 if clin_audit_passed else 1.5

        # Provenance Quality
        prov_quality = 5.0 if prov_audit_passed else 2.0

        # Uncertainty Handling
        unc = scenario.get("uncertainty", {})
        if unc.get("level") and unc.get("reason"):
            unc_handling = 4.8
        else:
            unc_handling = 2.0

        overall_realism = round((ref_consistency + ev_consistency + clin_consistency + prov_quality + unc_handling) / 5.0, 2)

        return {
            "reference_data_consistency": round(ref_consistency, 2),
            "evidence_consistency": round(ev_consistency, 2),
            "clinical_consistency": round(clin_consistency, 2),
            "provenance_quality": round(prov_quality, 2),
            "uncertainty_handling": round(unc_handling, 2),
            "overall_realism_score": overall_realism
        }
