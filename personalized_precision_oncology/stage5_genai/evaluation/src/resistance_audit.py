"""
Resistance Stress Auditor.
Audits the presence of the 10 distinct resistance-stress dimensions (A to J)
and profiles resistance mechanisms.
"""

from typing import Dict, Any, List, Set


class ResistanceStressAuditor:
    """Evaluates resistance stress dimensions without double-counting."""

    DIMENSIONS = {
        "A": "Single rare resistance signal",
        "B": "Compound mutation resistance",
        "C": "Multiple resistance mechanisms",
        "D": "Conflicting biomarkers",
        "E": "Sparse evidence",
        "F": "Treatment failure despite favorable signal",
        "G": "Unobserved mutation combination",
        "H": "Multiple competing genomic signals",
        "I": "High uncertainty",
        "J": "Incomplete genomic coverage"
    }

    def audit_resistance_stress(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detects targeted dimensions for the scenario:
        A: single rare resistance or variant
        B: compound alterations (e.g. cis/trans, dual co-drivers)
        C: multiple mechanisms (e.g. on-target + bypass)
        D: conflicting biomarkers (e.g. TMB-high vs PD-L1 0%)
        E: sparse evidence (NTRK, NRG1 fusions)
        F: treatment failure / progression after initial response
        G: unobserved combination in reference
        H: competing drivers (e.g. EGFR + KRAS)
        I: high uncertainty
        J: source limitations or missing variables
        """
        category = scenario.get("scenario_category", "")
        blind_spot = scenario.get("target_blind_spot", {})
        bs_id = blind_spot.get("blind_spot_id", "")
        genomic_profile = scenario.get("genomic_profile", {})
        alterations = genomic_profile.get("alterations", [])
        cooccurrences = genomic_profile.get("cooccurring_alterations", [])
        resistance = genomic_profile.get("resistance_related_features", [])
        uncertainty = scenario.get("uncertainty", {})
        prior_tx = scenario.get("patient_context", {}).get("prior_treatment_context", "").lower()
        biomarkers = scenario.get("biomarkers", {})

        identified_dims: Set[str] = set()

        # Dimension A: Single rare resistance signal
        if "rare" in category or "BS001" in bs_id or "BS002" in bs_id or "BS004" in bs_id or "BS005" in bs_id:
            identified_dims.add("A")

        # Dimension B: Compound mutation resistance
        if "compound" in category or len(cooccurrences) > 1 or len(resistance) > 1 or "cis" in str(resistance):
            identified_dims.add("B")

        # Dimension C: Multiple resistance mechanisms (e.g. tertiary + bypass, or lineage switch)
        if len(resistance) >= 2 or "SCLC" in str(resistance) or "MET" in str(resistance) and "EGFR" in str(resistance):
            identified_dims.add("C")

        # Dimension D: Conflicting biomarkers
        if "conflict" in category or "BS023" in bs_id:
            identified_dims.add("D")

        # Dimension E: Sparse evidence
        if "sparse" in category or "BS007" in bs_id or "BS010" in bs_id:
            identified_dims.add("E")

        # Dimension F: Treatment failure despite initial response
        if "progression" in prior_tx or "refractory" in prior_tx or "acquired" in str(scenario.get("clinical_context")):
            identified_dims.add("F")

        # Dimension G: Unobserved mutation combination
        if any("synthetic_combination_not_observed_in_reference" in str(a) for a in alterations + cooccurrences + resistance):
            identified_dims.add("G")

        # Dimension H: Multiple competing genomic signals
        if len(alterations) + len(cooccurrences) >= 2:
            genes = {a.get("gene") for a in alterations + cooccurrences if a.get("gene")}
            if len(genes) >= 2:
                identified_dims.add("H")

        # Dimension I: High uncertainty
        if uncertainty.get("level") == "high":
            identified_dims.add("I")

        # Dimension J: Incomplete genomic coverage / source limitation
        if "BS021" in bs_id or "source_limitation" in str(biomarkers) or "BS024" in bs_id:
            identified_dims.add("J")

        # Wildcard encompasses almost all dimensions
        if category == "wildcard":
            identified_dims.update(["B", "C", "D", "F", "G", "H", "I"])

        sorted_dims = sorted(list(identified_dims))
        named_dims = [f"{d}: {self.DIMENSIONS[d]}" for d in sorted_dims]

        profile_summary = f"{category.replace('_', ' ').title()} targeting {bs_id} ({len(sorted_dims)} stress dimensions)"

        return {
            "identified_dimensions": named_dims,
            "dimension_count": len(sorted_dims),
            "resistance_mechanism_profile": profile_summary
        }
