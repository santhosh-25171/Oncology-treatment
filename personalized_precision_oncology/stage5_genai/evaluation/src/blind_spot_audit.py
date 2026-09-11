"""
Blind-Spot Coverage Auditor.
Verifies that each synthetic scenario targets a documented blind spot from
stage5_genai/eda_prompteng/reports/blind_spot_report.json and calculates coverage scores.
"""

from typing import Dict, Any, List, Tuple


class BlindSpotAuditor:
    """Audits blind-spot targeting and coverage alignment against the Stage 5 EDA catalog."""

    VALID_TAXONOMY = {
        "well_represented",
        "rare",
        "sparse",
        "not_observed",
        "insufficient_evidence",
        "conflicting_evidence",
        "source_limitation"
    }

    def __init__(self, blind_spots_catalog: List[Dict[str, Any]]):
        self.catalog = {bs["blind_spot_id"]: bs for bs in blind_spots_catalog}

    def audit_blind_spot(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the target blind spot of a scenario:
        - blind_spot_targeted: ID of the targeted blind spot
        - blind_spot_supported: whether the ID exists in the catalog
        - blind_spot_type: taxonomy category from catalog
        - blind_spot_evidence: citation of reason/evidence from catalog
        - blind_spot_coverage_score: 0.0 to 5.0 rating of how effectively the scenario tests the gap
        """
        tbs = scenario.get("target_blind_spot", {})
        bs_id = tbs.get("blind_spot_id")

        if not bs_id or bs_id not in self.catalog:
            return {
                "blind_spot_targeted": bs_id or "MISSING",
                "blind_spot_supported": False,
                "blind_spot_type": "UNKNOWN",
                "blind_spot_evidence": "No matching blind spot found in Stage 5 EDA catalog.",
                "blind_spot_coverage_score": 0.0,
                "issues": [f"Blind spot ID '{bs_id}' is not documented in blind_spot_report.json."]
            }

        catalog_entry = self.catalog[bs_id]
        status = catalog_entry.get("coverage_status", "unknown")
        cat_reason = catalog_entry.get("reason", "")
        pattern = catalog_entry.get("pattern", "")

        # Scoring heuristic (0 to 5)
        # Base score 4.0 for valid matching documented blind spot
        score = 4.0

        # +1.0 if scenario has detailed synthetic assumptions and clinical progression context testing the pattern
        assumptions = scenario.get("synthetic_assumptions", [])
        progression = scenario.get("clinical_context", {}).get("progression_context", "")

        if len(assumptions) > 0 and len(progression) > 20:
            score += 1.0

        # Cap at 5.0
        score = min(5.0, score)

        return {
            "blind_spot_targeted": bs_id,
            "blind_spot_supported": True,
            "blind_spot_type": status,
            "blind_spot_evidence": f"Pattern: {pattern}. {cat_reason}",
            "blind_spot_coverage_score": round(score, 2),
            "issues": []
        }
