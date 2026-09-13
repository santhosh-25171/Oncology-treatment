"""
Seed Compliance Validator for Stage 5 Interactive Generation.
Verifies that generated synthetic oncology patients strictly adhere to user-selected seed conditions.
"""

from typing import Dict, Any, List, Tuple, Optional


class SeedComplianceValidator:
    """Audits generated patient attributes against user-specified seed inputs."""

    @staticmethod
    def validate_seed_compliance(scenario: Dict[str, Any], seed_conditions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates scenario against requested seed conditions.
        Returns:
            {
                "status": "PASS" | "REVIEW" | "FAIL" | "NOT_SPECIFIED",
                "matched_fields": [...],
                "mismatched_fields": [...],
                "details": "..."
            }
        """
        if not seed_conditions:
            return {
                "status": "NOT_SPECIFIED",
                "matched_fields": [],
                "mismatched_fields": [],
                "details": "No custom seed conditions specified; default template generation used."
            }

        matched = []
        mismatched = []

        p_ctx = scenario.get("patient_context", {})
        gen_prof = scenario.get("genomic_profile", {})
        b_marks = scenario.get("biomarkers", {})

        # 1. Cancer Type check
        if "cancer_type" in seed_conditions and seed_conditions["cancer_type"]:
            req = str(seed_conditions["cancer_type"]).lower()
            actual = str(p_ctx.get("cancer_type", "")).lower()
            if req in actual or actual in req:
                matched.append("cancer_type")
            else:
                mismatched.append(f"cancer_type (requested: {req}, got: {actual})")

        # 2. Stage check
        if "stage" in seed_conditions and seed_conditions["stage"]:
            req = str(seed_conditions["stage"]).lower().replace("stage", "").strip()
            actual = str(p_ctx.get("stage", "")).lower().replace("stage", "").strip()
            if req in actual or actual in req:
                matched.append("stage")
            else:
                mismatched.append(f"stage (requested: {req}, got: {actual})")

        # 3. Sex check
        if "sex" in seed_conditions and seed_conditions["sex"]:
            req = str(seed_conditions["sex"]).lower()
            actual = str(p_ctx.get("sex", "")).lower()
            if req == actual:
                matched.append("sex")
            else:
                mismatched.append(f"sex (requested: {req}, got: {actual})")

        # 4. Histology check
        if "histology" in seed_conditions and seed_conditions["histology"]:
            req = str(seed_conditions["histology"]).lower()
            actual = str(p_ctx.get("histology", "")).lower()
            if req in actual or actual in req:
                matched.append("histology")
            else:
                mismatched.append(f"histology (requested: {req}, got: {actual})")

        # 5. Age check
        if "age" in seed_conditions and seed_conditions["age"]:
            try:
                req_age = float(seed_conditions["age"])
                actual_age = float(p_ctx.get("age", 0)) if "age" in p_ctx else None
                if actual_age is None and "age_group" in p_ctx:
                    parts = str(p_ctx["age_group"]).split("-")
                    if len(parts) == 2:
                        actual_age = (float(parts[0]) + float(parts[1])) / 2.0
                if actual_age and abs(actual_age - req_age) <= 5.0:
                    matched.append("age")
                else:
                    mismatched.append(f"age (requested: {req_age}, got: {actual_age})")
            except (ValueError, TypeError):
                pass

        # 6. Driver alteration / gene check
        if "driver_alteration" in seed_conditions and seed_conditions["driver_alteration"]:
            req_driver = str(seed_conditions["driver_alteration"]).upper()
            found = False
            for alt in gen_prof.get("alterations", []):
                gene = str(alt.get("gene", "")).upper()
                var = str(alt.get("variant", "")).upper()
                if req_driver in gene or req_driver in var:
                    found = True
                    break
            if found:
                matched.append("driver_alteration")
            else:
                mismatched.append(f"driver_alteration (requested: {req_driver} not in genomic profile)")

        # Determine compliance status
        if len(mismatched) == 0:
            status = "PASS"
            details = f"All {len(matched)} specified seed condition(s) were successfully satisfied."
        elif len(mismatched) == 1 and len(matched) >= 2:
            status = "REVIEW"
            details = f"Minor seed divergence: {', '.join(mismatched)}."
        else:
            status = "FAIL"
            details = f"Critical seed divergence: {', '.join(mismatched)}."

        return {
            "status": status,
            "matched_fields": matched,
            "mismatched_fields": mismatched,
            "details": details
        }
