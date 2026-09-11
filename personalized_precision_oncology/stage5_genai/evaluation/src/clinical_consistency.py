"""
Clinical Consistency Auditor.
Audits logical consistency across cancer type, stage, prior treatment line,
progression context, and biomarker measurements.
"""

from typing import Dict, Any, List, Tuple


class ClinicalConsistencyAuditor:
    """Audits clinical logic and detects contradictions without making medical diagnoses."""

    def audit_clinical_consistency(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Checks for internal logical contradictions:
        - Early stage with distant metastases
        - Treatment-naive marked with multiple prior lines of therapy
        - Biomarker contradiction (e.g. TPS > 100 or negative values)
        - Resistance mechanism claimed without corresponding treatment context
        """
        issues = []
        p_context = scenario.get("patient_context", {})
        stage = str(p_context.get("stage", "")).upper()
        prior_tx = str(p_context.get("prior_treatment_context", "")).lower()

        c_context = scenario.get("clinical_context", {})
        progression = str(c_context.get("progression_context", "")).lower()
        disease_status = str(c_context.get("disease_status", "")).lower()

        # 1. Stage vs metastatic presentation
        # Only flag if Stage is early (Stage I or Stage II) and has distant metastases without recurrence/adjuvant context
        is_early_stage = ("STAGE I" in stage or "STAGE II" in stage) and ("STAGE IV" not in stage and "STAGE III" not in stage)
        if is_early_stage:
            distant_indicators = ["hepatic", "cerebral", "brain", "osseous", "bone metastasis", "distant visceral"]
            if any(di in progression for di in distant_indicators) and "recurrence" not in progression and "adjuvant" not in prior_tx:
                issues.append(f"Contradiction: Stage is '{stage}' (Early Stage) but progression context notes distant metastases without recurrence context.")

        # 2. Treatment history vs status
        if "naive" in prior_tx and ("progressed" in prior_tx or "resistant" in prior_tx or "cycle" in prior_tx):
            issues.append(f"Contradiction in treatment history: '{prior_tx}' combines treatment-naive with prior therapy progression.")

        # 3. Biomarker range checks
        biomarkers = scenario.get("biomarkers", {})
        if "pdl1_tps" in biomarkers:
            tps = biomarkers["pdl1_tps"]
            if isinstance(tps, (int, float)):
                if tps < 0 or tps > 100:
                    issues.append(f"Invalid PD-L1 TPS value '{tps}'; must be between 0 and 100%.")

        if "tmb" in biomarkers:
            tmb = biomarkers["tmb"]
            if isinstance(tmb, (int, float)):
                if tmb < 0 or tmb > 500:
                    issues.append(f"Invalid TMB value '{tmb}'; must be non-negative.")

        # 4. Resistance alteration vs prior treatment context
        res_features = scenario.get("genomic_profile", {}).get("resistance_related_features", [])
        if res_features and len(res_features) > 0:
            # If explicit tertiary resistance (like C797S, G1202R) is present, check prior treatment mentions targeted drug or trial
            for rf in res_features:
                variant = str(rf.get("variant", "")).lower()
                gene = str(rf.get("gene", "")).lower()
                if "c797s" in variant and not ("osimertinib" in prior_tx or "tki" in prior_tx or "erlotinib" in prior_tx or "trial" in prior_tx):
                    issues.append("EGFR C797S tertiary resistance claimed but prior treatment context does not mention EGFR TKI exposure.")
                if "g1202r" in variant and not ("alectinib" in prior_tx or "crizotinib" in prior_tx or "brigatinib" in prior_tx or "alk" in prior_tx):
                    issues.append("ALK G1202R solvent-front resistance claimed but prior treatment context does not mention ALK TKI exposure.")

        is_valid = len(issues) == 0
        return is_valid, issues
