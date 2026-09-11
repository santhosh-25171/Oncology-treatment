"""
Comprehensive Scenario Validator.
Performs JSON Schema validation, semantic consistency checks, synthetic labeling audits,
provenance verification, and evidence/assumption boundary enforcement.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from jsonschema import validate, ValidationError

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"
SYNTHETIC_PATIENT_SCHEMA = SCHEMAS_DIR / "synthetic_patient_schema.json"


class ScenarioValidator:
    """Validates synthetic scenarios for structural and semantic integrity."""

    def __init__(self, schema_path: Path = SYNTHETIC_PATIENT_SCHEMA):
        self.schema_path = schema_path
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def validate_schema(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate scenario structure against JSON Schema."""
        try:
            validate(instance=scenario, schema=self.schema)
            return True, []
        except ValidationError as e:
            return False, [f"Schema validation error: {e.message} at path '{'/'.join(map(str, e.path))}'"]

    def validate_semantic_consistency(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Audit clinical and genomic semantic consistency:
        - Synthetic flag must be true
        - Scenario ID format
        - Target blind spot present and valid
        - Separation of reference_evidence and synthetic_assumptions
        - Non-empty provenance
        - Absence of contradictory stage/metastasis metadata
        - Valid uncertainty level
        """
        issues = []

        # 1. Synthetic flag check
        if scenario.get("synthetic") is not True:
            issues.append("Missing mandatory 'synthetic': true flag.")

        # 2. Scenario ID
        sid = scenario.get("scenario_id", "")
        if not sid.startswith("EDGE_"):
            issues.append(f"Invalid scenario_id '{sid}'; must start with 'EDGE_'.")

        # 3. Target blind spot
        tbs = scenario.get("target_blind_spot", {})
        if not tbs.get("blind_spot_id"):
            issues.append("Missing 'blind_spot_id' in target_blind_spot.")
        if not tbs.get("reason"):
            issues.append("Missing 'reason' in target_blind_spot.")

        # 4. Separation of evidence vs assumptions
        ref_ev = scenario.get("reference_evidence", [])
        syn_ass = scenario.get("synthetic_assumptions", [])
        if not ref_ev or len(ref_ev) == 0:
            issues.append("Missing 'reference_evidence'; historical grounding must be provided.")
        if not syn_ass or len(syn_ass) == 0:
            issues.append("Missing 'synthetic_assumptions'; stress-test assumptions must be explicitly isolated.")

        # Check for cross-contamination of labels
        for ev in ref_ev:
            if "synthetic assumption" in ev.lower():
                issues.append("reference_evidence contains 'synthetic assumption'; evidence boundaries violated.")
        for ass in syn_ass:
            if "historically proven patient" in ass.lower():
                issues.append("synthetic_assumptions claims historical proof; evidence boundaries violated.")

        # 5. Uncertainty
        unc = scenario.get("uncertainty", {})
        if unc.get("level") not in ["low", "medium", "high"]:
            issues.append(f"Invalid uncertainty level '{unc.get('level')}'; must be low, medium, or high.")
        if not unc.get("reason"):
            issues.append("Missing uncertainty reason.")

        # 6. Provenance
        prov = scenario.get("provenance", {})
        if not prov.get("reference_sources"):
            issues.append("Missing 'reference_sources' in provenance.")
        if not prov.get("generation_timestamp"):
            issues.append("Missing 'generation_timestamp' in provenance.")
        if not prov.get("generation_method"):
            issues.append("Missing 'generation_method' in provenance.")

        # 7. Semantic clinical sanity check
        pcontext = scenario.get("patient_context", {})
        stage = pcontext.get("stage", "")
        clin = scenario.get("clinical_context", {})
        prog = clin.get("progression_context", "").lower()
        if "stage i" in stage.lower() and "multifocal hepatic" in prog:
            issues.append("Contradictory clinical context: Early Stage I cancer described with distant visceral metastases.")

        return len(issues) == 0, issues

    def validate_scenario(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Run full schema and semantic validation."""
        s_valid, s_issues = self.validate_schema(scenario)
        if not s_valid:
            return False, s_issues

        c_valid, c_issues = self.validate_semantic_consistency(scenario)
        if not c_valid:
            return False, c_issues

        return True, []

    def validate_scenario_set(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate an entire collection of scenarios for uniqueness and coverage."""
        total = len(scenarios)
        valid_count = 0
        rejected_count = 0
        issues_log = []
        seen_ids = set()
        duplicate_ids = set()

        for sc in scenarios:
            sid = sc.get("scenario_id")
            if sid in seen_ids:
                duplicate_ids.add(sid)
            seen_ids.add(sid)

            is_valid, errors = self.validate_scenario(sc)
            if is_valid:
                valid_count += 1
            else:
                rejected_count += 1
                issues_log.append({"scenario_id": sid, "errors": errors})

        if duplicate_ids:
            issues_log.append({"error": f"Duplicate scenario IDs detected: {list(duplicate_ids)}"})

        status = "PASS" if rejected_count == 0 and len(duplicate_ids) == 0 else "FAIL"

        return {
            "status": status,
            "total_evaluated": total,
            "valid_scenarios": valid_count,
            "rejected_scenarios": rejected_count,
            "duplicate_ids": list(duplicate_ids),
            "issues": issues_log
        }
