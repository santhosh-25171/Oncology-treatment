"""
Unit tests for ClinicalConsistencyAuditor.
"""

import copy
import pytest
from personalized_precision_oncology.stage5_genai.evaluation.src.clinical_consistency import ClinicalConsistencyAuditor
from personalized_precision_oncology.stage5_genai.evaluation.src.scenario_loader import ScenarioLoader


def test_clinical_consistency_all_scenarios_valid():
    loader = ScenarioLoader()
    auditor = ClinicalConsistencyAuditor()
    scenarios = loader.get_scenarios()

    for sc in scenarios:
        valid, issues = auditor.audit_clinical_consistency(sc)
        assert valid is True, f"Scenario {sc.get('scenario_id')} had clinical issues: {issues}"


def test_clinical_consistency_detects_treatment_history_contradiction():
    loader = ScenarioLoader()
    auditor = ClinicalConsistencyAuditor()
    sc = copy.deepcopy(loader.get_scenarios()[0])
    sc["patient_context"]["prior_treatment_context"] = "Treatment naive with progression through 2 cycles chemo"

    valid, issues = auditor.audit_clinical_consistency(sc)
    assert valid is False
    assert any("contradiction in treatment history" in issue.lower() for issue in issues)


def test_clinical_consistency_detects_invalid_pdl1_percentage():
    loader = ScenarioLoader()
    auditor = ClinicalConsistencyAuditor()
    sc = copy.deepcopy(loader.get_scenarios()[0])
    sc["biomarkers"]["pdl1_tps"] = 150  # Over 100%

    valid, issues = auditor.audit_clinical_consistency(sc)
    assert valid is False
    assert any("invalid pd-l1" in issue.lower() for issue in issues)
