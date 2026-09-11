"""
Unit tests for BlindSpotAuditor.
"""

import copy
import pytest
from personalized_precision_oncology.stage5_genai.evaluation.src.blind_spot_audit import BlindSpotAuditor
from personalized_precision_oncology.stage5_genai.evaluation.src.scenario_loader import ScenarioLoader


def test_blind_spot_auditor_all_scenarios_supported():
    loader = ScenarioLoader()
    auditor = BlindSpotAuditor(loader.get_blind_spots())
    scenarios = loader.get_scenarios()

    for sc in scenarios:
        res = auditor.audit_blind_spot(sc)
        assert res["blind_spot_supported"] is True
        assert res["blind_spot_coverage_score"] >= 4.0
        assert res["blind_spot_type"] in BlindSpotAuditor.VALID_TAXONOMY


def test_blind_spot_auditor_detects_unsupported_id():
    loader = ScenarioLoader()
    auditor = BlindSpotAuditor(loader.get_blind_spots())
    sc = copy.deepcopy(loader.get_scenarios()[0])
    sc["target_blind_spot"]["blind_spot_id"] = "BS999_NONEXISTENT"

    res = auditor.audit_blind_spot(sc)
    assert res["blind_spot_supported"] is False
    assert res["blind_spot_coverage_score"] == 0.0
    assert len(res["issues"]) > 0
