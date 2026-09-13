"""
Tests for Synthetic Data Isolation and Contamination Protection.
Guarantees that synthetic research scenarios are strictly separated from real/historical patient data,
enforces synthetic=true labeling, prevents silent cohort contamination, and verifies ID disjointness.
"""

import csv
import json
from pathlib import Path
import pytest

from personalized_precision_oncology.stage5_genai.integration.src.scenario_loader import ScenarioLoader
from personalized_precision_oncology.stage5_genai.integration.src.scenario_adapter import ScenarioAdapter, SYNTHETIC_BADGE
from personalized_precision_oncology.stage5_genai.integration.src.dashboard_service import DashboardService


COHORT_CSV = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "stage5_genai"
    / "data_engineering"
    / "processed"
    / "cleaned_cohort.csv"
)

BASELINE_JSONL = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "stage5_genai"
    / "data_engineering"
    / "processed"
    / "genai_reference_baseline.jsonl"
)


def test_synthetic_scenarios_cannot_be_mixed_with_real_patients(tmp_path):
    """
    Explicit verification of mandatory contamination protection:
    1. Synthetic scenarios are identified using synthetic=true.
    2. Synthetic scenarios are kept separate from historical/reference patient records.
    3. Synthetic scenarios cannot silently enter the real patient cohort.
    4. Dashboard synthetic views only consume synthetic scenario inputs.
    5. Synthetic scenario IDs are not treated as real patient IDs.
    6. The synthetic label is preserved during adapter transformation.
    """
    loader = ScenarioLoader()
    valid_scenarios, rejected = loader.load_scenarios()

    assert len(valid_scenarios) > 0, "No synthetic scenarios loaded"

    # 1. Synthetic scenarios are identified using synthetic=true
    for sc in valid_scenarios:
        assert sc.get("synthetic") is True, f"Scenario {sc.get('scenario_id')} must have synthetic: True"

    # Read historical/reference cohort IDs
    historical_patient_ids = set()
    if COHORT_CSV.exists():
        with open(COHORT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("patient_id")
                if pid:
                    historical_patient_ids.add(pid)

    assert len(historical_patient_ids) > 0, "Cleaned historical cohort must not be empty"

    # 2 & 5. Verify complete ID disjointness: synthetic IDs are NOT real patient IDs
    synthetic_ids = {sc["scenario_id"] for sc in valid_scenarios}
    overlap = synthetic_ids.intersection(historical_patient_ids)
    assert len(overlap) == 0, f"Critical contamination: Synthetic IDs overlap with real patient cohort: {overlap}"

    # Confirm ID prefix distinction
    for sid in synthetic_ids:
        assert sid.startswith("EDGE_"), f"Synthetic ID {sid} must have EDGE_ prefix"
    for pid in historical_patient_ids:
        assert not pid.startswith("EDGE_"), f"Historical patient ID {pid} must not have synthetic prefix EDGE_"

    # 3. Synthetic scenarios cannot silently enter the real patient cohort
    # Verify that attempting to load a real patient into ScenarioLoader fails due to missing synthetic: true
    sample_real_patient = {
        "scenario_id": "TCGA-44-2655",
        "patient_context": {"age_group": "65-69", "sex": "Male", "cancer_type": "NSCLC", "histology": "Adenocarcinoma", "stage": "Stage IB"},
        "genomic_profile": {"alterations": [{"gene": "KRAS", "variant": "p.G12C"}]},
        # Notice: synthetic flag intentionally omitted or false
    }
    test_file = tmp_path / "real_patient_attempt.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_real_patient) + "\n")

    test_loader = ScenarioLoader(scenario_path=test_file)
    loaded_valid, loaded_rejected = test_loader.load_scenarios()
    assert len(loaded_valid) == 0, "Real patient record must not be accepted by ScenarioLoader"
    assert len(loaded_rejected) == 1
    assert loaded_rejected[0]["error_type"] == "SYNTHETIC_FLAG_VIOLATION"

    # 4. Dashboard synthetic views only consume synthetic scenario inputs
    service = DashboardService()
    summary = service.get_summary()
    assert summary["synthetic_scenarios_only"] is True
    assert summary["synthetic_badge"] == "[SYNTHETIC TEST SCENARIOS - RESEARCH ONLY]"

    # 6. The synthetic label is strictly preserved during adapter transformation
    for sc in valid_scenarios:
        item = ScenarioAdapter.to_dashboard_item(sc)
        assert item["synthetic"] is True
        assert item["synthetic_badge"] == SYNTHETIC_BADGE

        detail = ScenarioAdapter.to_detailed_view(sc)
        assert detail["synthetic"] is True
        assert detail["synthetic_badge"] == SYNTHETIC_BADGE
        assert "CRITICAL" in detail["safety_warning"]
        assert "NOT REAL PATIENT" in detail["safety_warning"].upper() or "NEVER INTERPRET AS A REAL PATIENT" in detail["safety_warning"].upper()


def test_blind_spot_coverage_is_target_coverage_not_pass_rate():
    """
    Verifies that blind-spot coverage explicitly represents EDA targets covered
    and is dynamically calculated, never confused with scenario pass rate.
    """
    service = DashboardService()
    summary = service.get_summary()

    # Pass count and blind spot coverage are distinct metrics
    assert "blind_spot_coverage" in summary
    assert "total_blind_spot_targets" in summary
    assert "blind_spot_coverage_display" in summary

    assert summary["blind_spot_coverage"] == 18
    assert summary["total_blind_spot_targets"] == 24
    assert summary["blind_spot_coverage_pct"] == 75.0
    assert "18/24 targets (75.0%)" in summary["blind_spot_coverage_display"]

    # Must be distinct from scenario pass count
    assert summary["passed"] in [16, 17]
    assert summary["blind_spot_coverage"] != summary["passed"]
