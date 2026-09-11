"""
Unit tests for blind spot detection report and taxonomy validation.
"""

import json
import pytest
from pathlib import Path
from jsonschema import validate

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
SCHEMAS_DIR = BASE_DIR / "schemas"
BLIND_SPOT_REPORT_JSON = REPORTS_DIR / "blind_spot_report.json"
BLIND_SPOT_SCHEMA_JSON = SCHEMAS_DIR / "blind_spot_schema.json"

VALID_TAXONOMY = {
    "well_represented",
    "rare",
    "sparse",
    "not_observed",
    "insufficient_evidence",
    "conflicting_evidence",
    "source_limitation"
}


def test_blind_spot_report_schema_compliance():
    """Verify blind_spot_report.json complies with blind_spot_schema.json."""
    assert BLIND_SPOT_REPORT_JSON.exists(), "blind_spot_report.json missing!"
    assert BLIND_SPOT_SCHEMA_JSON.exists(), "blind_spot_schema.json missing!"

    with open(BLIND_SPOT_REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(BLIND_SPOT_SCHEMA_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Validates without throwing ValidationError
    validate(instance=data, schema=schema)


def test_taxonomy_consistency():
    """Verify all blind spots use valid taxonomy labels."""
    with open(BLIND_SPOT_REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for bs in data["blind_spots"]:
        status = bs["coverage_status"]
        assert status in VALID_TAXONOMY, f"Invalid taxonomy: {status}"


def test_unobserved_classification_wording():
    """Verify unobserved alterations are labeled not_observed_in_reference_baseline rather than biologically absent."""
    with open(BLIND_SPOT_REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for bs in data["blind_spots"]:
        if bs["coverage_status"] == "not_observed":
            # Must NOT claim biologically absent
            assert "biologically absent" not in bs["evidence_status"].lower()
            assert "biologically absent" not in bs["reason"].lower()
            # Must mention reference baseline
            assert "not observed in reference baseline" in bs["reason"].lower() or "not_observed_in_reference_baseline" in bs["evidence_status"].lower()


def test_summary_count_integrity():
    """Verify summary counts match array elements."""
    with open(BLIND_SPOT_REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    blind_spots = data["blind_spots"]
    assert summary["total_blind_spots"] == len(blind_spots)
    assert summary["rare_patterns"] == sum(1 for b in blind_spots if b["coverage_status"] == "rare")
    assert summary["sparse_patterns"] == sum(1 for b in blind_spots if b["coverage_status"] == "sparse")
    assert summary["not_observed_patterns"] == sum(1 for b in blind_spots if b["coverage_status"] == "not_observed")
    assert summary["conflicting_evidence_patterns"] == sum(1 for b in blind_spots if b["coverage_status"] == "conflicting_evidence")
    assert summary["source_limitation_patterns"] == sum(1 for b in blind_spots if b["coverage_status"] == "source_limitation")
