"""
Tests for ScenarioLoader.
Verifies correct JSONL loading, schema compliance, duplicate rejection,
and strict requirement for synthetic=true.
"""

import json
from pathlib import Path
import pytest

from personalized_precision_oncology.stage5_genai.integration.src.scenario_loader import ScenarioLoader


def test_loader_loads_official_scenarios():
    """Verifies that all 20 official scenarios load cleanly with zero rejections."""
    loader = ScenarioLoader()
    valid, rejected = loader.load_scenarios()

    assert len(valid) == 20, f"Expected exactly 20 valid scenarios, found {len(valid)}"
    assert len(rejected) == 0, f"Expected 0 rejections, found: {rejected}"

    # Confirm all have synthetic=true and EDGE_ prefix
    for sc in valid:
        assert sc.get("synthetic") is True
        assert sc.get("scenario_id", "").startswith("EDGE_")
        assert "provenance" in sc
        assert "target_blind_spot" in sc


def test_loader_get_scenario_by_id():
    """Verifies fetching an existing and non-existing scenario by ID."""
    loader = ScenarioLoader()
    sc = loader.get_scenario_by_id("EDGE_001")
    assert sc is not None
    assert sc["scenario_id"] == "EDGE_001"
    assert sc["synthetic"] is True

    sc_none = loader.get_scenario_by_id("NON_EXISTENT_ID")
    assert sc_none is None


def test_loader_rejects_missing_or_false_synthetic_flag(tmp_path):
    """Verifies that records with synthetic=false or missing are strictly rejected."""
    bad_file = tmp_path / "bad_synthetic.jsonl"
    loader_ref = ScenarioLoader()
    valid_sample, _ = loader_ref.load_scenarios()
    valid_record = valid_sample[0]

    # Record with synthetic=False
    rec_false = dict(valid_record)
    rec_false["scenario_id"] = "EDGE_998"
    rec_false["synthetic"] = False

    # Record with missing synthetic
    rec_missing = dict(valid_record)
    rec_missing["scenario_id"] = "EDGE_999"
    rec_missing.pop("synthetic", None)

    with open(bad_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec_false) + "\n")
        f.write(json.dumps(rec_missing) + "\n")

    loader = ScenarioLoader(scenario_path=bad_file)
    valid, rejected = loader.load_scenarios()

    assert len(valid) == 0
    assert len(rejected) == 2
    for r in rejected:
        assert r["error_type"] == "SYNTHETIC_FLAG_VIOLATION"


def test_loader_detects_duplicate_scenario_ids(tmp_path):
    """Verifies that duplicate scenario IDs on different lines are detected and rejected."""
    dup_file = tmp_path / "dup_scenarios.jsonl"
    loader_ref = ScenarioLoader()
    valid_sample, _ = loader_ref.load_scenarios()
    rec = valid_sample[0]

    with open(dup_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.write(json.dumps(rec) + "\n")  # exact same ID

    loader = ScenarioLoader(scenario_path=dup_file)
    valid, rejected = loader.load_scenarios()

    assert len(valid) == 1
    assert len(rejected) == 1
    assert rejected[0]["error_type"] == "DUPLICATE_SCENARIO_ID"


def test_loader_handles_malformed_json(tmp_path):
    """Verifies that invalid JSON syntax does not crash the loader and is logged in rejections."""
    malformed_file = tmp_path / "malformed.jsonl"
    with open(malformed_file, "w", encoding="utf-8") as f:
        f.write('{"scenario_id": "EDGE_001", "synthetic": true, broken_json\n')

    loader = ScenarioLoader(scenario_path=malformed_file)
    valid, rejected = loader.load_scenarios()

    assert len(valid) == 0
    assert len(rejected) == 1
    assert rejected[0]["error_type"] == "JSON_PARSE_ERROR"
