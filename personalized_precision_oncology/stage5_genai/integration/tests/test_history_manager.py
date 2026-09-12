"""
Tests for HistoryManager.
Verifies append-only persistence of evaluation results, run numbering,
scenario-specific retrieval, and preservation of raw inputs.
"""

from pathlib import Path
import pytest

from personalized_precision_oncology.stage5_genai.integration.src.history_manager import HistoryManager


def test_history_manager_append_and_retrieve(tmp_path):
    """Verifies appending evaluation records and retrieving them in order."""
    hist_file = tmp_path / "test_history.jsonl"
    mgr = HistoryManager(history_file=hist_file)

    eval_1 = {
        "scenario_id": "EDGE_005",
        "evaluation_status": "REVIEW",
        "decision_stress_score": {"overall_stress_score": 3.8},
        "realism_score": {"overall_realism_score": 4.5},
        "blind_spot_targeted": "BS005"
    }
    rec_1 = mgr.record_evaluation(eval_1)

    assert rec_1["run_number"] == 1
    assert rec_1["run_id"] == "RUN_EDGE_005_001"
    assert rec_1["evaluation_status"] == "REVIEW"

    # Second evaluation of same scenario
    eval_2 = {
        "scenario_id": "EDGE_005",
        "evaluation_status": "PASS",
        "decision_stress_score": {"overall_stress_score": 4.1},
        "realism_score": {"overall_realism_score": 4.8},
        "blind_spot_targeted": "BS005"
    }
    rec_2 = mgr.record_evaluation(eval_2)

    assert rec_2["run_number"] == 2
    assert rec_2["run_id"] == "RUN_EDGE_005_002"
    assert rec_2["evaluation_status"] == "PASS"

    # Check scenario history
    hist = mgr.get_history_for_scenario("EDGE_005")
    assert len(hist) == 2
    assert hist[0]["evaluation_status"] == "REVIEW"
    assert hist[1]["evaluation_status"] == "PASS"


def test_history_manager_batch_recording(tmp_path):
    """Verifies batch appending to history log."""
    hist_file = tmp_path / "batch_history.jsonl"
    mgr = HistoryManager(history_file=hist_file)

    batch_data = {
        "results": [
            {"scenario_id": "EDGE_001", "evaluation_status": "PASS"},
            {"scenario_id": "EDGE_002", "evaluation_status": "PASS"},
            {"scenario_id": "EDGE_003", "evaluation_status": "REVIEW"}
        ]
    }
    recorded = mgr.record_batch(batch_data)
    assert len(recorded) == 3

    all_hist = mgr.get_all_history()
    assert len(all_hist) == 3


def test_history_does_not_mutate_scenarios_file():
    """Confirms that history writes to history directory and never touches the raw synthetic scenario file."""
    genai_scenarios = (
        Path(__file__).resolve().parent.parent.parent
        / "genai"
        / "scenarios"
        / "synthetic_edge_cases.jsonl"
    )
    mtime_before = genai_scenarios.stat().st_mtime

    mgr = HistoryManager()
    mgr.record_evaluation({"scenario_id": "EDGE_001", "evaluation_status": "PASS"})

    mtime_after = genai_scenarios.stat().st_mtime
    assert mtime_before == mtime_after, "Raw synthetic scenarios file must not be modified by history operations"
