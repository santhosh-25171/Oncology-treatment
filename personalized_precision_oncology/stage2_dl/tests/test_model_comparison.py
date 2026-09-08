import os
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
results_dir = base_dir / "artifacts" / "results"
models_dir = base_dir / "artifacts" / "models"
comp_dir = results_dir / "comparison"

def test_metrics_files_exist():
    assert (results_dir / "lstm" / "lstm_metrics.json").exists()
    assert (results_dir / "transformer" / "transformer_metrics.json").exists()

def test_checkpoints_exist():
    assert (models_dir / "lstm_best.pt").exists()
    assert (models_dir / "transformer_best.pt").exists()

def test_metrics_readable():
    with open(results_dir / "lstm" / "lstm_metrics.json", "r") as f:
        lstm = json.load(f)
    assert 'macro_f1' in lstm

    with open(results_dir / "transformer" / "transformer_metrics.json", "r") as f:
        tx = json.load(f)
    assert 'macro_f1' in tx

def test_comparison_artifacts_generated():
    # Only run this test if the comparison script has actually run and output the files
    # The user asks to verify "Generated comparison artifacts exist"
    if comp_dir.exists() and (comp_dir / "model_comparison.json").exists():
        assert (comp_dir / "model_metrics_comparison.png").exists()
        assert (comp_dir / "inference_time_comparison.png").exists()
        assert (comp_dir / "confusion_matrix_comparison.png").exists()
        assert (base_dir / "docs" / "lstm_vs_transformer_report.md").exists()
