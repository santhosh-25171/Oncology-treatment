"""
Stage 4 SLM — EDA Verification Test Suite
Role: Stage 4 EDA Engineer
Project: Personalized Precision Medicine for Oncology Treatment Optimization

Validates:
1. Generation and integrity of the Stage 4 EDA report (stage4_slm_eda_report.md).
2. Generation and file integrity of all 10 required EDA publication plots.
3. Maximum sequence token fit within the recommended 512-token context window.
4. Strict zero patient leakage verification across train, validation, and test splits.
"""

from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EDA_DIR = PROJECT_ROOT / "stage4_slm" / "eda"
PLOTS_DIR = EDA_DIR / "plots"
REPORT_PATH = EDA_DIR / "stage4_slm_eda_report.md"

TRAIN_PATH = PROJECT_ROOT / "stage4_slm" / "data" / "splits" / "train.csv"
VAL_PATH = PROJECT_ROOT / "stage4_slm" / "data" / "splits" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "stage4_slm" / "data" / "splits" / "test.csv"


def test_eda_report_exists_and_populated():
    """Verify stage4_slm_eda_report.md exists and has comprehensive content (> 10KB)."""
    assert REPORT_PATH.exists(), f"EDA report not found at {REPORT_PATH}"
    size = REPORT_PATH.stat().st_size
    assert size > 10000, f"EDA report appears truncated: {size} bytes"

    content = REPORT_PATH.read_text(encoding="utf-8")
    assert "## 1. Executive Summary" in content
    assert "## 5. SLM Prompt & Context Length Analysis" in content
    assert "## 11. Data Leakage & Similarity Audit" in content
    assert "## 14. SLM Context-Window Recommendation" in content
    assert "## 15. Training-Readiness Assessment" in content


def test_all_ten_plots_exist_and_valid():
    """Verify all 10 EDA visualization PNG files exist and are non-empty."""
    expected_plots = [
        "01_clinical_report_length_distribution.png",
        "02_target_summary_length_distribution.png",
        "03_slm_prompt_token_distribution.png",
        "04_stage1_probability_distributions.png",
        "05_stage2_probability_distributions.png",
        "06_stage3_urgency_distribution.png",
        "07_source_type_distribution.png",
        "08_records_per_patient_distribution.png",
        "09_train_val_test_comparison.png",
        "10_top_oncology_entities.png"
    ]
    for plot_name in expected_plots:
        plot_file = PLOTS_DIR / plot_name
        assert plot_file.exists(), f"Missing required EDA plot: {plot_name}"
        assert plot_file.stat().st_size > 15000, f"Plot {plot_name} file size unexpectedly small"


def test_eda_context_window_fits_512():
    """Verify prompt words and approximate tokens fit well within the 512 token budget."""
    df_train = pd.read_csv(TRAIN_PATH)
    max_prompt_words = df_train["slm_prompt"].str.split().str.len().max()
    max_summary_words = df_train["target_summary"].str.split().str.len().max()
    # Prompt words + summary words should be < 200 words
    assert max_prompt_words + max_summary_words < 250, "Sequence length exceeds 512-token context feasibility"


def test_eda_reconfirms_zero_patient_leakage():
    """Verify zero patient ID overlap across splits in EDA inspection."""
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    train_pts = set(df_train["patient_id"])
    val_pts = set(df_val["patient_id"])
    test_pts = set(df_test["patient_id"])

    assert len(train_pts & val_pts) == 0
    assert len(train_pts & test_pts) == 0
    assert len(val_pts & test_pts) == 0
