#!/usr/bin/env python3
"""
Automated unit and contract tests for Stage 4 SLM Evaluation.
Role: Stage 4 SLM Evaluation Engineer (Role 4)
Project: Personalized Precision Medicine for Oncology Treatment Optimization

DISCLAIMER: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.
"""

import sys
import json
from pathlib import Path
import pytest
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "stage4_slm" / "data" / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "validation.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

EVAL_DIR = PROJECT_ROOT / "stage4_slm" / "evaluation"
PRED_CSV = EVAL_DIR / "full_test_predictions.csv"
MODEL_DIR = PROJECT_ROOT / "stage4_slm" / "models" / "qwen2.5_0.5b"
ADAPTER_DIR = MODEL_DIR / "adapter"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"


def test_actual_split_counts():
    """Verify authoritative filesystem record counts in train, val, and test splits."""
    assert TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists()
    df_tr = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_te = pd.read_csv(TEST_CSV)

    assert len(df_tr) == 7896, f"Expected 7,896 train rows, got {len(df_tr)}"
    assert len(df_val) == 1010, f"Expected 1,010 validation rows, got {len(df_val)}"
    assert len(df_te) == 950, f"Expected 950 test rows, got {len(df_te)}"
    assert len(df_tr) + len(df_val) + len(df_te) == 9856


def test_no_patient_leakage():
    """Verify strictly 0% patient ID leakage across train, val, and test splits."""
    df_tr = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_te = pd.read_csv(TEST_CSV)

    pts_tr = set(df_tr["patient_id"])
    pts_val = set(df_val["patient_id"])
    pts_te = set(df_te["patient_id"])

    assert len(pts_tr & pts_val) == 0, "Patient leakage detected between train and val!"
    assert len(pts_tr & pts_te) == 0, "Patient leakage detected between train and test!"
    assert len(pts_val & pts_te) == 0, "Patient leakage detected between val and test!"


def test_no_report_leakage():
    """Verify strictly 0% identical clinical report text overlap across splits."""
    df_tr = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_te = pd.read_csv(TEST_CSV)

    reps_tr = set(df_tr["clinical_report"].dropna())
    reps_val = set(df_val["clinical_report"].dropna())
    reps_te = set(df_te["clinical_report"].dropna())

    assert len(reps_tr & reps_val) == 0, "Report text leakage between train and val!"
    assert len(reps_tr & reps_te) == 0, "Report text leakage between train and test!"
    assert len(reps_val & reps_te) == 0, "Report text leakage between val and test!"


def test_model_integrity():
    """Verify LoRA adapter artifacts and configuration contracts on disk."""
    assert (ADAPTER_DIR / "adapter_model.safetensors").exists()
    assert (ADAPTER_DIR / "adapter_config.json").exists()

    with open(ADAPTER_DIR / "adapter_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["r"] == 16
    assert cfg["lora_alpha"] == 32
    assert "q_proj" in cfg["target_modules"]
    assert "down_proj" in cfg["target_modules"]


def test_prompt_contract():
    """Verify prompt formatting contract adheres to all 4 multi-stage intelligence inputs."""
    from stage4_slm.inference.run_inference import Stage4InferenceEngine

    p = Stage4InferenceEngine.format_prompt(
        patient_id="SYN-001",
        source_type="consultation",
        clinical_report="Patient with lung cancer.",
        stage1_context={"mortality_prob": 0.25},
        stage2_context={"fused_prediction": "Stable"},
        stage3_context={"urgency_level": "Routine"}
    )
    assert "### Instruction:" in p
    assert "### Clinical Report (consultation):" in p
    assert "Stage 1 ML:" in p
    assert "Stage 2 DL:" in p
    assert "Stage 3 NLP:" in p
    assert "### Target Oncology Summary:" in p


def test_prediction_schema():
    """Verify that predictions CSV adheres to the mandatory evaluation schema."""
    assert PRED_CSV.exists(), f"Missing prediction file {PRED_CSV}"
    df = pd.read_csv(PRED_CSV)
    required_cols = [
        "patient_id", "clinical_report", "target_summary",
        "stage1_context", "stage2_context", "stage3_context",
        "generated_summary", "input_tokens", "output_tokens",
        "generation_time_seconds", "format_valid", "sentence_count",
        "entity_precision", "entity_recall", "entity_f1",
        "entity_preservation", "faithfulness_status"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required evaluation column: {col}"


def test_non_empty_output():
    """Verify that generated summaries are non-empty and have valid lengths."""
    df = pd.read_csv(PRED_CSV)
    for idx, row in df.iterrows():
        summary = str(row["generated_summary"]).strip()
        assert len(summary) > 10, f"Empty or truncated summary at row {idx}"


def test_sentence_constraint():
    """Verify that generated summaries strictly adhere to 1-2 sentence constraint."""
    df = pd.read_csv(PRED_CSV)
    for idx, row in df.iterrows():
        cnt = row["sentence_count"]
        assert cnt in [1, 2], f"Sentence count violation ({cnt}) at row {idx}"


def test_no_prompt_leakage():
    """Verify no prompt instruction or JSON formatting tokens leak into output text."""
    df = pd.read_csv(PRED_CSV)
    for idx, row in df.iterrows():
        summary = str(row["generated_summary"])
        assert "### Instruction:" not in summary, f"Prompt leakage at row {idx}"
        assert "### Target" not in summary, f"Target header leakage at row {idx}"


def test_entity_metrics():
    """Verify that entity metrics are in the valid range [0.0, 1.0]."""
    df = pd.read_csv(PRED_CSV)
    assert (df["entity_precision"] >= 0.0).all() and (df["entity_precision"] <= 1.0).all()
    assert (df["entity_recall"] >= 0.0).all() and (df["entity_recall"] <= 1.0).all()
    assert (df["entity_f1"] >= 0.0).all() and (df["entity_f1"] <= 1.0).all()


def test_faithfulness_metrics():
    """Verify that faithfulness status values belong strictly to authorized enum."""
    df = pd.read_csv(PRED_CSV)
    valid_statuses = {"FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED_CLAIM", "CONTRADICTORY"}
    for status in df["faithfulness_status"]:
        assert status in valid_statuses, f"Invalid faithfulness status: {status}"


def test_stage123_compatibility():
    """Verify that the Stage 123 compatibility report exists and contains adaptation specs."""
    compat_path = PROJECT_ROOT / "stage4_slm" / "data" / "processed" / "stage123_dataset_compatibility_report.md"
    assert compat_path.exists(), f"Compatibility report missing at {compat_path}"
    content = compat_path.read_text(encoding="utf-8")
    assert "READY WITH ADAPTATION" in content
    assert "adapt_stage123_to_stage4_context" in content


def test_latency_measurement():
    """Verify that latency benchmark file exists and contains measured values."""
    lat_csv = EVAL_DIR / "latency_benchmark.csv"
    assert lat_csv.exists(), f"Missing latency benchmarks at {lat_csv}"
    df = pd.read_csv(lat_csv)
    assert len(df) >= 5
    assert "Mean Generation Latency (s)" in df["benchmark_metric"].values
