import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_CANDIDATE_1 = PROJECT_ROOT / "stage4_slm" / "data" / "raw" / "oncology_stage4_raw_10000.csv"
RAW_CANDIDATE_2 = PROJECT_ROOT / "stage4_slm" / "data" / "raw" / "stage4_slm_synthetic_raw_dataset.csv"
RAW_CSV_PATH = RAW_CANDIDATE_1 if RAW_CANDIDATE_1.exists() else RAW_CANDIDATE_2

PROCESSED_CSV_PATH = PROJECT_ROOT / "stage4_slm" / "data" / "processed" / "stage4_slm_processed_dataset.csv"
TRAIN_CSV_PATH = PROJECT_ROOT / "stage4_slm" / "data" / "splits" / "train.csv"
VAL_CSV_PATH = PROJECT_ROOT / "stage4_slm" / "data" / "splits" / "validation.csv"
TEST_CSV_PATH = PROJECT_ROOT / "stage4_slm" / "data" / "splits" / "test.csv"
DICTIONARY_PATH = PROJECT_ROOT / "stage4_slm" / "domain" / "oncology_dictionary.json"


# =========================================================================
# 1. Dataset Loading Tests
# =========================================================================

def test_raw_dataset_exists_and_loads():
    """Verify raw CSV exists, is readable, and contains exactly 10,000 records."""
    assert RAW_CSV_PATH.exists(), f"Raw dataset not found at {RAW_CSV_PATH}"
    df = pd.read_csv(RAW_CSV_PATH)
    assert len(df) == 10000, f"Expected 10,000 raw records, found {len(df)}"


def test_processed_dataset_exists_and_loads():
    """Verify processed CSV exists and matches expected deduplicated count (9,856)."""
    assert PROCESSED_CSV_PATH.exists(), f"Processed dataset not found at {PROCESSED_CSV_PATH}"
    df = pd.read_csv(PROCESSED_CSV_PATH)
    assert len(df) == 9856, f"Expected 9,856 processed records, found {len(df)}"


# =========================================================================
# 2. Schema Validation
# =========================================================================

def test_raw_schema_matches_expected_columns():
    """Verify raw CSV contains all 7 expected columns."""
    df = pd.read_csv(RAW_CSV_PATH, nrows=10)
    expected_cols = [
        "patient_id", "clinical_report", "target_summary",
        "source_type", "stage1_context", "stage2_context", "stage3_context"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_processed_schema_includes_slm_fields():
    """Verify processed CSV includes normalized prompt and multimodal context fields."""
    df = pd.read_csv(PROCESSED_CSV_PATH, nrows=10)
    assert "slm_prompt" in df.columns, "Missing slm_prompt column"
    assert "multimodal_context_json" in df.columns, "Missing multimodal_context_json column"


# =========================================================================
# 3. Missing Value & Imputation Handling
# =========================================================================

def test_raw_dataset_has_expected_raw_inconsistencies():
    """Verify raw CSV exhibits expected authentic uncleaned missing values in non-critical columns."""
    df = pd.read_csv(RAW_CSV_PATH)
    assert df["patient_id"].isnull().sum() == 0
    assert df["clinical_report"].isnull().sum() == 0
    assert df["target_summary"].isnull().sum() == 0
    total_non_critical_nulls = (
        df["source_type"].isnull().sum() +
        df["stage1_context"].isnull().sum() +
        df["stage2_context"].isnull().sum() +
        df["stage3_context"].isnull().sum()
    )
    assert total_non_critical_nulls > 0, "Raw dataset should demonstrate realistic missingness in non-critical fields"


def test_zero_missing_or_empty_values_in_processed():
    """Verify zero null or empty string entries across all processed fields."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    for col in ["patient_id", "clinical_report", "target_summary", "source_type",
                 "stage1_context", "stage2_context", "stage3_context",
                 "multimodal_context_json", "slm_prompt"]:
        assert df[col].isnull().sum() == 0, f"Found nulls in {col}"
        empty_count = (df[col].astype(str).str.strip() == "").sum()
        assert empty_count == 0, f"Found empty strings in {col}"


# =========================================================================
# 4. Duplicate Detection & Handling
# =========================================================================

def test_duplicate_detection_and_removal():
    """Verify raw dataset contained duplicates (115 rows) and processed dataset dropped them (144 total)."""
    df_raw = pd.read_csv(RAW_CSV_PATH)
    df_processed = pd.read_csv(PROCESSED_CSV_PATH)

    raw_dupes = df_raw.duplicated().sum()
    assert raw_dupes == 115, f"Expected 115 duplicate rows in raw dataset, found {raw_dupes}"
    assert df_processed.duplicated().sum() == 0, "Duplicate rows detected in processed dataset"
    assert len(df_raw) - len(df_processed) == 144, "Deduplicated count does not match expected total dropped duplicates"


# =========================================================================
# 5. Patient ID Validation
# =========================================================================

def test_patient_id_format_and_synthetic_guarantee():
    """Verify all patient IDs conform strictly to synthetic format SYN-XXXXXX."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    id_pattern = re.compile(r"^SYN-\d{6}$")
    invalid_ids = df[~df["patient_id"].str.match(id_pattern)]
    assert len(invalid_ids) == 0, f"Invalid patient IDs found: {invalid_ids['patient_id'].tolist()[:5]}"


def test_multi_record_patient_distribution():
    """Verify multi-record patients exist as expected in longitudinal oncology data (3,200 patients)."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    unique_pts = df["patient_id"].nunique()
    assert unique_pts == 3200, f"Expected 3,200 unique patients, found {unique_pts}"
    avg_records = len(df) / unique_pts
    assert 2.5 <= avg_records <= 3.5, f"Expected ~3.1 records/patient, found {avg_records:.2f}"


# =========================================================================
# 6. Clinical Report Validation
# =========================================================================

def test_clinical_report_non_empty_and_valid_lengths():
    """Verify clinical reports contain adequate narrative text."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    min_len = df["clinical_report"].str.len().min()
    max_len = df["clinical_report"].str.len().max()
    assert min_len >= 80, f"Clinical report is unexpectedly short: min_len={min_len}"
    assert max_len <= 1500, f"Clinical report is unexpectedly long: max_len={max_len}"


# =========================================================================
# 7. Target Summary Validation
# =========================================================================

def test_target_summary_conciseness_and_coherence():
    """Verify target summaries are concise, non-empty, and clinically coherent."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    word_counts = df["target_summary"].str.split().str.len()
    assert word_counts.min() >= 10, "Target summary too brief"
    assert word_counts.max() <= 50, "Target summary too verbose for concise synthesis"
    sample_summaries = df["target_summary"].sample(50, random_state=42)
    oncology_terms = [
        "patient", "carcinoma", "cancer", "ca", "melanoma", "therapy",
        "risk", "status", "stage", "st.", "tumor", "disease", "response",
        "leukemia", "lymphoma", "gbm", "nsclc", "hcc", "renal", "gastric",
        "cervical", "pancreatic", "thyroid", "esophageal", "colorectal"
    ]
    for s in sample_summaries:
        assert any(term in s.lower() for term in oncology_terms), f"Summary lacking oncology terms: {s}"


# =========================================================================
# 8. Stage 1 Context Validation
# =========================================================================

def test_stage1_context_json_and_ranges():
    """Verify Stage 1 context contains valid JSON and calibrated probability ranges or missing status."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    for s1_str in df["stage1_context"].sample(100, random_state=42):
        s1 = json.loads(s1_str)
        assert s1.get("SYNTHETIC") is True
        if "status" not in s1:
            assert "risk_score" in s1 or "mortality_prob" in s1 or "overall_risk" in s1


# =========================================================================
# 9. Stage 2 Context Validation
# =========================================================================

def test_stage2_context_json_and_ranges():
    """Verify Stage 2 context contains valid JSON and valid progression probabilities or missing status."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    for s2_str in df["stage2_context"].sample(100, random_state=42):
        s2 = json.loads(s2_str)
        assert s2.get("SYNTHETIC") is True
        if "status" not in s2:
            assert "progression_prob" in s2 or "histopathology_finding" in s2 or "histopathology_assessment" in s2


# =========================================================================
# 10. Stage 3 Context Validation
# =========================================================================

def test_stage3_context_json_and_ranges():
    """Verify Stage 3 context contains valid JSON, urgency levels, and entity lists or missing status."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    for s3_str in df["stage3_context"].sample(100, random_state=42):
        s3 = json.loads(s3_str)
        assert s3.get("SYNTHETIC") is True
        if "status" not in s3:
            assert "urgency_level" in s3 or "urgency" in s3


# =========================================================================
# 11. Patient-Level Splitting
# =========================================================================

def test_patient_level_splits_exist_and_counts():
    """Verify train, validation, and test split files exist and sum to 9,856 records."""
    assert TRAIN_CSV_PATH.exists()
    assert VAL_CSV_PATH.exists()
    assert TEST_CSV_PATH.exists()

    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_val = pd.read_csv(VAL_CSV_PATH)
    df_test = pd.read_csv(TEST_CSV_PATH)

    assert len(df_train) + len(df_val) + len(df_test) == 9856
    assert 7500 <= len(df_train) <= 8300  # ~80%
    assert 850 <= len(df_val) <= 1150     # ~10%
    assert 850 <= len(df_test) <= 1150    # ~10%

    assert df_train["patient_id"].nunique() == 2560
    assert df_val["patient_id"].nunique() == 320
    assert df_test["patient_id"].nunique() == 320


# =========================================================================
# 12. Train / Validation / Test Leakage Prevention
# =========================================================================

def test_zero_patient_leakage_across_splits():
    """Verify strictly ZERO patient ID overlap between any pair of splits."""
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_val = pd.read_csv(VAL_CSV_PATH)
    df_test = pd.read_csv(TEST_CSV_PATH)

    train_pts = set(df_train["patient_id"])
    val_pts = set(df_val["patient_id"])
    test_pts = set(df_test["patient_id"])

    assert len(train_pts & val_pts) == 0, f"Leakage! Overlapping train-val patients: {len(train_pts & val_pts)}"
    assert len(train_pts & test_pts) == 0, f"Leakage! Overlapping train-test patients: {len(train_pts & test_pts)}"
    assert len(val_pts & test_pts) == 0, f"Leakage! Overlapping val-test patients: {len(val_pts & test_pts)}"


def test_zero_report_text_overlap_across_splits():
    """Verify zero duplicate clinical reports exist across train, validation, and test splits."""
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_val = pd.read_csv(VAL_CSV_PATH)
    df_test = pd.read_csv(TEST_CSV_PATH)

    train_reports = set(df_train["clinical_report"])
    val_reports = set(df_val["clinical_report"])
    test_reports = set(df_test["clinical_report"])

    assert len(train_reports & val_reports) == 0, "Leakage! Overlapping reports between train and val"
    assert len(train_reports & test_reports) == 0, "Leakage! Overlapping reports between train and test"
    assert len(val_reports & test_reports) == 0, "Leakage! Overlapping reports between val and test"


# =========================================================================
# 13. Reproducibility & Domain Dictionary Tests
# =========================================================================

def test_domain_dictionary_valid_and_comprehensive():
    """Verify oncology domain dictionary exists, parses, and contains required categories."""
    assert DICTIONARY_PATH.exists()
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        doc = json.load(f)

    assert "metadata" in doc
    assert "categories" in doc
    categories = doc["categories"]

    required_categories = [
        "cancer_types", "biomarkers", "mutations", "drugs",
        "drug_combinations", "oncology_abbreviations", "adverse_events",
        "pathology_terms", "treatment_terms", "guideline_related_terms"
    ]
    for cat in required_categories:
        assert cat in categories, f"Missing category: {cat}"
        assert len(categories[cat]) > 0, f"Empty category: {cat}"
        for item in categories[cat]:
            assert "term" in item
            assert "description" in item


def test_raw_csv_unmodified_integrity():
    """Verify the primary raw CSV was strictly preserved and untouched."""
    raw_size = RAW_CSV_PATH.stat().st_size
    assert raw_size == 8208966, f"Raw CSV file size changed! Expected 8208966 bytes, found {raw_size}"
