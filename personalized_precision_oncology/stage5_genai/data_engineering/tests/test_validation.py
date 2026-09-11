"""
Unit tests for validation pipeline and data quality auditing.
Verifies schema compliance, range checking, provenance validation, and report generation.
"""

import json
import pytest
import pandas as pd
from personalized_precision_oncology.stage5_genai.data_engineering.src.ingestion import OncologyDataIngestion
from personalized_precision_oncology.stage5_genai.data_engineering.src.cleaning import OncologyDataCleaner
from personalized_precision_oncology.stage5_genai.data_engineering.src.validation import OncologyDataValidator
from personalized_precision_oncology.stage5_genai.data_engineering.src.config import (
    DATA_QUALITY_REPORT_JSON,
    SOURCE_METADATA_CSV,
    SCHEMAS_DIR
)


def test_schema_file_existence():
    assert (SCHEMAS_DIR / "raw_source_schema.json").exists()
    assert (SCHEMAS_DIR / "cleaned_cohort_schema.json").exists()
    assert (SCHEMAS_DIR / "data_quality_report_schema.json").exists()
    assert (SCHEMAS_DIR / "genai_reference_baseline_schema.json").exists()


def test_schema_validation():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    validator = OncologyDataValidator()
    assert validator.validate_schema(cohort) is True


def test_provenance_validation():
    assert SOURCE_METADATA_CSV.exists()
    meta_df = pd.read_csv(SOURCE_METADATA_CSV)
    required_fields = [
        "source_name", "dataset_name", "source_url", "description",
        "cancer_type", "data_type", "record_count", "license_access_information",
        "retrieval_date", "provenance_reference_citation"
    ]
    for rf in required_fields:
        assert rf in meta_df.columns, f"Missing provenance column {rf}"
    assert len(meta_df) >= 5


def test_validation_execution_and_report():
    ingestion = OncologyDataIngestion()
    raw_datasets, raw_counts = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    validator = OncologyDataValidator()
    report = validator.run_all(cohort, raw_counts, cleaner.metrics)

    assert report["final_validation_status"] == "PASSED"
    assert report["validation_status"]["critical_errors"] == 0
    assert report["validation_status"]["total_checks_passed"] >= 5
    assert report["rare_mutation_preservation_audit"]["audit_passed"] is True
    assert DATA_QUALITY_REPORT_JSON.exists()


def test_quality_report_schema_compliance():
    with open(SCHEMAS_DIR / "data_quality_report_schema.json", "r") as sf:
        schema = json.load(sf)

    with open(DATA_QUALITY_REPORT_JSON, "r") as rf:
        report = json.load(rf)

    required_keys = schema["required"]
    for k in required_keys:
        assert k in report, f"Missing required key '{k}' in data_quality_report.json"
