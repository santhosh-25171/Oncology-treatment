"""
Unit tests for data cleaning and standardization pipeline.
Verifies:
- duplicate removal
- invalid-value detection
- missing-value handling
- rare mutation preservation
- empty dataset handling
- malformed input handling
"""

import pytest
import pandas as pd
import numpy as np
from personalized_precision_oncology.stage5_genai.data_engineering.src.ingestion import OncologyDataIngestion
from personalized_precision_oncology.stage5_genai.data_engineering.src.cleaning import OncologyDataCleaner, to_snake_case
from personalized_precision_oncology.stage5_genai.data_engineering.src.config import UNAVAILABLE_SENTINEL


def test_snake_case_conversion():
    assert to_snake_case("bcr_patient_barcode") == "bcr_patient_barcode"
    assert to_snake_case("Age at Index") == "age_at_index"
    assert to_snake_case("AJCC Pathologic Stage") == "ajcc_pathologic_stage"
    assert to_snake_case("PD-L1 TPS (%)") == "pd_l1_tps"


def test_duplicate_removal():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    # Patient IDs must be unique
    assert cohort["patient_id"].is_unique
    assert cleaner.metrics["duplicates_removed"] >= 2


def test_invalid_value_detection():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    # Verify TCGA-ERR-9999 (age -5, Stage X) was quarantined and not included in cleaned cohort
    assert "TCGA-ERR-9999" not in cohort["patient_id"].values
    assert cleaner.metrics["invalid_records_removed_or_quarantined"] >= 1


def test_rare_mutation_preservation():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    all_muts = set(cohort["primary_mutation"].unique())
    for sm in cohort["secondary_resistance_mutation"].unique():
        if pd.notna(sm) and sm not in [UNAVAILABLE_SENTINEL, "None"]:
            for part in str(sm).split(";"):
                all_variants = part.split(":")[-1].strip()
                all_muts.add(all_variants)

    # Critical rare resistance mutations must be retained!
    assert "p.T790M" in all_muts or any("T790M" in x for x in all_muts)
    assert "p.C797S" in all_muts or any("C797S" in x for x in all_muts)
    assert "p.G1202R" in all_muts or any("G1202R" in x for x in all_muts)
    assert "p.Y99C" in all_muts or any("Y99C" in x for x in all_muts)
    assert cleaner.metrics["rare_mutations_retained"] > 0


def test_missing_value_handling():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    # In TCGA, ctDNA was not assayed in historical freeze -> must be 'not_available_in_source'
    tcga_rows = cohort[cohort["data_source"].str.contains("TCGA")]
    assert (tcga_rows["ctdna_maf_percent"] == UNAVAILABLE_SENTINEL).all()


def test_empty_dataset_handling():
    cleaner = OncologyDataCleaner()
    empty_df = pd.DataFrame()
    res_clin = cleaner.clean_tcga_clinical(empty_df)
    assert isinstance(res_clin, pd.DataFrame)
    assert len(res_clin) == 0

    res_mut = cleaner.clean_tcga_mutations(empty_df)
    assert isinstance(res_mut, pd.DataFrame)
    assert len(res_mut) == 0


def test_malformed_input_handling():
    cleaner = OncologyDataCleaner()
    malformed_df = pd.DataFrame([{
        "bcr_patient_barcode": "TCGA-BAD-001",
        "cancer_type": "LUAD",
        "age_at_index": "not_a_number",
        "gender": "UnknownGender",
        "ajcc_pathologic_stage": "Stage 999",
        "vital_status": "Unknown"
    }])
    cleaned = cleaner.clean_tcga_clinical(malformed_df)
    # The malformed record should be dropped/quarantined due to invalid age
    assert len(cleaned) == 0
    assert cleaner.metrics["invalid_records_removed_or_quarantined"] >= 1
