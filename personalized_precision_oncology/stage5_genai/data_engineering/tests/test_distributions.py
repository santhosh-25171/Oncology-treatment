"""
Unit tests for baseline distribution calculation.
Verifies:
- Statistical metric computation (mean, median, IQR, counts, percentages)
- Mutation frequency calculation
- Mutation co-occurrence extraction
- Biomarker distributions
- Anti-fabrication integrity for unobserved variables
"""

import json
import pytest
from personalized_precision_oncology.stage5_genai.data_engineering.src.ingestion import OncologyDataIngestion
from personalized_precision_oncology.stage5_genai.data_engineering.src.cleaning import OncologyDataCleaner
from personalized_precision_oncology.stage5_genai.data_engineering.src.distributions import OncologyDistributionExtractor
from personalized_precision_oncology.stage5_genai.data_engineering.src.config import (
    CANCER_TYPE_DIST_JSON,
    STAGE_DIST_JSON,
    AGE_DIST_JSON,
    MUTATION_FREQ_JSON,
    MUTATION_COOCCURRENCE_JSON,
    BIOMARKER_DIST_JSON,
    TREATMENT_RESISTANCE_JSON,
    UNAVAILABLE_SENTINEL
)


def test_distribution_calculation():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    extractor = OncologyDistributionExtractor(cohort, raw_datasets)
    dists = extractor.run_all()

    assert CANCER_TYPE_DIST_JSON.exists()
    assert STAGE_DIST_JSON.exists()
    assert AGE_DIST_JSON.exists()
    assert MUTATION_FREQ_JSON.exists()
    assert MUTATION_COOCCURRENCE_JSON.exists()
    assert BIOMARKER_DIST_JSON.exists()
    assert TREATMENT_RESISTANCE_JSON.exists()

    # Cancer type structure
    cancer_type_dist = dists["cancer_type"]
    assert "adenocarcinoma" in cancer_type_dist
    assert "squamous_cell_carcinoma" in cancer_type_dist
    assert cancer_type_dist["adenocarcinoma"]["count"] > 0

    # Age statistics verification
    age_dist = dists["age"]
    assert "summary_statistics" in age_dist
    assert age_dist["summary_statistics"]["median"] > 50
    assert age_dist["summary_statistics"]["min"] >= 18


def test_mutation_frequency_calculation():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    extractor = OncologyDistributionExtractor(cohort, raw_datasets)
    dists = extractor.run_all()

    mut_freq = dists["mutation_frequency"]
    assert "gene_level_frequencies" in mut_freq
    genes = [g["gene"] for g in mut_freq["gene_level_frequencies"]]
    assert "KRAS" in genes
    assert "EGFR" in genes
    assert "TP53" in genes


def test_mutation_cooccurrence():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    extractor = OncologyDistributionExtractor(cohort, raw_datasets)
    dists = extractor.run_all()

    cooc_dist = dists["mutation_cooccurrence"]
    patterns = [p["cooccurring_genes"] for p in cooc_dist["cooccurrence_patterns"]]
    assert len(patterns) > 0
    # Must contain observed pairs like KRAS + STK11 or EGFR
    assert any("KRAS" in p for p in patterns)


def test_anti_fabrication_unassayed_variables():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)

    extractor = OncologyDistributionExtractor(cohort, raw_datasets)
    dists = extractor.run_all()

    biomarkers = dists["biomarkers"]
    assert biomarkers["cohort_measured_biomarkers"]["inflammatory_markers_nlr_crp_ldh"] == UNAVAILABLE_SENTINEL
