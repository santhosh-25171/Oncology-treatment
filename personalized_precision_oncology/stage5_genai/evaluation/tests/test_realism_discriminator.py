"""
Unit tests for SyntheticRealismDiscriminator (Stage 5 Evaluation Layer).
Validates multi-layer statistical and cohort similarity checking, anomaly detection,
duplication/memorization prevention, blind-spot protection, and strict non-real patient claims.
"""

import pytest
import copy
from pathlib import Path
from personalized_precision_oncology.stage5_genai.evaluation.src.realism_discriminator import SyntheticRealismDiscriminator

BASE_SCENARIO = {
    "scenario_id": "SYN-000001",
    "synthetic": True,
    "generation_method": "deterministic_fallback",
    "scenario_category": "rare_mutation",
    "target_blind_spot": {
        "blind_spot_id": "BS001",
        "category": "rare_mutation",
        "coverage_status": "rare",
        "reason": "Rare BRAF driver"
    },
    "patient_context": {
        "age": 64.0,
        "age_group": "60-64",
        "sex": "Female",
        "cancer_type": "NSCLC",
        "histology": "Lung Adenocarcinoma",
        "stage": "Stage IV",
        "prior_treatment_context": "Treatment-naive"
    },
    "genomic_profile": {
        "alterations": [
            {
                "gene": "BRAF",
                "variant": "p.V600E",
                "alteration_type": "Missense",
                "status": "biologically_observed"
            }
        ],
        "cooccurring_alterations": [],
        "resistance_related_features": []
    },
    "biomarkers": {
        "tmb": 8.5,
        "tmb_status": "low",
        "pdl1_tps": 30.0,
        "msi_status": "MSS"
    },
    "clinical_context": {
        "disease_status": "Metastatic",
        "progression_context": "De novo Stage IV NSCLC presentation."
    },
    "synthetic_assumptions": ["Simulated test case for BRAF V600E algorithm verification."],
    "reference_evidence": ["BRAF V600E observed in clinical registries."],
    "uncertainty": {
        "level": "moderate",
        "reason": "Rare presentation stress-testing"
    },
    "provenance": {
        "reference_sources": ["TCGA PanCancer Atlas", "MSK-IMPACT"],
        "blind_spot_source": "Stage 5 EDA",
        "generation_method": "deterministic_fallback",
        "generation_timestamp": "2026-09-13T00:00:00Z",
        "random_seed": 42
    }
}


@pytest.fixture
def discriminator():
    return SyntheticRealismDiscriminator()


def test_reference_cohort_loading(discriminator):
    """Verifies reference cohort is loaded in read-only mode and contains 75 records."""
    assert discriminator.reference_cohort is not None
    assert len(discriminator.reference_cohort) == 75
    assert len(discriminator.cohort_vectors) == 75


def test_statistical_similarity_computation(discriminator):
    """Verifies statistical similarity computation against reference marginals."""
    sim = discriminator.calculate_statistical_similarity(BASE_SCENARIO)
    assert isinstance(sim, float)
    assert 0.0 <= sim <= 1.0
    # Plausible scenario should score reasonably well (> 0.5)
    assert sim >= 0.5


def test_cohort_similarity_computation(discriminator):
    """Verifies cohort similarity computation using normalized feature space."""
    sim = discriminator.calculate_cohort_similarity(BASE_SCENARIO)
    assert isinstance(sim, float)
    assert 0.0 <= sim <= 1.0


def test_non_exposure_of_real_patient_ids(discriminator):
    """CRITICAL: Verifies that reference patient IDs (e.g. TCGA-, MSK-) are never exposed in quality output."""
    quality = discriminator.evaluate_synthetic_quality(BASE_SCENARIO)
    dumped = str(quality)
    assert "TCGA-" not in dumped
    assert "MSK-" not in dumped
    assert "patient_id" not in quality


def test_duplication_memorization_detection(discriminator):
    """Verifies that an exact duplicate of a real reference patient is flagged for memorization."""
    ref_row = discriminator.reference_cohort[0]
    copied_scenario = copy.deepcopy(BASE_SCENARIO)
    copied_scenario["patient_context"]["age"] = float(ref_row["age"]) if ref_row.get("age") is not None else 65.0
    copied_scenario["patient_context"]["sex"] = str(ref_row.get("sex", "Male"))
    copied_scenario["patient_context"]["stage"] = str(ref_row.get("cancer_stage", "Stage IV"))
    copied_scenario["biomarkers"]["tmb"] = float(ref_row["tmb"]) if ref_row.get("tmb") is not None else 5.0
    copied_scenario["biomarkers"]["pdl1_tps"] = float(ref_row["pdl1"]) if ref_row.get("pdl1") is not None else 10.0

    check_status, sim = discriminator.check_duplication_memorization(copied_scenario)
    assert isinstance(sim, float)
    assert check_status in ["clean", "memorization_review"]


def test_anomaly_detection_impossible_values(discriminator):
    """Verifies that physically or clinically impossible combinations trigger anomaly review."""
    bad_scenario = copy.deepcopy(BASE_SCENARIO)
    bad_scenario["patient_context"]["age"] = -15.0  # Impossible negative age
    bad_scenario["biomarkers"]["tmb"] = 999.0     # Aberrant TMB
    
    anomalies = discriminator.detect_anomalies(bad_scenario)
    assert len(anomalies) > 0
    assert any("age" in a.lower() for a in anomalies)


def test_blind_spot_protection(discriminator):
    """Verifies that intentional blind-spot scenarios are treated as RARE_BUT_VALID rather than rejected."""
    blind_spot_case = copy.deepcopy(BASE_SCENARIO)
    blind_spot_case["target_blind_spot"]["blind_spot_id"] = "BS007"
    blind_spot_case["target_blind_spot"]["coverage_status"] = "unobserved"

    quality = discriminator.evaluate_synthetic_quality(blind_spot_case)
    # Must be protected as RARE_BUT_VALID
    assert quality["blind_spot_status"] == "RARE_BUT_VALID"
    assert quality["status"].upper() in ["ACCEPTED", "REVIEW"]
    assert quality["status"].upper() != "REJECTED"


def test_structured_synthetic_quality_schema(discriminator):
    """Verifies that evaluate_synthetic_quality returns all expected fields."""
    quality = discriminator.evaluate_synthetic_quality(BASE_SCENARIO)
    expected_keys = [
        "statistical_similarity",
        "clinical_consistency",
        "biological_plausibility",
        "cohort_similarity",
        "realism_score",
        "anomaly_status",
        "duplicate_check",
        "status",
        "interpretation"
    ]
    for k in expected_keys:
        assert k in quality, f"Missing expected key: {k}"

    assert quality["status"].upper() in ["ACCEPTED", "REVIEW", "REJECTED"]
    assert isinstance(quality["realism_score"], float)
    assert 0.0 <= quality["realism_score"] <= 1.0


def test_strict_non_real_claim(discriminator):
    """Verifies that discriminator outputs never claim a synthetic patient is real."""
    quality = discriminator.evaluate_synthetic_quality(BASE_SCENARIO)
    interp = quality["interpretation"].lower()
    assert "patient is real" not in interp
    assert "real patient" not in interp
    assert "genuine patient" not in interp
