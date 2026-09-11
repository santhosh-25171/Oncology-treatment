"""
Unit tests for Stage 5 Genomic EDA sub-modules.
Validates calculation accuracy, non-fabrication, and source separation.
"""

import json
import pytest
from pathlib import Path

from personalized_precision_oncology.stage5_genai.eda_prompteng.eda.genomic_coverage import GenomicCoverageAnalyzer
from personalized_precision_oncology.stage5_genai.eda_prompteng.eda.mutation_analysis import MutationLandscapeAnalyzer
from personalized_precision_oncology.stage5_genai.eda_prompteng.eda.cooccurrence_analysis import CooccurrenceAnalyzer
from personalized_precision_oncology.stage5_genai.eda_prompteng.eda.resistance_analysis import ResistanceAnalyzer
from personalized_precision_oncology.stage5_genai.eda_prompteng.eda.biomarker_analysis import BiomarkerAnalyzer


def test_genomic_coverage_feature_availability():
    """Verify feature availability audit checks variables before calculation."""
    analyzer = GenomicCoverageAnalyzer()
    res = analyzer.run()
    assert "feature_availability_audit" in res
    audit = res["feature_availability_audit"]
    assert "age" in audit
    assert audit["age"]["status"] == "available"
    assert audit["age"]["available_records"] == 75

    # Stage distribution integrity
    stages = res["stage_genomic_coverage"]["overall_stage_counts"]
    assert sum(stages.values()) == 75


def test_mutation_analysis_vaf_limitation():
    """Verify that VAF unavailability is logged as source_limitation rather than failing or fabricating."""
    analyzer = MutationLandscapeAnalyzer()
    res = analyzer.run()
    vaf_audit = res["vaf_feature_audit"]
    assert vaf_audit["status"] == "source_limitation"
    assert "Variable 'variant_allele_fraction' not available" in vaf_audit["reason"]


def test_rare_mutation_identification():
    """Verify rare mutations (<3% cohort prevalence) are identified correctly."""
    analyzer = MutationLandscapeAnalyzer()
    rare_muts = analyzer.identify_rare_mutations(threshold_percent=3.0)
    assert len(rare_muts) > 0
    for rm in rare_muts:
        assert rm["frequency_percent"] <= 3.0 or rm["observed_count"] <= 2
        assert rm["classification"] == "rare"


def test_candidate_check_targets_non_fabrication():
    """Verify candidate targets evaluate to not_observed_in_reference_baseline if not present."""
    analyzer = MutationLandscapeAnalyzer()
    targets = analyzer.evaluate_check_targets()
    target_genes = {t["target_gene"]: t for t in targets}
    
    # NTRK1 should be not_observed
    assert "NTRK1" in target_genes
    assert target_genes["NTRK1"]["coverage_status"] == "not_observed"
    assert "not_observed_in_reference_baseline" in target_genes["NTRK1"]["finding"]


def test_cooccurrence_analysis_known_combinations():
    """Verify co-occurrence analyzer extracts observed combinations."""
    analyzer = CooccurrenceAnalyzer()
    res = analyzer.run()
    observed = res["observed_cooccurrences"]
    combos = [c["combination"] for c in observed]
    assert any("KRAS" in c for c in combos)
    assert any("EGFR" in c for c in combos)


def test_resistance_mechanisms_audited():
    """Verify resistance mechanisms audit reports correct representations."""
    analyzer = ResistanceAnalyzer()
    res = analyzer.run()
    evals = res["candidate_resistance_evaluation"]
    assert len(evals) >= 6
    mechs = {e["mechanism"]: e for e in evals}
    assert "EGFR C797S Tertiary Resistance" in mechs
    assert mechs["EGFR C797S Tertiary Resistance"]["coverage_status"] == "sparse"


def test_biomarker_analysis_tmb_and_pdl1():
    """Verify TMB and PD-L1 distribution calculations from verified records."""
    analyzer = BiomarkerAnalyzer()
    res = analyzer.run()
    tmb = res["tmb_coverage"]
    assert tmb["status"] == "available"
    assert tmb["assayed_patients"] == 25
    assert tmb["median"] == 7.0

    pdl1 = res["pdl1_coverage"]
    assert pdl1["status"] == "available"
    assert pdl1["assayed_patients"] == 25
    assert "categories" in pdl1
    assert pdl1["categories"]["high_gte_50pct"]["count"] == 7
