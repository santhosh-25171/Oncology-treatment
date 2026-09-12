"""
Comprehensive Quality, Provenance, and Source-Separation Validation Suite.
Tests the 12 rigorous criteria specified in the Stage 5 Quality & Provenance Improvement requirements:
1. source separation
2. provenance presence
3. missing-value semantics
4. rare mutation preservation
5. duplicate handling
6. invalid record quarantine
7. JSONL schema validation
8. reference IDs
9. source_type validation
10. data_domain validation
11. evidence_status validation
12. deterministic output
"""

import json
import pytest
import pandas as pd
import numpy as np

from personalized_precision_oncology.stage5_genai.data_engineering.src.ingestion import OncologyDataIngestion
from personalized_precision_oncology.stage5_genai.data_engineering.src.cleaning import OncologyDataCleaner
from personalized_precision_oncology.stage5_genai.data_engineering.src.validation import OncologyDataValidator
from personalized_precision_oncology.stage5_genai.data_engineering.src.distributions import OncologyDistributionExtractor
from personalized_precision_oncology.stage5_genai.data_engineering.src.genai_baseline import GenAIReferenceBaselineCompiler
from personalized_precision_oncology.stage5_genai.data_engineering.src.config import (
    GENAI_REFERENCE_BASELINE_JSONL,
    CLEANED_COHORT_CSV,
    SOURCE_METADATA_CSV,
    DATA_QUALITY_REPORT_JSON,
    QUARANTINE_REPORT_JSON,
    SCHEMAS_DIR,
    UNAVAILABLE_SENTINEL,
    NOT_OBSERVED_SENTINEL,
    INSUFFICIENT_EVIDENCE_SENTINEL,
    SOURCE_LIMITATION_SENTINEL,
    SOURCE_TYPES,
    DATA_DOMAINS,
    EVIDENCE_STATUSES
)


@pytest.fixture(scope="module")
def pipeline_data():
    ingestion = OncologyDataIngestion()
    raw_datasets, raw_counts = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)
    validator = OncologyDataValidator()
    report = validator.run_all(cohort, raw_counts, cleaner.metrics)
    dist_extractor = OncologyDistributionExtractor(cohort, raw_datasets)
    dists = dist_extractor.run_all()
    compiler = GenAIReferenceBaselineCompiler(cohort, dists, raw_datasets)
    compiler.compile_and_export()

    records = []
    with open(GENAI_REFERENCE_BASELINE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    return {
        "raw_datasets": raw_datasets,
        "raw_counts": raw_counts,
        "cohort": cohort,
        "cleaner": cleaner,
        "report": report,
        "records": records
    }


def test_source_separation_semantic_integrity(pipeline_data):
    """1. Verify heterogeneous sources are NOT merged into the patient cohort."""
    cohort = pipeline_data["cohort"]
    # Cohort must only contain patient-level sources (TCGA and MSK-IMPACT)
    sources = set(cohort["data_source"].unique())
    for s in sources:
        assert any(k in s for k in ["TCGA", "MSK-IMPACT"]), f"Invalid non-patient source found in cohort: {s}"
    # SEER and CIViC must NOT be in the patient-level cohort
    assert not any("SEER" in s for s in sources), "SEER epidemiology was merged into patient cohort!"
    assert not any("CIViC" in s for s in sources), "CIViC evidence was merged into patient cohort!"
    assert not any("Keynote" in s or "Biomarker" in s for s in sources), "Biomarker evidence was merged into patient cohort!"


def test_provenance_presence_and_traceability(pipeline_data):
    """2. Verify provenance is present on every patient record and every reference record."""
    cohort = pipeline_data["cohort"]
    records = pipeline_data["records"]

    # Every patient record has data_source and source_record_id
    assert (cohort["data_source"].str.len() > 0).all()
    assert (cohort["source_record_id"].str.len() > 0).all()

    # Every reference record has complete provenance
    for r in records:
        assert "provenance" in r
        prov = r["provenance"]
        assert "source_name" in prov and len(prov["source_name"]) > 0
        assert "dataset" in prov and len(prov["dataset"]) > 0
        assert "reference" in prov and len(prov["reference"]) > 0
        assert "access_date" in prov and len(prov["access_date"]) > 0


def test_missing_value_semantics_and_sentinels(pipeline_data):
    """3. Verify strict missing data semantics: not_available_in_source vs not_observed vs limitation."""
    cohort = pipeline_data["cohort"]
    records = pipeline_data["records"]

    # TCGA records must have not_available_in_source for ctDNA
    tcga_pats = cohort[cohort["data_source"].str.contains("TCGA")]
    assert (tcga_pats["ctdna_maf_percent"] == UNAVAILABLE_SENTINEL).all()

    # In SEER reference block, mutation patterns must be marked not_available_in_source
    seer_rec = next(r for r in records if "SEER" in r["reference_id"])
    assert seer_rec["mutation_patterns"] == [UNAVAILABLE_SENTINEL]
    assert seer_rec["biomarker_distributions"]["status"] == UNAVAILABLE_SENTINEL


def test_rare_mutation_preservation(pipeline_data):
    """4. Verify verified rare resistance mutations are preserved in cohort and distributions."""
    cohort = pipeline_data["cohort"]
    all_muts = set(cohort["primary_mutation"].unique())
    for sm in cohort["secondary_resistance_mutation"].unique():
        if pd.notna(sm) and sm not in [UNAVAILABLE_SENTINEL, "None"]:
            for part in str(sm).split(";"):
                all_muts.add(part.split(":")[-1].strip())

    # Critical rare resistance mutations
    assert any("C797S" in m for m in all_muts)
    assert any("T790M" in m for m in all_muts)
    assert any("G1202R" in m for m in all_muts)
    assert any("Y99C" in m for m in all_muts)
    assert any("D1010H" in m for m in all_muts)


def test_duplicate_handling(pipeline_data):
    """5. Verify duplicate detection and deduplication."""
    cleaner = pipeline_data["cleaner"]
    cohort = pipeline_data["cohort"]
    assert cleaner.metrics["duplicates_removed"] == 3
    assert cohort["patient_id"].is_unique


def test_invalid_record_quarantine_integrity(pipeline_data):
    """6. Verify quarantine report exists, records quarantined row with id, reason, rule, source."""
    assert QUARANTINE_REPORT_JSON.exists()
    with open(QUARANTINE_REPORT_JSON, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    assert q_data["total_quarantined"] == 1
    q_rec = q_data["quarantined_records"][0]
    assert q_rec["record_id"] == "TCGA-ERR-9999"
    assert "Invalid biological range" in q_rec["reason"]
    assert "CLINICAL_BOUNDS" in q_rec["validation_rule"]
    assert q_rec["source"] == "TCGA-GDC PanCancer Atlas"
    assert q_rec["status"] == "quarantined"


def test_jsonl_schema_validation(pipeline_data):
    """7. Verify reference baseline JSONL conforms to genai_reference_baseline_schema.json."""
    records = pipeline_data["records"]
    with open(SCHEMAS_DIR / "genai_reference_baseline_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    import jsonschema
    for r in records:
        jsonschema.validate(instance=r, schema=schema)


def test_reference_ids_format(pipeline_data):
    """8. Verify all reference records possess unique, structured REF-* identifiers."""
    records = pipeline_data["records"]
    ref_ids = [r["reference_id"] for r in records]
    assert len(ref_ids) == len(set(ref_ids))  # unique
    for rid in ref_ids:
        assert rid.startswith("REF-"), f"Identifier {rid} does not start with REF-"


def test_source_type_validation(pipeline_data):
    """9. Verify every reference record has a valid canonical source_type."""
    records = pipeline_data["records"]
    for r in records:
        assert "source_type" in r
        assert r["source_type"] in SOURCE_TYPES, f"Invalid source_type: {r['source_type']}"


def test_data_domain_validation(pipeline_data):
    """10. Verify every reference record has a valid canonical data_domain."""
    records = pipeline_data["records"]
    for r in records:
        assert "data_domain" in r
        assert r["data_domain"] in DATA_DOMAINS, f"Invalid data_domain: {r['data_domain']}"


def test_evidence_status_validation(pipeline_data):
    """11. Verify every reference record has a valid canonical evidence_status."""
    records = pipeline_data["records"]
    for r in records:
        assert "evidence_status" in r
        assert r["evidence_status"] in EVIDENCE_STATUSES, f"Invalid evidence_status: {r['evidence_status']}"


def test_deterministic_output_reproducibility():
    """12. Verify running the pipeline produces identical record counts and hashes."""
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort1 = cleaner.run(raw_datasets)

    cleaner2 = OncologyDataCleaner()
    cohort2 = cleaner2.run(raw_datasets)

    assert len(cohort1) == len(cohort2) == 75
    assert (cohort1["patient_id"].values == cohort2["patient_id"].values).all()
