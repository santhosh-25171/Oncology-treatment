"""
Unit tests for GenAI reference baseline export.
Verifies:
- JSONL schema validation
- Provenance validation
- Source-separation rule adherence
- Line-by-line JSON validity
"""

import json
import pytest
from personalized_precision_oncology.stage5_genai.data_engineering.src.ingestion import OncologyDataIngestion
from personalized_precision_oncology.stage5_genai.data_engineering.src.cleaning import OncologyDataCleaner
from personalized_precision_oncology.stage5_genai.data_engineering.src.distributions import OncologyDistributionExtractor
from personalized_precision_oncology.stage5_genai.data_engineering.src.genai_baseline import GenAIReferenceBaselineCompiler
from personalized_precision_oncology.stage5_genai.data_engineering.src.config import GENAI_REFERENCE_BASELINE_JSONL, SCHEMAS_DIR


def test_genai_jsonl_compilation_and_validity():
    ingestion = OncologyDataIngestion()
    raw_datasets, _ = ingestion.load_all()
    cleaner = OncologyDataCleaner()
    cohort = cleaner.run(raw_datasets)
    dist_extractor = OncologyDistributionExtractor(cohort, raw_datasets)
    dists = dist_extractor.run_all()

    compiler = GenAIReferenceBaselineCompiler(cohort, dists, raw_datasets)
    count = compiler.compile_and_export()

    assert count >= 5
    assert GENAI_REFERENCE_BASELINE_JSONL.exists()

    records = []
    with open(GENAI_REFERENCE_BASELINE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rec = json.loads(line_str)
                records.append(rec)

    assert len(records) == count

    # Validate against JSON Schema
    with open(SCHEMAS_DIR / "genai_reference_baseline_schema.json", "r") as sf:
        schema = json.load(sf)
    required_keys = schema["required"]

    for r in records:
        for k in required_keys:
            assert k in r, f"Missing required key '{k}' in JSONL record {r.get('reference_id')}"
        assert "provenance" in r
        assert "source_name" in r["provenance"]
        assert "reference" in r["provenance"]


def test_source_separation_preservation():
    records = []
    with open(GENAI_REFERENCE_BASELINE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    sources = [r["source"] for r in records]
    data_types = [r["data_type"] for r in records]

    # Verify that distinct sources and data types are preserved separately
    assert any("TCGA" in s for s in sources)
    assert any("MSK" in s for s in sources)
    assert any("SEER" in s for s in sources)
    assert any("CIViC" in s for s in sources)

    assert any("Epidemiology" in dt for dt in data_types)
    assert any("Genomic" in dt for dt in data_types)
    assert any("Evidence" in dt for dt in data_types)
