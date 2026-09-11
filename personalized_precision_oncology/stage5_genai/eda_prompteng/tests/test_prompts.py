"""
Unit tests for prompt templates and prompt coverage report.
"""

import json
import pytest
from pathlib import Path
from jsonschema import validate

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
REPORTS_DIR = BASE_DIR / "reports"
SCHEMAS_DIR = BASE_DIR / "schemas"

PROMPT_SCHEMA_JSON = SCHEMAS_DIR / "prompt_schema.json"
PROMPT_COVERAGE_REPORT_JSON = REPORTS_DIR / "prompt_coverage_report.json"
BLIND_SPOT_REPORT_JSON = REPORTS_DIR / "blind_spot_report.json"

REQUIRED_PROMPT_FILES = [
    "system_prompt.txt",
    "extreme_resistance_prompt.txt",
    "compound_mutation_prompt.txt",
    "conflicting_evidence_prompt.txt",
    "sparse_evidence_prompt.txt",
    "wildcard_prompt.txt"
]

REQUIRED_PLACEHOLDERS = [
    "{{reference_baseline}}",
    "{{patient_context}}",
    "{{genomic_profile}}",
    "{{biomarkers}}",
    "{{treatment_context}}",
    "{{blind_spot_id}}",
    "{{blind_spot_pattern}}"
]

EVIDENCE_SECTIONS = [
    "[SUPPORTED EVIDENCE]",
    "[SYNTHETIC ASSUMPTION]",
    "[UNKNOWN / INSUFFICIENT EVIDENCE]"
]


def test_prompt_files_exist():
    """Verify all 6 prompt template files exist."""
    for pf in REQUIRED_PROMPT_FILES:
        path = PROMPTS_DIR / pf
        assert path.exists(), f"Prompt file missing: {pf}"


def test_prompt_template_placeholders():
    """Verify each scenario prompt contains the required placeholder tokens."""
    for pf in REQUIRED_PROMPT_FILES:
        if pf == "system_prompt.txt":
            continue
        content = (PROMPTS_DIR / pf).read_text(encoding="utf-8")
        for placeholder in REQUIRED_PLACEHOLDERS:
            assert placeholder in content, f"Missing placeholder {placeholder} in {pf}"


def test_three_tier_evidence_separation():
    """Verify all prompt files enforce the three-tier evidence governance."""
    for pf in REQUIRED_PROMPT_FILES:
        content = (PROMPTS_DIR / pf).read_text(encoding="utf-8")
        for sec in EVIDENCE_SECTIONS:
            assert sec in content, f"Missing evidence section {sec} in {pf}"


def test_prompt_coverage_report_schema():
    """Verify prompt_coverage_report.json complies with prompt_schema.json."""
    assert PROMPT_COVERAGE_REPORT_JSON.exists(), "prompt_coverage_report.json missing!"
    with open(PROMPT_COVERAGE_REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(PROMPT_SCHEMA_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validate(instance=data, schema=schema)


def test_prompt_references_valid_blind_spots():
    """Verify all prompts reference actual detected blind spot IDs."""
    with open(PROMPT_COVERAGE_REPORT_JSON, "r", encoding="utf-8") as f:
        p_data = json.load(f)
    with open(BLIND_SPOT_REPORT_JSON, "r", encoding="utf-8") as f:
        bs_data = json.load(f)

    valid_bs_ids = {bs["blind_spot_id"] for bs in bs_data["blind_spots"]}
    for p in p_data["prompts"]:
        target_id = p["target_blind_spot_id"]
        assert target_id in valid_bs_ids, f"Prompt {p['prompt_id']} references nonexistent blind spot {target_id}"
