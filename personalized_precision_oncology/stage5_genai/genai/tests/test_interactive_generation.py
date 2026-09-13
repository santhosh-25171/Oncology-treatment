"""
Unit tests for Interactive Patient Generator (Stage 5 GenAI Layer).
Tests interactive synthetic generation, seed fidelity, sequential ID allocation,
deterministic fallback routing, and strict synthetic labeling.
"""

import pytest
import json
from pathlib import Path
from personalized_precision_oncology.stage5_genai.genai.generators.interactive_generator import InteractivePatientGenerator
from personalized_precision_oncology.stage5_genai.eda_prompteng.eda.prompt_builder import PromptBuilder


@pytest.fixture
def temp_scenarios_file(tmp_path):
    return tmp_path / "test_generated_scenarios.jsonl"


@pytest.fixture
def generator(temp_scenarios_file):
    return InteractivePatientGenerator(scenarios_file=temp_scenarios_file)


def test_sequential_id_allocation(generator, temp_scenarios_file):
    """Verifies that generated synthetic IDs are sequential (SYN-000001, SYN-000002, ...)."""
    assert generator.get_next_synthetic_id() == "SYN-000001"

    # Simulate existing entries in file
    with open(temp_scenarios_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": "SYN-000001", "synthetic": True}) + "\n")
        f.write(json.dumps({"scenario_id": "SYN-000002", "synthetic": True}) + "\n")

    assert generator.get_next_synthetic_id() == "SYN-000003"


def test_deterministic_fallback_execution(generator):
    """Verifies deterministic fallback generation when live LLM API is not configured."""
    seed = {
        "age": 58.0,
        "sex": "Female",
        "cancer_type": "NSCLC",
        "stage": "Stage IV",
        "histology": "Lung Adenocarcinoma",
        "smoking_status": "Never Smoker",
        "driver_alteration": "EGFR",
        "tmb": 8.0,
        "pd_l1": 50.0
    }

    scenario, meta = generator.generate_patient(seed_conditions=seed, blind_spot_id="BS001")

    # Assert metadata
    assert meta["generation_source"] in ["DETERMINISTIC_FALLBACK", "LLM"]
    if not generator.is_llm_configured():
        assert meta["generation_source"] == "DETERMINISTIC_FALLBACK"
        assert meta["fallback_reason"] == "LLM_NOT_CONFIGURED"

    # Assert scenario structure
    assert scenario["scenario_id"].startswith("SYN-")
    assert scenario["synthetic"] is True
    assert scenario["patient_context"]["age"] == 58.0
    assert scenario["patient_context"]["sex"] == "Female"
    assert scenario["patient_context"]["stage"] == "Stage IV"
    assert scenario["patient_context"]["histology"] == "Lung Adenocarcinoma"
    assert scenario["biomarkers"]["tmb"] == 8.0
    assert scenario["biomarkers"]["pdl1_tps"] == 50.0
    assert scenario["genomic_profile"]["alterations"][0]["gene"] == "EGFR"


def test_prompt_builder_construction():
    """Verifies PromptBuilder constructs valid system and user prompts incorporating seed conditions."""
    builder = PromptBuilder()
    seed = {
        "age": 72.0,
        "sex": "Male",
        "cancer_type": "NSCLC",
        "stage": "Stage IIIA",
        "histology": "Lung Squamous Cell Carcinoma",
        "driver_alteration": "KRAS",
        "tmb": 12.0,
        "pd_l1": 0.0
    }

    prompt = builder.build_prompt(seed_conditions=seed, scenario_id="SYN-000005", blind_spot_id="BS011")

    assert "system_prompt" in prompt
    assert "user_prompt" in prompt
    assert "target_blind_spot" in prompt
    assert "SYN-000005" in prompt["user_prompt"]
    assert "KRAS" in prompt["user_prompt"]
    assert "72.0" in prompt["user_prompt"]
    # Verify safety governance tiers are embedded
    assert "THREE-TIER EVIDENCE GOVERNANCE MANDATE" in prompt["system_prompt"]
    assert "SUPPORTED EVIDENCE" in prompt["system_prompt"]
    assert "UNKNOWN / INSUFFICIENT EVIDENCE" in prompt["system_prompt"]


def test_persistence_isolation(generator, temp_scenarios_file):
    """Verifies newly generated scenarios are persisted to generated_scenarios.jsonl without touching reference cohort."""
    seed = {"age": 60.0, "sex": "Male", "driver_alteration": "BRAF"}
    scenario, _ = generator.generate_patient(seed_conditions=seed, blind_spot_id="BS001")

    assert temp_scenarios_file.exists()
    with open(temp_scenarios_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 1
    assert lines[0]["scenario_id"] == scenario["scenario_id"]
    assert lines[0]["synthetic"] is True
