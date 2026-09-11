"""
Unit tests for Scenario Generation, Diversity, and Seed Reproducibility.
"""

import pytest
from personalized_precision_oncology.stage5_genai.genai.generators.template_generator import TemplateScenarioGenerator
from personalized_precision_oncology.stage5_genai.genai.generators.llm_generator import LLMScenarioGenerator


def test_template_generator_count():
    gen = TemplateScenarioGenerator(seed=42)
    scenarios = gen.generate_20_edge_cases()
    assert len(scenarios) == 20


def test_scenario_unique_ids():
    gen = TemplateScenarioGenerator(seed=42)
    scenarios = gen.generate_20_edge_cases()
    ids = [s["scenario_id"] for s in scenarios]
    assert len(ids) == len(set(ids)), "Scenario IDs must be unique"
    for sid in ids:
        assert sid.startswith("EDGE_")


def test_reproducibility_with_same_seed():
    gen1 = TemplateScenarioGenerator(seed=42)
    gen2 = TemplateScenarioGenerator(seed=42)
    sc1 = gen1.generate_20_edge_cases()
    sc2 = gen2.generate_20_edge_cases()

    assert len(sc1) == len(sc2)
    for s1, s2 in zip(sc1, sc2):
        assert s1["scenario_id"] == s2["scenario_id"]
        assert s1["target_blind_spot"]["blind_spot_id"] == s2["target_blind_spot"]["blind_spot_id"]
        assert s1["patient_context"] == s2["patient_context"]


def test_llm_generator_graceful_fallback():
    # When no API key is provided, should fall back to template mode
    llm_gen = LLMScenarioGenerator(seed=42)
    scenarios = llm_gen.generate_scenarios(count=20)
    assert len(scenarios) == 20
    for sc in scenarios:
        assert sc["generation_method"] == "template"
        assert sc.get("fallback_reason") == "LLM configuration unavailable"
