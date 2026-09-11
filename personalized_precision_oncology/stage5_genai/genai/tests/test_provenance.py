"""
Unit tests for Scenario Provenance and Synthetic Labeling.
"""

import pytest
from personalized_precision_oncology.stage5_genai.genai.generators.template_generator import TemplateScenarioGenerator


def test_synthetic_flag_true():
    gen = TemplateScenarioGenerator(seed=42)
    scenarios = gen.generate_20_edge_cases()

    for sc in scenarios:
        assert sc["synthetic"] is True, "Scenario must explicitly declare synthetic: true"


def test_provenance_structure():
    gen = TemplateScenarioGenerator(seed=42)
    scenarios = gen.generate_20_edge_cases()

    for sc in scenarios:
        assert "provenance" in sc
        prov = sc["provenance"]
        assert len(prov["reference_sources"]) > 0
        assert "blind_spot_source" in prov
        assert "generation_timestamp" in prov
        assert prov["random_seed"] == 42
        assert prov["generation_method"] == "template"


def test_evidence_and_assumption_separation():
    gen = TemplateScenarioGenerator(seed=42)
    scenarios = gen.generate_20_edge_cases()

    for sc in scenarios:
        assert len(sc["reference_evidence"]) > 0
        assert len(sc["synthetic_assumptions"]) > 0
        for ev in sc["reference_evidence"]:
            assert "synthetic assumption" not in ev.lower()
