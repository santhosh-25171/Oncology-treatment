"""
Unit tests for GenomicConsistencyAuditor.
"""

import copy
import pytest
from personalized_precision_oncology.stage5_genai.evaluation.src.genomic_consistency import GenomicConsistencyAuditor
from personalized_precision_oncology.stage5_genai.evaluation.src.scenario_loader import ScenarioLoader


def test_genomic_consistency_all_scenarios_valid():
    loader = ScenarioLoader()
    auditor = GenomicConsistencyAuditor(
        loader.mutation_frequencies,
        loader.mutation_cooccurrences,
        loader.treatment_resistance
    )
    scenarios = loader.get_scenarios()

    for sc in scenarios:
        res = auditor.audit_genomics(sc)
        assert res["valid"] is True, f"Scenario {sc.get('scenario_id')} had genomic issues: {res['issues']}"
        assert len(res["classifications"]) > 0


def test_genomic_consistency_detects_false_historical_claim():
    loader = ScenarioLoader()
    auditor = GenomicConsistencyAuditor(
        loader.mutation_frequencies,
        loader.mutation_cooccurrences,
        loader.treatment_resistance
    )
    sc = copy.deepcopy(loader.get_scenarios()[0])
    # Add fictitious gene claimed as historically observed
    sc["genomic_profile"]["alterations"].append({
        "gene": "FICTITIOUS_GENE_XYZ",
        "variant": "p.X123Y",
        "status": "historically_observed"  # Violation
    })

    res = auditor.audit_genomics(sc)
    assert res["valid"] is False
    assert any("FICTITIOUS_GENE_XYZ" in issue for issue in res["issues"])
