"""
Unit tests for Stage 5 ReferenceDataLoader.
Verifies loading of historical baseline files without fabrication or mutation of original records.
"""

import pytest
from personalized_precision_oncology.stage5_genai.genai.src.reference_loader import ReferenceDataLoader


def test_reference_loader_initialization():
    loader = ReferenceDataLoader()
    assert loader.mutation_frequencies is not None
    assert "gene_level_frequencies" in loader.mutation_frequencies
    assert len(loader.baseline_blocks) > 0


def test_reference_loader_gene_frequency():
    loader = ReferenceDataLoader()
    kras_freq = loader.get_gene_frequency("KRAS")
    assert kras_freq == 24.0
    egfr_freq = loader.get_gene_frequency("EGFR")
    assert egfr_freq == 22.67


def test_reference_loader_cooccurrences():
    loader = ReferenceDataLoader()
    coocs = loader.get_top_cooccurrences()
    assert len(coocs) > 0


def test_reference_loader_summary_text():
    loader = ReferenceDataLoader()
    summary = loader.get_summary_text()
    assert "Historical Reference Baseline" in summary
    assert "KRAS" in summary
