import pytest
from integration.dashboard.app import (
    STAGE_1_PAGES,
    STAGE_2_PAGES,
    STAGE_3_PAGES,
    ALL_PAGE_KEYS
)

def test_stage_groupings_defined_and_disjoint():
    """Verify each stage has distinct, non-empty, and mutually disjoint pages."""
    assert len(STAGE_1_PAGES) == 4, f"Expected 4 pages in Stage 1, got {len(STAGE_1_PAGES)}"
    assert len(STAGE_2_PAGES) == 4, f"Expected 4 pages in Stage 2, got {len(STAGE_2_PAGES)}"
    assert len(STAGE_3_PAGES) == 3, f"Expected 3 pages in Stage 3, got {len(STAGE_3_PAGES)}"

    # Check mutual disjointness
    assert STAGE_1_PAGES.isdisjoint(STAGE_2_PAGES)
    assert STAGE_1_PAGES.isdisjoint(STAGE_3_PAGES)
    assert STAGE_2_PAGES.isdisjoint(STAGE_3_PAGES)

def test_stage1_subpages_contents():
    """Verify expected Stage 1 sub-pages are present."""
    expected = {
        "👤 Stage 1 — Clinical Risk Prediction",
        "🏆 Stage 1 — Model Benchmarks",
        "🧬 Stage 1 — Biomarker Analysis",
        "📁 Stage 1 — Batch Evaluation"
    }
    assert STAGE_1_PAGES == expected

def test_stage2_subpages_contents():
    """Verify expected Stage 2 sub-pages are present."""
    expected = {
        "🔬 Stage 2 — Histopathology Analysis (CNN)",
        "📈 Stage 2 — Biomarker Trajectory (Transformer)",
        "🧬 Stage 2 — Multimodal Fusion",
        "🔍 Stage 2 — Grad-CAM Explainability"
    }
    assert STAGE_2_PAGES == expected

def test_stage3_subpages_contents():
    """Verify expected Stage 3 sub-pages are present."""
    expected = {
        "📝 Stage 3 — Clinical Note Analysis (NLP)",
        "🚨 Stage 3 — Urgency Classification",
        "🏷️ Stage 3 — Oncology NER"
    }
    assert STAGE_3_PAGES == expected

def test_all_pages_count_and_inclusions():
    """Verify total of 15 unique domain pages are registered in ALL_PAGE_KEYS."""
    assert len(ALL_PAGE_KEYS) == 15
    assert "🏠 Dashboard Home" in ALL_PAGE_KEYS
    assert "🌐 Unified Patient Analysis" in ALL_PAGE_KEYS
    assert "🩺 System & Model Health" in ALL_PAGE_KEYS
    assert "ℹ️ About & Research Disclaimer" in ALL_PAGE_KEYS
    for p in STAGE_1_PAGES:
        assert p in ALL_PAGE_KEYS
    for p in STAGE_2_PAGES:
        assert p in ALL_PAGE_KEYS
    for p in STAGE_3_PAGES:
        assert p in ALL_PAGE_KEYS

def test_auto_expansion_logic():
    """Verify the auto-expansion logic for each stage."""
    def compute_expansions(active_page):
        s1 = active_page in STAGE_1_PAGES
        s2 = active_page in STAGE_2_PAGES
        s3 = active_page in STAGE_3_PAGES
        return s1, s2, s3

    # On Dashboard Home (default startup): neither stage is forced active by page
    s1, s2, s3 = compute_expansions("🏠 Dashboard Home")
    assert not s1 and not s2 and not s3

    # On Stage 1 page
    s1, s2, s3 = compute_expansions("🏆 Stage 1 — Model Benchmarks")
    assert s1 and not s2 and not s3

    # On Stage 2 page
    s1, s2, s3 = compute_expansions("📈 Stage 2 — Biomarker Trajectory (Transformer)")
    assert not s1 and s2 and not s3

    # On Stage 3 page
    s1, s2, s3 = compute_expansions("🚨 Stage 3 — Urgency Classification")
    assert not s1 and not s2 and s3
