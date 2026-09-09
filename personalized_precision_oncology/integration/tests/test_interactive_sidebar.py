import pytest
from streamlit.testing.v1 import AppTest

def test_app_sidebar_interaction_workflow():
    at = AppTest.from_file("integration/dashboard/app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # 1. Startup State: Exactly 12 buttons initially
    # (Home, Unified, Stage 1 Header + 4 subpages, Stage 2 Header, Stage 3 Header, Health, About, Disclaimer)
    assert len(at.sidebar.button) == 12
    labels = [b.label for b in at.sidebar.button]

    # Verify Stage 1 is expanded (its sub-pages are in labels)
    assert any("Clinical Risk Prediction" in l for l in labels)
    assert any("Model Benchmarks" in l for l in labels)
    assert any("Biomarker Analysis" in l for l in labels)
    assert any("Batch Evaluation" in l for l in labels)

    # Verify Stage 2 & 3 are collapsed initially (their sub-pages not in labels)
    assert not any("Histopathology Analysis" in l for l in labels)
    assert not any("Urgency Classification" in l for l in labels)

    # 2. Toggle Stage 2 Header (should expand Stage 2 without navigating)
    btn_s2_header = [b for b in at.sidebar.button if "STAGE 2" in b.label][0]
    btn_s2_header.click()
    at.run()
    assert not at.exception

    labels_after_s2_toggle = [b.label for b in at.sidebar.button]
    # Now Stage 2 sub-pages should be present!
    assert any("Histopathology Analysis" in l for l in labels_after_s2_toggle)
    assert any("Biomarker Trajectory" in l for l in labels_after_s2_toggle)
    assert any("Multimodal Fusion" in l for l in labels_after_s2_toggle)
    assert any("Grad-CAM Explainability" in l for l in labels_after_s2_toggle)

    # Active page should still be Dashboard Home (header click didn't navigate!)
    assert at.session_state["active_page"] == "🏠 Dashboard Home"

    # 3. Click a Stage 2 sub-page
    btn_histo = [b for b in at.sidebar.button if "Histopathology Analysis" in b.label][0]
    btn_histo.click()
    at.run()
    assert not at.exception

    # Active page is now Histopathology Analysis
    assert at.session_state["active_page"] == "🔬 Stage 2 — Histopathology Analysis (CNN)"
    
    # Stage 2 header should now have the Active indicator (●)
    s2_header_now = [b for b in at.sidebar.button if "STAGE 2" in b.label][0]
    assert "●" in s2_header_now.label

    # 4. Collapse Stage 1 by clicking its header
    btn_s1_header = [b for b in at.sidebar.button if "STAGE 1" in b.label][0]
    btn_s1_header.click()
    at.run()
    assert not at.exception

    labels_after_s1_collapse = [b.label for b in at.sidebar.button]
    # Stage 1 sub-pages should now be hidden
    assert not any("Model Benchmarks" in l for l in labels_after_s1_collapse)
    assert not any("Biomarker Analysis" in l for l in labels_after_s1_collapse)
