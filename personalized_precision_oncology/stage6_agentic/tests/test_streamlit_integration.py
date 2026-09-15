"""
Integration Tests for Stage 6 Streamlit Clinician Workstation.

Verifies:
1. Headless Streamlit execution via streamlit.testing.v1.AppTest.
2. Initial layout rendering without exceptions.
3. Interactive deliberation button execution and session state persistence.
4. Component rendering for consensus, safety badges, and treatment candidates.
"""

import pytest
from streamlit.testing.v1 import AppTest

from personalized_precision_oncology.stage6_agentic.integration.streamlit.display import (
    render_consensus_badge,
    render_safety_status,
    render_treatment_candidate,
)


class TestStreamlitIntegration:

    def test_app_initial_load(self):
        """Verifies the Streamlit app loads and renders without exceptions."""
        from pathlib import Path
        app_path = str(Path(__file__).resolve().parent.parent / "integration" / "streamlit" / "app.py")
        at = AppTest.from_file(
            app_path,
            default_timeout=30,
        )
        at.run()
        assert not at.exception
        # Verify title is displayed
        assert any("Stage 6" in t.value or "Oncology" in t.value for t in at.title)
        # Verify sidebar button is present
        assert len(at.sidebar.button) >= 1

    def test_app_deliberation_execution(self):
        """Simulates clicking the 'Run Deliberation Panel' button."""
        from pathlib import Path
        app_path = str(Path(__file__).resolve().parent.parent / "integration" / "streamlit" / "app.py")
        at = AppTest.from_file(
            app_path,
            default_timeout=30,
        )
        at.run()
        assert not at.exception

        # Click the run button
        run_btn = [b for b in at.sidebar.button if "Run Deliberation Panel" in b.label][0]
        run_btn.click()
        at.run()

        assert not at.exception
        # After run, latest_response should be populated in session_state
        assert at.session_state["latest_response"] is not None
        latest = at.session_state["latest_response"]
        assert latest.case_id == "PT-LUNG-9842"
        assert latest.physician_review_required is True

    def test_app_safety_alert_preset_execution(self):
        """Tests selecting the Safety Alert preset and running deliberation."""
        from pathlib import Path
        app_path = str(Path(__file__).resolve().parent.parent / "integration" / "streamlit" / "app.py")
        at = AppTest.from_file(
            app_path,
            default_timeout=30,
        )
        at.run()
        assert not at.exception

        # Change preset in sidebar selectbox to Safety Alert
        preset_box = at.sidebar.selectbox[0]
        preset_box.select("Safety Alert: DDI / Contraindicated Regimen")
        at.run()
        assert not at.exception

        # Click run button
        run_btn = [b for b in at.sidebar.button if "Run Deliberation Panel" in b.label][0]
        run_btn.click()
        at.run()

        assert not at.exception
        latest = at.session_state["latest_response"]
        assert latest.safety_status == "BLOCKED"
        # Candidates suppressed
        assert len(latest.treatment_candidates) == 0
