"""
Display and Formatting Utilities for Stage 6 Streamlit Dashboard.

Provides clinical formatting functions for multidisciplinary consensus badges,
safety indicators, evidence citations, treatment candidate cards, and audit events.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st


def render_consensus_badge(consensus: str) -> None:
    """Renders a visual indicator for multidisciplinary consensus."""
    norm = consensus.upper()
    if norm == "CONSENSUS":
        st.success(f"**Multidisciplinary Consensus:** {consensus} — Unified clinical agreement reached.")
    elif norm == "DISCORDANT":
        st.warning(f"**Multidisciplinary Consensus:** {consensus} — Divergence detected across modalities.")
    elif norm == "BLOCKED":
        st.error(f"**Multidisciplinary Consensus:** {consensus} — Synthesis blocked by Safety Guardian.")
    else:
        st.info(f"**Multidisciplinary Consensus:** {consensus} — Incomplete or partial agreement.")


def render_safety_status(safety_status: str, warnings: Optional[List[str]] = None) -> None:
    """Renders the SafetyGuardian clearance badge and relevant warnings."""
    status_upper = safety_status.upper()
    if "SAFE" in status_upper:
        st.success("🟢 **Safety Status:** SAFE TO SYNTHESIZE — No critical contraindications.")
    elif "REVIEW" in status_upper:
        st.warning("🟡 **Safety Status:** REVIEW REQUIRED — Immunogenomic or cross-modal conflicts require oncologist scrutiny.")
    elif "BLOCK" in status_upper:
        st.error("🔴 **Safety Status:** BLOCKED — Hard pharmacological contraindication or critical data discrepancy.")
    else:
        st.info(f"⚪ **Safety Status:** {safety_status}")

    if warnings:
        with st.expander("⚠️ Clinical Safety Precautions & Warnings", expanded=True):
            for w in warnings:
                st.write(f"- {w}")


def render_treatment_candidate(candidate: Dict[str, Any], index: int) -> None:
    """Renders an individual treatment candidate card with evidence and rationale."""
    name = candidate.get("name", "Unknown Regimen")
    drug_class = candidate.get("drug_class", "Antineoplastic")
    target = candidate.get("target_biomarker", "N/A")
    rationale = candidate.get("rationale", "No rationale provided")
    evidence_ids = candidate.get("evidence_ids", [])
    warnings = candidate.get("safety_warnings", [])

    st.markdown(f"#### Option {index + 1}: **{name}** ({drug_class})")
    st.markdown(f"- **Target Biomarker:** `{target or 'Broad Spectrum'}`")
    st.markdown(f"- **Clinical Rationale:** {rationale}")
    
    if evidence_ids:
        st.markdown(f"- **Supporting Evidence:** {', '.join([f'`{e}`' for e in evidence_ids])}")
    if warnings:
        st.markdown(f"- **Safety Precautions:** {'; '.join(warnings)}")
    st.caption("🔒 *Clinical Decision Support Only — Requires Attending Oncologist Sign-off*")
    st.divider()


def render_audit_timeline(events: List[Dict[str, Any]]) -> None:
    """Renders structured chronological audit history table."""
    if not events:
        st.info("No audit events recorded for this session.")
        return

    table_data = []
    for ev in events:
        table_data.append({
            "Timestamp (UTC)": ev.get("timestamp", "")[:19].replace("T", " "),
            "Event": ev.get("event_type", ""),
            "Component": ev.get("component", ""),
            "Status": ev.get("status", ""),
            "Warnings": len(ev.get("warnings", [])),
        })
    st.dataframe(table_data, use_container_width=True)
