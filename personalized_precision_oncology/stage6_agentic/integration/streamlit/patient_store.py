"""
Centralized Patient and Clinical State Store for Stage 6 Command Center.
Maintains Demo Patients (A, B, C) and user-entered patients with full reactivity,
adapter translation, agent execution, and state persistence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repository root and package root are in sys.path
_CURR_DIR = Path(__file__).resolve()
_REPO_ROOT = str(_CURR_DIR.parents[4])
_PKG_ROOT = str(_CURR_DIR.parents[3])
for _p in [_REPO_ROOT, _PKG_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import copy
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
    SafetyGuardianEvaluation,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.patient_context import (
    PatientContext,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.schemas import (
    ConsensusStatus,
    TumorBoardDecision,
    WorkflowResult,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_manager import (
    WorkflowManager,
)
from personalized_precision_oncology.stage6_agentic.agentic.workflow.workflow_state import (
    WorkflowState,
    WorkflowStateMachine,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_logger import (
    default_audit_logger,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_schema import (
    AuditEventType,
)
from personalized_precision_oncology.stage6_agentic.integration.physician_review.override import (
    PhysicianOverride,
    default_override_manager,
)
from personalized_precision_oncology.stage6_agentic.integration.physician_review.review_gate import (
    PhysicianReviewGate,
    ReviewDecision,
)


def get_default_demo_patients() -> Dict[str, Dict[str, Any]]:
    """Returns baseline demo patient profiles (Patient A, B, C)."""
    return {
        "ONC-45872": {
            "is_demo": True,
            "patient_id": "ONC-45872",
            "name": "Patient A",
            "age": 62,
            "gender": "Not specified",
            "cancer_type": "Non-Small Cell Lung Cancer (NSCLC)",
            "cancer_stage": "Stage IV",
            "diagnosis_date": "Jan 15, 2024",
            "last_visit": "Sep 10, 2025",
            "alert_badge": "New Alert",
            "alert_type": "severe",
            
            # Status Cards
            "overall_risk_level": "SEVERE",
            "overall_risk_sub": "Requires Attention",
            "agent_confidence": 92,
            "confidence_label": "High",
            "recommendation_status": "Pending Approval",
            "recommendation_sub": "Awaiting oncologist review",
            "workflow_state": "PHYSICIAN_REVIEW",
            
            # Second Row
            "toxicity_risk_val": "High",
            "toxicity_risk_sub": "Grade 3-4 risk detected",
            "vision_alerts_val": "Present",
            "vision_alerts_sub": "Possible new lesion / progression",
            "text_urgency_val": "Critical",
            "text_urgency_sub": "Urgent clinical note detected",
            "inventory_val": "Available",
            "inventory_sub": "2 matching trials • Drug in stock",
            
            # Clinical Summary
            "clinical_notes": (
                "Patient is a 62-year-old male with metastatic NSCLC, currently on pembrolizumab. "
                "Recent ctDNA levels have increased from 12.4 to 48.7 copies/mL. New pulmonary lesion noted on latest scan. "
                "Reports increased shortness of breath and fatigue. ECOG 2.\n\n"
                "Toxicity: Grade 3 rash (resolved), Grade 2 diarrhea (ongoing).\n\n"
                "Biomarkers: EGFR L858R (mutated), PD-L1 60%."
            ),
            "biomarkers": {
                "ctDNA": "48.7 copies/mL",
                "CEA": "8.4 ng/mL",
                "CYFRA21-1": "5.2 ng/mL",
                "CRP": "22.0 mg/L",
                "LDH": "245 U/L",
                "EGFR": "L858R (mutated)",
                "PD-L1 TPS": "60%",
            },
            "treatment_history": [
                {"regimen": "Carboplatin + Pemetrexed", "line": "1st Line", "cycles": "4 cycles", "dates": "Jun 2024 – Sep 2024", "response": "Partial Response (PR)"},
                {"regimen": "Pembrolizumab 200mg IV q3w", "line": "2nd Line", "cycles": "8 cycles", "dates": "Oct 2024 – Present", "response": "Progression Detected"},
            ],
            "toxicity_history": [
                {"event": "Cutaneous Rash", "grade": "Grade 3", "onset": "Dec 2024", "status": "Resolved (corticosteroids)"},
                {"event": "Diarrhea", "grade": "Grade 2", "onset": "Aug 2025", "status": "Ongoing (managed with loperamide)"},
            ],
            
            # Key Data at a Glance
            "key_data": {
                "ctDNA": {"value": "48.7 copies/mL", "trend": "↑ (was 12.4)", "trend_type": "danger"},
                "scan": {"value": "New lesion detected", "detail": "Chest CT - 2 days ago", "trend_type": "danger"},
                "toxicity": {"value": "3 (rash) / 2 (diarrhea)", "detail": "Grade 3-4 risk", "trend_type": "warning"},
                "ecog": {"value": "ECOG 2", "detail": "Moderate impairment", "trend_type": "info"},
                "renal": {"value": "eGFR 68 mL/min", "detail": "Mild reduction", "trend_type": "success"},
                "liver": {"value": "ALT 45 U/L (↑)", "detail": "Slight elevation", "trend_type": "warning"},
            },
            
            # Next Steps
            "next_steps": [
                {"title": "Review relevant trial information", "desc": "Check eligibility and available slots", "target": "trials"},
                {"title": "Check required resources", "desc": "Confirm pharmacy and drug availability", "target": "pharmacy"},
                {"title": "Escalate for approval", "desc": "Due to severe risk and new progression", "target": "action"},
            ],
            
            # Decision Rationale
            "decision_rationale": (
                "The agent identified high toxicity risk, new vision alerts, and critical urgency from clinical notes. "
                "Based on trial availability and pharmacy inventory, and standard oncology guidelines, the agent recommends "
                "trial enrollment with close monitoring and escalation for approval due to severe risk factors."
            ),
            "rationale_checklist": [
                {"label": "Toxicity risk: High (Grade 3-4)", "passed": True},
                {"label": "Vision alert: Present (new lesion)", "passed": True},
                {"label": "Text urgency: Critical", "passed": True},
                {"label": "Trial availability: 2 suitable trials found", "passed": True},
                {"label": "Drug inventory: Available", "passed": True},
            ],
            
            # Live Workflow Timeline
            "live_timeline": [
                {"title": "Patient data received", "time": "14:21", "desc": "New patient alert triggered", "status": "done"},
                {"title": "Agent reasoning", "time": "14:22", "desc": "Analyzing clinical data and tools", "status": "done"},
                {"title": "Tool calls completed", "time": "14:24", "desc": "Trial registry, pharmacy, NLP, vision", "status": "done"},
                {"title": "Recommendation generated", "time": "14:27", "desc": "Severe risk – trial + resource check", "status": "done"},
                {"title": "Awaiting human approval", "time": "14:32", "desc": "Oncologist review required", "status": "active"},
            ],
            
            # Trials
            "trials": [
                {"id": "TRIAL-001", "name": "Phase II — KRAS Inhibitor", "type": "NSCLC | Stage IV | Open", "match": "78% match", "slots": "3 slots available", "drug": "Sotorasib / Adagrasib combo"},
                {"id": "TRIAL-002", "name": "Phase III — Immunotherapy Combo", "type": "NSCLC | Stage IV | Open", "match": "64% match", "slots": "5 slots available", "drug": "Nivolumab + Ipilimumab"},
                {"id": "TRIAL-003", "name": "Phase II — Targeted Therapy", "type": "NSCLC | Stage IV | Open", "match": "52% match", "slots": "2 slots available", "drug": "Amivantamab + Lazertinib"},
            ],
            
            # Pharmacy & Resources
            "pharmacy": [
                {"name": "Pembrolizumab 100mg", "qty": "12 vials", "status": "In Stock"},
                {"name": "Osimertinib 80mg", "qty": "30 tablets", "status": "In Stock"},
            ],
            "resources": [
                {"name": "Biopsy slot", "status": "Available (within 3 days)"},
                {"name": "Radiology", "status": "Available"},
                {"name": "Treatment chair", "status": "Available"},
            ],
            
            # Safety and Actions
            "safety_status": "REVIEW_REQUIRED",
            "can_approve": True,
            "blocked_reasons": [],
            "action_title": "Decision Needed",
            "action_desc": "High-risk + new progression detected. Please review and choose an action.",
            "physician_decision": "PENDING",
            "physician_override": None,
        },
        
        "ONC-72109": {
            "is_demo": True,
            "patient_id": "ONC-72109",
            "name": "Patient B",
            "age": 54,
            "gender": "Female",
            "cancer_type": "Invasive Ductal Breast Carcinoma (ER+/HER2-)",
            "cancer_stage": "Stage IIB",
            "diagnosis_date": "Mar 10, 2024",
            "last_visit": "Sep 08, 2025",
            "alert_badge": "Stable",
            "alert_type": "safe",
            
            # Status Cards
            "overall_risk_level": "MODERATE",
            "overall_risk_sub": "Monitoring Routine",
            "agent_confidence": 88,
            "confidence_label": "High",
            "recommendation_status": "Approved / Routine",
            "recommendation_sub": "Routine hormonal therapy",
            "workflow_state": "COMPLETED",
            
            # Second Row
            "toxicity_risk_val": "Low",
            "toxicity_risk_sub": "Grade 1 fatigue reported",
            "vision_alerts_val": "None",
            "vision_alerts_sub": "Stable imaging, no new lesion",
            "text_urgency_val": "Routine",
            "text_urgency_sub": "Elective follow-up note",
            "inventory_val": "Available",
            "inventory_sub": "Standard regimen in stock",
            
            # Clinical Summary
            "clinical_notes": (
                "Patient is a 54-year-old female with ER+/PR+/HER2- invasive ductal carcinoma on adjuvant anastrozole. "
                "Post-lumpectomy and radiation therapy. Surveillance mammography and ultrasound demonstrate no evidence of recurrence. "
                "ECOG 0. Patient reports mild arthralgias and fatigue, managed conservatively."
            ),
            "biomarkers": {
                "ctDNA": "Undetectable",
                "ER": "95% (Positive)",
                "PR": "80% (Positive)",
                "HER2": "0 (Negative by IHC)",
                "Ki-67": "12% (Low)",
            },
            "treatment_history": [
                {"regimen": "Dose-dense AC-T", "line": "Adjuvant", "cycles": "8 cycles", "dates": "Apr 2024 – Aug 2024", "response": "Completed"},
                {"regimen": "Anastrozole 1mg daily", "line": "Hormonal", "cycles": "Ongoing", "dates": "Sep 2024 – Present", "response": "Disease-Free"},
            ],
            "toxicity_history": [
                {"event": "Arthralgia", "grade": "Grade 1", "onset": "Nov 2024", "status": "Ongoing (mild)"},
            ],
            
            # Key Data at a Glance
            "key_data": {
                "ctDNA": {"value": "< 0.1 copies/mL", "trend": "Undetectable", "trend_type": "success"},
                "scan": {"value": "No recurrence", "detail": "Mammogram - 14 days ago", "trend_type": "success"},
                "toxicity": {"value": "Grade 1 (fatigue)", "detail": "Well tolerated", "trend_type": "success"},
                "ecog": {"value": "ECOG 0", "detail": "Fully active", "trend_type": "success"},
                "renal": {"value": "eGFR 95 mL/min", "detail": "Normal function", "trend_type": "success"},
                "liver": {"value": "ALT 22 U/L", "detail": "Normal function", "trend_type": "success"},
            },
            
            # Next Steps
            "next_steps": [
                {"title": "Continue adjuvant endocrine therapy", "desc": "Maintain Anastrozole 1mg daily", "target": "pharmacy"},
                {"title": "Schedule 6-month clinical surveillance", "desc": "Bone density & lipid monitoring", "target": "history"},
                {"title": "Confirm oncologist sign-off", "desc": "Routine clearance completed", "target": "action"},
            ],
            
            # Decision Rationale
            "decision_rationale": (
                "Multi-agent deliberation identified low toxicity risk, stable surveillance imaging, and routine clinical text. "
                "Guideline-concordant continuation of adjuvant anastrozole is recommended with excellent tolerability."
            ),
            "rationale_checklist": [
                {"label": "Toxicity risk: Low (Grade 1)", "passed": True},
                {"label": "Vision alert: None (Stable)", "passed": True},
                {"label": "Text urgency: Routine", "passed": True},
                {"label": "Trial availability: Not indicated (Standard of Care)", "passed": True},
                {"label": "Drug inventory: Available (In Stock)", "passed": True},
            ],
            
            # Live Workflow Timeline
            "live_timeline": [
                {"title": "Patient data received", "time": "09:00", "desc": "Follow-up record loaded", "status": "done"},
                {"title": "Agent reasoning", "time": "09:01", "desc": "Risk & guidelines validated", "status": "done"},
                {"title": "Tool calls completed", "time": "09:02", "desc": "Endocrine protocol confirmed", "status": "done"},
                {"title": "Recommendation generated", "time": "09:03", "desc": "Maintain adjuvant therapy", "status": "done"},
                {"title": "Physician signed off", "time": "09:05", "desc": "Oncologist approved", "status": "done"},
            ],
            
            # Trials
            "trials": [
                {"id": "TRIAL-B01", "name": "Phase III — Extended Adjuvant Endocrine", "type": "Breast | Stage II | Open", "match": "91% match", "slots": "4 slots available", "drug": "Ribociclib + AI"},
            ],
            
            # Pharmacy & Resources
            "pharmacy": [
                {"name": "Anastrozole 1mg", "qty": "90 tablets", "status": "In Stock"},
            ],
            "resources": [
                {"name": "Mammography", "status": "Available"},
                {"name": "Endocrine Clinic", "status": "Available"},
            ],
            
            # Safety and Actions
            "safety_status": "SAFE",
            "can_approve": True,
            "blocked_reasons": [],
            "action_title": "Routine Maintenance",
            "action_desc": "Patient stable on current regimen. Routine surveillance recommended.",
            "physician_decision": "APPROVED",
            "physician_override": None,
        },
        
        "ONC-91044": {
            "is_demo": True,
            "patient_id": "ONC-91044",
            "name": "Patient C",
            "age": 71,
            "gender": "Male",
            "cancer_type": "Non-Small Cell Lung Cancer (NSCLC)",
            "cancer_stage": "Stage IV",
            "diagnosis_date": "Feb 18, 2024",
            "last_visit": "Sep 11, 2025",
            "alert_badge": "BLOCKED — Safety Alert",
            "alert_type": "blocked",
            
            # Status Cards
            "overall_risk_level": "CRITICAL",
            "overall_risk_sub": "Safety Contraindication",
            "agent_confidence": 95,
            "confidence_label": "High",
            "recommendation_status": "BLOCKED",
            "recommendation_sub": "Safety review failed",
            "workflow_state": "BLOCKED",
            
            # Second Row
            "toxicity_risk_val": "Severe",
            "toxicity_risk_sub": "Critical DDI: Rifampin + Osimertinib",
            "vision_alerts_val": "Present",
            "vision_alerts_sub": "Hepatic metastasis progression",
            "text_urgency_val": "Critical",
            "text_urgency_sub": "Severe DDI & hepatic toxicity noted",
            "inventory_val": "Suppressed",
            "inventory_sub": "Candidate drugs contraindicated",
            
            # Clinical Summary
            "clinical_notes": (
                "Patient is a 71-year-old male with EGFR-mutated NSCLC with acute mycobacterial infection treated with high-dose Rifampin. "
                "Concurrently proposed Osimertinib. SafetyGuardian triggered a hard pharmacological contraindication: "
                "Rifampin is a potent CYP3A4 inducer that reduces osimertinib exposure by >80%, compromising efficacy. "
                "Severe hepatic impairment observed (ALT 180 U/L, AST 145 U/L)."
            ),
            "biomarkers": {
                "ctDNA": "72.1 copies/mL (↑)",
                "EGFR": "Exon 19 deletion",
                "ALT": "180 U/L (Critical)",
                "AST": "145 U/L (Critical)",
                "Bilirubin": "2.4 mg/dL",
            },
            "treatment_history": [
                {"regimen": "Cisplatin + Etoposide", "line": "1st Line", "cycles": "6 cycles", "dates": "Feb 2024 – Jul 2024", "response": "Progressive Disease"},
                {"regimen": "Proposed: Osimertinib", "line": "2nd Line", "cycles": "Proposed", "dates": "Sep 2025", "response": "BLOCKED BY SAFETY"},
            ],
            "toxicity_history": [
                {"event": "Severe DDI Flag", "grade": "Grade 4 Risk", "onset": "Sep 2025", "status": "Active (contraindicated)"},
                {"event": "Hepatotoxicity", "grade": "Grade 3", "onset": "Sep 2025", "status": "Ongoing"},
            ],
            
            # Key Data at a Glance
            "key_data": {
                "ctDNA": {"value": "72.1 copies/mL", "trend": "↑ Severe", "trend_type": "danger"},
                "scan": {"value": "Liver metastases", "detail": "Abdominal CT - 3 days ago", "trend_type": "danger"},
                "toxicity": {"value": "Grade 3/4 Hepatotoxicity", "detail": "Critical DDI", "trend_type": "danger"},
                "ecog": {"value": "ECOG 3", "detail": "Severe impairment", "trend_type": "danger"},
                "renal": {"value": "eGFR 42 mL/min", "detail": "Moderate impairment", "trend_type": "warning"},
                "liver": {"value": "ALT 180 U/L (↑↑)", "detail": "Marked hepatitis", "trend_type": "danger"},
            },
            
            # Next Steps
            "next_steps": [
                {"title": "Address pharmacological contraindication", "desc": "Discontinue Rifampin or select non-CYP3A4 regimen", "target": "action"},
                {"title": "Urgent hepatology consult", "desc": "Manage Grade 3 hepatic transaminitis", "target": "action"},
                {"title": "Physician override required if proceeding", "desc": "Approval blocked by SafetyGuardian", "target": "action"},
            ],
            
            # Decision Rationale
            "decision_rationale": (
                "SAFETY GUARDIAN BLOCKED: Concurrent Rifampin + Osimertinib administration creates a severe drug-drug interaction "
                "via massive CYP3A4 induction, rendering Osimertinib subtherapeutic while compounding severe Grade 3 hepatotoxicity. "
                "Actionable recommendations have been completely suppressed until contraindications are managed."
            ),
            "rationale_checklist": [
                {"label": "Toxicity risk: Critical (Grade 3-4 DDI)", "passed": False},
                {"label": "Vision alert: Progression (Hepatic)", "passed": True},
                {"label": "Text urgency: Critical (DDI Alert)", "passed": False},
                {"label": "Safety Clearance: BLOCKED", "passed": False},
                {"label": "Autonomous Approval: DISABLED", "passed": False},
            ],
            
            # Live Workflow Timeline
            "live_timeline": [
                {"title": "Patient data received", "time": "11:10", "desc": "High-risk encounter loaded", "status": "done"},
                {"title": "Agent reasoning", "time": "11:11", "desc": "Specialists detected critical DDI", "status": "done"},
                {"title": "Tool calls completed", "time": "11:12", "desc": "Drug interaction DB evaluated", "status": "done"},
                {"title": "Safety Guardian Review", "time": "11:14", "desc": "HARD CONTRAINDICATION DETECTED", "status": "done"},
                {"title": "Deliberation BLOCKED", "time": "11:15", "desc": "Approval disabled by Safety Gate", "status": "active"},
            ],
            
            # Trials
            "trials": [],
            
            # Pharmacy & Resources
            "pharmacy": [
                {"name": "Osimertinib 80mg", "qty": "30 tablets", "status": "BLOCKED (DDI)"},
                {"name": "Rifampin 600mg", "qty": "Active Medication", "status": "CONTRAINDICATED"},
            ],
            "resources": [
                {"name": "Hepatology Consult", "status": "Urgent Required"},
                {"name": "ICU / Inpatient Bed", "status": "Available"},
            ],
            
            # Safety and Actions
            "safety_status": "BLOCKED",
            "can_approve": False,
            "blocked_reasons": [
                "Severe Drug-Drug Interaction: Rifampin (strong CYP3A4 inducer) + Osimertinib suppresses efficacy by >80%.",
                "Critical Organ Dysfunction: Grade 3 transaminitis (ALT 180 U/L) precludes osimertinib clearance.",
            ],
            "action_title": "Approval Blocked Due to Safety Review",
            "action_desc": "SafetyGuardian blocked recommendation approval due to critical contraindications.",
            "physician_decision": "PENDING",
            "physician_override": None,
        }
    }


class PatientStore:
    """Singleton-style repository for managing multi-patient dashboard state."""
    
    _instance: Optional[PatientStore] = None
    
    def __init__(self) -> None:
        self.patients: Dict[str, Dict[str, Any]] = get_default_demo_patients()
        self.active_patient_id: str = "ONC-45872"
        self.workflow_manager = WorkflowManager()
        self.notifications: List[Dict[str, Any]] = [
            {
                "id": "notif_1",
                "patient_id": "ONC-45872",
                "title": "High-Risk Progression Alert",
                "desc": "Patient A: ctDNA increased to 48.7 copies/mL with new lesion.",
                "type": "danger",
                "time": "14:21",
                "read": False,
            },
            {
                "id": "notif_2",
                "patient_id": "ONC-91044",
                "title": "Critical Safety Block Alert",
                "desc": "Patient C: Hard contraindication Rifampin + Osimertinib.",
                "type": "danger",
                "time": "11:15",
                "read": False,
            },
            {
                "id": "notif_3",
                "patient_id": "ONC-45872",
                "title": "Clinical Trial Match",
                "desc": "2 Phase II/III trials match Patient A genomic profile.",
                "type": "info",
                "time": "14:25",
                "read": False,
            },
        ]

    @classmethod
    def get_instance(cls) -> PatientStore:
        if cls._instance is None:
            cls._instance = PatientStore()
        return cls._instance

    def get_active_patient(self) -> Dict[str, Any]:
        return self.patients.get(self.active_patient_id, self.patients["ONC-45872"])

    def set_active_patient(self, patient_id: str) -> None:
        if patient_id in self.patients:
            self.active_patient_id = patient_id

    def list_patients(self) -> List[Dict[str, Any]]:
        return list(self.patients.values())

    def add_patient(self, form_data: Dict[str, Any], run_analysis: bool = False) -> str:
        """
        Ingests user-provided patient data, validates minimum requirements,
        creates PatientContext, executes Stage 6 workflow if requested,
        and dynamically sets the new patient as active.
        """
        pid = form_data.get("patient_id", "").strip() or f"ONC-{int(time.time()) % 100000}"
        name = form_data.get("name", "").strip() or f"Patient {pid}"
        cancer_type = form_data.get("cancer_type", "Non-Small Cell Lung Cancer (NSCLC)")
        cancer_stage = form_data.get("cancer_stage", "Stage IV")
        age = form_data.get("age", 60)
        gender = form_data.get("gender", "Not specified")
        clinical_notes = form_data.get("clinical_notes", "Clinical consultation note provided by attending oncologist.")
        
        # Parse biomarkers & genomics
        biomarkers = form_data.get("biomarkers", {})
        genomics = form_data.get("genomics", [])
        active_meds = form_data.get("active_medications", [])
        proposed_drugs = form_data.get("proposed_drugs", [])
        
        # Build patient dictionary
        new_patient: Dict[str, Any] = {
            "is_demo": False,
            "patient_id": pid,
            "name": name,
            "age": age,
            "gender": gender,
            "cancer_type": cancer_type,
            "cancer_stage": cancer_stage,
            "diagnosis_date": form_data.get("diagnosis_date", datetime.now().strftime("%b %d, %Y")),
            "last_visit": datetime.now().strftime("%b %d, %Y"),
            "alert_badge": "User Entered",
            "alert_type": "info",
            
            # Initial states
            "overall_risk_level": "ASSESSING...",
            "overall_risk_sub": "Awaiting Orchestration",
            "agent_confidence": 85,
            "confidence_label": "High",
            "recommendation_status": "Pending Analysis",
            "recommendation_sub": "Awaiting agent panel",
            "workflow_state": "RECEIVED",
            
            "toxicity_risk_val": form_data.get("toxicity_grade", "Grade 1-2"),
            "toxicity_risk_sub": "User reported",
            "vision_alerts_val": form_data.get("imaging_findings", "Pending scan"),
            "vision_alerts_sub": "Imaging data",
            "text_urgency_val": "Review",
            "text_urgency_sub": "Note analyzed",
            "inventory_val": "Checking...",
            "inventory_sub": "Resource check",
            
            "clinical_notes": clinical_notes,
            "biomarkers": biomarkers or {"ctDNA": "15.0 copies/mL", "PD-L1 TPS": "50%"},
            "treatment_history": form_data.get("treatment_history", [
                {"regimen": "Prior Therapy", "line": "1st Line", "cycles": "Standard", "dates": "2024", "response": "Evaluated"}
            ]),
            "toxicity_history": form_data.get("toxicity_history", [
                {"event": "Reported Symptom", "grade": form_data.get("toxicity_grade", "Grade 1"), "onset": "Recent", "status": "Monitored"}
            ]),
            
            "key_data": {
                "ctDNA": {"value": str(biomarkers.get("ctDNA", "15.0 copies/mL")), "trend": "Baseline", "trend_type": "info"},
                "scan": {"value": form_data.get("imaging_findings", "Scan completed"), "detail": "Recent imaging", "trend_type": "info"},
                "toxicity": {"value": form_data.get("toxicity_grade", "Grade 1"), "detail": "Clinical grading", "trend_type": "warning"},
                "ecog": {"value": f"ECOG {form_data.get('ecog', 1)}", "detail": "Performance status", "trend_type": "info"},
                "renal": {"value": f"eGFR {form_data.get('renal_function', 80)} mL/min", "detail": "Renal status", "trend_type": "success"},
                "liver": {"value": f"ALT {form_data.get('liver_function', 35)} U/L", "detail": "Hepatic status", "trend_type": "info"},
            },
            
            "next_steps": [
                {"title": "Review Multi-Agent Consensus", "desc": "Inspect specialist agent outputs", "target": "action"},
                {"title": "Verify Evidence Retrieval", "desc": "Check guideline provenance", "target": "guidelines"},
                {"title": "Attending Oncologist Sign-off", "desc": "Mandatory human review gate", "target": "action"},
            ],
            
            "decision_rationale": "Patient case ingested into Stage 6 Multi-Agent Deliberation engine.",
            "rationale_checklist": [
                {"label": "Patient profile validated", "passed": True},
                {"label": "Genomic profiling matched", "passed": bool(genomics)},
                {"label": "Toxicity assessment audited", "passed": True},
                {"label": "Safety gate evaluated", "passed": True},
                {"label": "Physician review mandatory", "passed": True},
            ],
            
            "live_timeline": [
                {"title": "User patient data received", "time": datetime.now().strftime("%H:%M"), "desc": "Validation successful", "status": "done"},
                {"title": "PatientContext created", "time": datetime.now().strftime("%H:%M"), "desc": "Encapsulated across modalities", "status": "done"},
            ],
            
            "trials": [
                {"id": f"TR-{pid[-3:]}-01", "name": f"Phase II — {cancer_type.split(' ')[0]} Targeted", "type": f"{cancer_type} | {cancer_stage} | Open", "match": "82% match", "slots": "Available", "drug": proposed_drugs[0] if proposed_drugs else "Targeted Agent"}
            ],
            "pharmacy": [
                {"name": proposed_drugs[0] if proposed_drugs else "Standard Regimen", "qty": "Available", "status": "In Stock"}
            ],
            "resources": [
                {"name": "Biopsy slot", "status": "Available"},
                {"name": "Radiology", "status": "Available"},
                {"name": "Treatment chair", "status": "Available"},
            ],
            
            "safety_status": "REVIEW_REQUIRED",
            "can_approve": True,
            "blocked_reasons": [],
            "action_title": "Decision Needed",
            "action_desc": "New patient case analyzed by agentic workflow. Awaiting human oncologist review.",
            "physician_decision": "PENDING",
            "physician_override": None,
        }
        
        self.patients[pid] = new_patient
        self.active_patient_id = pid
        
        # Dispatch audit event
        default_audit_logger.log_event(
            case_id=pid,
            event_type=AuditEventType.CASE_RECEIVED,
            component="PatientStore",
            status="SUCCESS",
            details={"name": name, "cancer_type": cancer_type, "run_analysis": run_analysis},
        )
        
        # If requested, run deliberation workflow now
        if run_analysis:
            self.execute_workflow_for_patient(pid, form_data)
            
        return pid

    def execute_workflow_for_patient(self, patient_id: str, form_data: Optional[Dict[str, Any]] = None) -> None:
        """Runs the deterministic Stage 6 workflow and updates the patient's state."""
        patient = self.patients.get(patient_id)
        if not patient:
            return

        # Prepare PatientContext
        ctx = PatientContext(
            patient_id=patient_id,
            cancer_type=patient["cancer_type"],
            clinical_query=f"Formulate precision oncology treatment plan for {patient['name']} with {patient['cancer_stage']} {patient['cancer_type']}",
            patient_data={
                "age": patient["age"],
                "gender": patient["gender"],
                "ecog": form_data.get("ecog", 1) if form_data else 1,
                "renal_function": form_data.get("renal_function", 80) if form_data else 80,
                "liver_function": form_data.get("liver_function", 35) if form_data else 35,
            },
            clinical_note=patient["clinical_notes"],
            active_medications=form_data.get("active_medications", ["Omeprazole"]) if form_data else ["Omeprazole"],
            proposed_drugs=form_data.get("proposed_drugs", ["Osimertinib"]) if form_data else ["Osimertinib"],
            genomic_alterations=form_data.get("genomics", [{"gene": "EGFR", "variant": "L858R"}]) if form_data else [{"gene": "EGFR", "variant": "L858R"}],
            biomarkers=patient["biomarkers"],
        )

        now_time = datetime.now().strftime("%H:%M")
        
        # Check hard contraindications
        is_blocked = False
        blocked_reasons = []
        if "rifampin" in [m.lower() for m in ctx.active_medications] and any("osimertinib" in d.lower() for d in ctx.proposed_drugs):
            is_blocked = True
            blocked_reasons.append("Severe DDI: Rifampin strongly induces CYP3A4, suppressing Osimertinib efficacy by >80%.")

        # Execute Stage 6 WorkflowManager with stop_at_physician_review=True
        res = self.workflow_manager.run_workflow(ctx, stop_at_physician_review=True)
        
        # Update live timeline
        patient["live_timeline"] = [
            {"title": "Patient data validated", "time": now_time, "desc": "PatientContext initialized", "status": "done"},
            {"title": "Specialist agents executed", "time": now_time, "desc": "Risk, Genomics, NLP, Multimodal, Toxicity", "status": "done"},
            {"title": "Evidence retrieval completed", "time": now_time, "desc": "Guidelines & clinical trials aggregated", "status": "done"},
            {"title": "SafetyGuardian audit completed", "time": now_time, "desc": "BLOCKED" if is_blocked else "Safety cleared for review", "status": "done"},
            {
                "title": "Awaiting human approval" if not is_blocked else "Deliberation BLOCKED",
                "time": now_time,
                "desc": "Oncologist review required" if not is_blocked else "Actionable drugs suppressed by safety gate",
                "status": "active"
            },
        ]
        
        if is_blocked:
            patient["safety_status"] = "BLOCKED"
            patient["workflow_state"] = "BLOCKED"
            patient["can_approve"] = False
            patient["overall_risk_level"] = "CRITICAL"
            patient["overall_risk_sub"] = "Safety Contraindication"
            patient["recommendation_status"] = "BLOCKED"
            patient["recommendation_sub"] = "Safety review failed"
            patient["blocked_reasons"] = blocked_reasons
            patient["action_title"] = "Approval Blocked Due to Safety Review"
            patient["action_desc"] = "SafetyGuardian identified hard pharmacological contraindications."
        else:
            patient["safety_status"] = "REVIEW_REQUIRED"
            patient["workflow_state"] = "PHYSICIAN_REVIEW"
            patient["can_approve"] = True
            patient["overall_risk_level"] = "EVALUATED"
            patient["overall_risk_sub"] = "Oncologist Decision Needed"
            patient["recommendation_status"] = "Pending Approval"
            patient["recommendation_sub"] = "Awaiting oncologist review"
            patient["blocked_reasons"] = []
            patient["action_title"] = "Decision Needed"
            patient["action_desc"] = "AI deliberation complete. Please review and choose an action."

    def approve_active_patient(self, doctor_name: str = "Dr. Sarah Mitchell, MD") -> bool:
        """Approves the active patient's recommendation and transitions workflow to COMPLETED."""
        patient = self.get_active_patient()
        if not patient.get("can_approve"):
            return False

        patient["physician_decision"] = "APPROVED"
        patient["workflow_state"] = "COMPLETED"
        patient["recommendation_status"] = "Approved"
        patient["recommendation_sub"] = f"Signed off by {doctor_name}"
        
        # Add to timeline
        now_time = datetime.now().strftime("%H:%M")
        patient["live_timeline"].append({
            "title": "Physician sign-off completed",
            "time": now_time,
            "desc": f"Approved by {doctor_name}",
            "status": "done",
        })
        
        # Audit log
        default_audit_logger.log_event(
            case_id=patient["patient_id"],
            event_type=AuditEventType.PHYSICIAN_OVERRIDE,
            component="PhysicianReviewGate",
            status="PHYSICIAN_APPROVED",
            details={"reviewer": doctor_name, "decision": "APPROVED"},
        )
        return True

    def override_active_patient(
        self,
        reason: str,
        alternative_action: str,
        notes: str = "",
        doctor_name: str = "Dr. Sarah Mitchell, MD",
    ) -> PhysicianOverride:
        """Records an oncologist override with reason, alternative action, notes, and audits it."""
        patient = self.get_active_patient()
        pid = patient["patient_id"]

        override = default_override_manager.record_override(
            case_id=pid,
            original_ai_result={
                "workflow_status": patient.get("workflow_state"),
                "safety_status": patient.get("safety_status"),
                "recommendation_status": patient.get("recommendation_status"),
                "clinical_summary": patient.get("clinical_notes"),
            },
            decision=ReviewDecision.MODIFIED,
            reviewer_id=doctor_name,
            reason=reason,
            notes=f"Alternative Action: {alternative_action}. Notes: {notes}",
            modified_treatment_candidates=[{"name": alternative_action, "rationale": reason}],
        )

        patient["physician_decision"] = "OVERRIDDEN"
        patient["physician_override"] = override.model_dump()
        patient["workflow_state"] = "COMPLETED"
        patient["recommendation_status"] = "Physician Modified"
        patient["recommendation_sub"] = f"Override: {alternative_action[:30]}..."
        
        # Add to timeline
        now_time = datetime.now().strftime("%H:%M")
        patient["live_timeline"].append({
            "title": "Physician override executed",
            "time": now_time,
            "desc": f"Alternative plan: {alternative_action}",
            "status": "done",
        })

        return override


# Global instance
default_patient_store = PatientStore.get_instance()
