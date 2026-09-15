"""
ReportLab PDF Generator for Stage 6 Clinical Decision Support Report.
Produces professional oncology decision-support documents adhering to
governance and physician-sign-off standards.
"""

import io
from datetime import datetime
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_patient_pdf_report(patient: Dict[str, Any]) -> bytes:
    """Generates a comprehensive precision oncology PDF report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f2b48"),
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0284c7"),
    )
    heading_style = ParagraphStyle(
        "SecHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#dc2626"),
    )

    # 1. Header Banner
    story.append(Paragraph("ONCOLOGY COMMAND CENTER", title_style))
    story.append(Paragraph("Precision Oncology Multidisciplinary Decision Support Report", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12))

    # 2. Patient Demographics Table
    pid = patient.get("patient_id", "UNKNOWN")
    name = patient.get("name", "Unknown Patient")
    age = str(patient.get("age", "N/A"))
    gender = patient.get("gender", "N/A")
    cancer = patient.get("cancer_type", "N/A")
    stage = patient.get("cancer_stage", "N/A")
    date_str = datetime.now().strftime("%B %d, %Y")

    demo_data = [
        [
            Paragraph("<b>Patient Name:</b> " + name, body_style),
            Paragraph("<b>Patient ID:</b> " + pid, body_style),
            Paragraph("<b>Report Date:</b> " + date_str, body_style),
        ],
        [
            Paragraph("<b>Age / Gender:</b> " + f"{age} / {gender}", body_style),
            Paragraph("<b>Diagnosis:</b> " + cancer, body_style),
            Paragraph("<b>Cancer Stage:</b> " + stage, body_style),
        ],
    ]
    t_demo = Table(demo_data, colWidths=[180, 200, 160])
    t_demo.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(t_demo)
    story.append(Spacer(1, 14))

    # 3. Decision Status & Safety Gate
    story.append(Paragraph("I. Clinical Status & Safety Gate Clearance", heading_style))
    story.append(Spacer(1, 4))
    
    risk_level = patient.get("overall_risk_level", "N/A")
    safety_stat = patient.get("safety_status", "REVIEW_REQUIRED")
    confidence = f"{patient.get('agent_confidence', 90)}%"
    dec_stat = patient.get("recommendation_status", "Pending Approval")

    status_data = [
        ["Overall Risk Level", "Safety Status", "Agent Confidence", "Recommendation Status"],
        [risk_level, safety_stat, confidence, dec_stat],
    ]
    t_status = Table(status_data, colWidths=[135, 135, 135, 135])
    t_status.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2b48")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ])
    )
    story.append(t_status)
    story.append(Spacer(1, 14))

    # 4. Clinical Summary & Biomarkers
    story.append(Paragraph("II. Patient Clinical Summary", heading_style))
    story.append(Spacer(1, 4))
    notes = patient.get("clinical_notes", "No clinical notes provided.")
    story.append(Paragraph(notes, body_style))
    story.append(Spacer(1, 10))

    # Biomarkers Table
    b_dict = patient.get("biomarkers", {})
    if b_dict:
        story.append(Paragraph("<b>Assayed Biomarkers & Molecular Profiles:</b>", body_style))
        story.append(Spacer(1, 4))
        b_rows = [["Biomarker", "Assay Value"]]
        for k, v in b_dict.items():
            b_rows.append([str(k), str(v)])
        t_bio = Table(b_rows, colWidths=[200, 340])
        t_bio.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(t_bio)
        story.append(Spacer(1, 14))

    # 5. Agent Decision Rationale
    story.append(Paragraph("III. Multi-Agent Decision Rationale", heading_style))
    story.append(Spacer(1, 4))
    rationale = patient.get("decision_rationale", "Multi-agent consensus deliberated based on guidelines.")
    story.append(Paragraph(rationale, body_style))
    story.append(Spacer(1, 8))

    checklist = patient.get("rationale_checklist", [])
    for item in checklist:
        mark = "[PASS]" if item.get("passed") else "[WARN]"
        story.append(Paragraph(f"• <b>{mark}</b> {item.get('label', '')}", body_style))
    story.append(Spacer(1, 14))

    # 6. Physician Review & Governance Block
    story.append(Paragraph("IV. Attending Physician Review & Sign-Off", heading_style))
    story.append(Spacer(1, 4))
    dec = patient.get("physician_decision", "PENDING")
    override_info = patient.get("physician_override")

    if dec == "APPROVED":
        review_text = "<b>Status:</b> <font color='#059669'><b>APPROVED BY ONCOLOGIST</b></font><br/>Approved for standard multidisciplinary care pathway."
    elif dec == "OVERRIDDEN":
        reason = override_info.get("reason", "Clinical judgment") if override_info else "Physician Departure"
        review_text = f"<b>Status:</b> <font color='#dc2626'><b>PHYSICIAN OVERRIDE RECORDED</b></font><br/><b>Rationale:</b> {reason}"
    else:
        review_text = "<b>Status:</b> <font color='#d97706'><b>AWAITING ATTENDING ONCOLOGIST REVIEW</b></font><br/>Treatment execution is pending physician verification."

    story.append(Paragraph(review_text, body_style))
    story.append(Spacer(1, 18))

    # Signature lines
    sig_data = [
        ["_________________________________________", "_________________________________________"],
        ["Attending Medical Oncologist Signature", "Date / Medical License #"],
    ]
    t_sig = Table(sig_data, colWidths=[270, 270])
    t_sig.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#64748b")),
        ])
    )
    story.append(t_sig)
    story.append(Spacer(1, 20))

    # Mandatory Legal / Regulatory Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=6))
    story.append(
        Paragraph(
            "MANDATORY GOVERNANCE NOTICE: This AI-generated decision support report is strictly an advisory, "
            "investigational tool. It does not constitute a clinical prescription, autonomous diagnostic finding, "
            "or direct medical order. All therapeutic and diagnostic regimens mandate independent verification, "
            "correlation with pathology, and written authorization by the attending medical oncologist.",
            disclaimer_style,
        )
    )

    doc.build(story)
    return buf.getvalue()
