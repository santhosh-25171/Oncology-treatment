import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Stage 1 Tabular ML — End-to-End Data Engineering Architecture Report")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "Personalized Precision Medicine for Oncology Treatment Optimization")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate Blue
    accent_color = colors.HexColor("#319795")    # Teal accent
    dark_text = colors.HexColor("#2D3748")       # Dark Charcoal
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=dark_text,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'CodeCustom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
        spaceAfter=6
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=body_style,
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2C5282")
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=body_style,
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=body_style,
        fontSize=8.5,
        leading=11,
        spaceAfter=0
    )

    story = []
    
    # Header Banner / Title Block
    story.append(Paragraph("Stage 1 Tabular Machine Learning", title_style))
    story.append(Paragraph("End-to-End Data Engineering Architecture & Operational Technical Specification", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=primary_color, spaceAfter=15))
    
    # Executive Summary Block
    callout_data = [[
        Paragraph(
            "<b>EXECUTIVE SUMMARY & ROLE SCOPE:</b><br/>"
            "As the <b>Data Engineer in Stage 1 Tabular Machine Learning</b>, the core responsibility is designing, building, and maintaining an enterprise-grade, reproducible data infrastructure. "
            "This encompasses raw clinical data ingestion, patient-level zero-leakage splitting, automated missing data imputation, domain-specific feature engineering (producing 66 numerical/categorical features), feature selection, and real-time transformation serving for model training and deployment.",
            callout_style
        )
    ]]
    callout_table = Table(callout_data, colWidths=[504])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#BEE3F8")),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 15))
    
    # Section 1: Architecture & Pipeline Overview
    story.append(Paragraph("1. Data Pipeline System Architecture", h1_style))
    story.append(Paragraph(
        "The Stage 1 data pipeline processes raw patient clinical records and converts them into normalized, encoded feature matrices consumed by XGBoost, CatBoost, and Random Forest classifiers. Below is the end-to-end flow:",
        body_style
    ))
    
    pipeline_steps = [
        "<b>Raw Clinical Ingestion:</b> Pulling patient demographics, laboratory vitals, comorbidities, and therapy dosage logs.",
        "<b>Group-Stratified Splitter:</b> Enforcing patient-level separation to isolate multi-visit records strictly across splits.",
        "<b>Missing Value Imputation:</b> Dual-strategy median/mode imputers avoiding statistical data leakage.",
        "<b>Domain Feature Factory:</b> Generating 66 domain-engineered features (clinical risk indices, interaction terms, dosage intensity).",
        "<b>Feature Selection & Variance Pruning:</b> Removing constant features via VarianceThreshold.",
        "<b>Encoding & Scaling:</b> One-Hot Encoding categorical descriptors and StandardScaling continuous lab variables.",
        "<b>Artifact Serialization:</b> Exporting reusable transformers (`.joblib`) for zero-latency API inference."
    ]
    for step in pipeline_steps:
        story.append(Paragraph(f"• {step}", bullet_style))
        
    story.append(Spacer(1, 12))

    # Section 2: Data Ingestion & Zero-Leakage Splitting
    story.append(Paragraph("2. Data Ingestion & Zero-Leakage Partitioning", h1_style))
    story.append(Paragraph(
        "Longitudinal medical datasets frequently contain multiple visits per patient. A naive random split leads to severe patient leakage, inflating validation metrics artificially. "
        "The Data Engineer implemented a patient-stratified partitioning strategy:",
        body_style
    ))
    
    split_table_data = [
        [Paragraph("Partition", table_header_style), Paragraph("Patient Count", table_header_style), Paragraph("Record Count", table_header_style), Paragraph("Percentage", table_header_style), Paragraph("Leakage Guarantee", table_header_style)],
        [Paragraph("Train Set", table_cell_style), Paragraph("2,560", table_cell_style), Paragraph("7,896", table_cell_style), Paragraph("80.0%", table_cell_style), Paragraph("0.00% Patient Overlap", table_cell_style)],
        [Paragraph("Validation Set", table_cell_style), Paragraph("320", table_cell_style), Paragraph("1,010", table_cell_style), Paragraph("10.0%", table_cell_style), Paragraph("0.00% Patient Overlap", table_cell_style)],
        [Paragraph("Holdout Test Set", table_cell_style), Paragraph("320", table_cell_style), Paragraph("950 / 750 (Unseen)", table_cell_style), Paragraph("10.0%", table_cell_style), Paragraph("0.00% Patient Overlap", table_cell_style)],
    ]
    split_table = Table(split_table_data, colWidths=[90, 80, 110, 84, 140])
    split_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(split_table)
    story.append(Spacer(1, 12))

    # Section 3: Data Cleaning & Missing Value Imputation
    story.append(Paragraph("3. Data Cleaning & Imputation Infrastructure", h1_style))
    story.append(Paragraph(
        "Clinical raw data exhibits missing values due to uncollected lab panels or varying hospital protocols. "
        "The Data Engineer built a dual-channel imputation engine using Scikit-Learn transformers fit strictly on training partitions:",
        body_style
    ))
    
    story.append(Paragraph("<b>1. Numerical Variables (Median Imputation):</b>", h2_style))
    story.append(Paragraph(
        "Continuous variables such as age, BMI, blood pressure, ALT/AST liver enzymes, and tumor size are imputed using <code>SimpleImputer(strategy='median')</code>. Median is selected over mean to remain robust against extreme physiological outliers.",
        body_style
    ))
    
    story.append(Paragraph("<b>2. Categorical Variables (Mode Imputation):</b>", h2_style))
    story.append(Paragraph(
        "Discrete descriptors such as tumor stage, histological grade, and treatment regimen are imputed using <code>SimpleImputer(strategy='most_frequent')</code>.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 4: Deep Dive into 66 Feature Engineering Pipeline
    story.append(Paragraph("4. Oncology Feature Engineering Framework", h1_style))
    story.append(Paragraph(
        "The centerpiece of Stage 1 ML is <code>feature_engineering.py</code>, which translates raw physiological metrics into <b>66 clinically meaningful features</b>. The key engineered domains include:",
        body_style
    ))
    
    feat_table_data = [
        [Paragraph("Feature Domain", table_header_style), Paragraph("Engineered Features / Derivation Formula", table_header_style), Paragraph("Clinical Rationale", table_header_style)],
        [
            Paragraph("<b>Age Binning</b>", table_cell_style),
            Paragraph("<code>age_group</code>: (&lt;50, 50-65, &gt;65)", table_cell_style),
            Paragraph("Captures non-linear risk escalation in elderly oncology cohorts.", table_cell_style)
        ],
        [
            Paragraph("<b>BMI Profiling</b>", table_cell_style),
            Paragraph("<code>bmi_category</code>: (underweight, normal, overweight, obese)", table_cell_style),
            Paragraph("Accounts for cachexia vs obesity-related toxicity risks.", table_cell_style)
        ],
        [
            Paragraph("<b>Tumor Size Bins</b>", table_cell_style),
            Paragraph("<code>tumor_size_category</code>: T1 (&le;2cm), T2 (2-5cm), T3 (&gt;5cm)", table_cell_style),
            Paragraph("Standardized AJCC tumor T-stage categorization.", table_cell_style)
        ],
        [
            Paragraph("<b>Treatment Intensity</b>", table_cell_style),
            Paragraph("<code>treatment_intensity = dose / duration</code>", table_cell_style),
            Paragraph("Measures daily therapeutic burden and potential drug toxicity.", table_cell_style)
        ],
        [
            Paragraph("<b>Clinical Composite Risk</b>", table_cell_style),
            Paragraph("<code>high_clinical_risk</code>: Binary flag (High Comorbidity & Poor ECOG)", table_cell_style),
            Paragraph("Identifies frailty in patients with multi-organ comorbidities.", table_cell_style)
        ],
        [
            Paragraph("<b>Biomarker Interactions</b>", table_cell_style),
            Paragraph("<code>biomarker_interaction = biomarker_1 * biomarker_2</code>", table_cell_style),
            Paragraph("Models synergistic gene expression interactions.", table_cell_style)
        ]
    ]
    feat_table = Table(feat_table_data, colWidths=[110, 200, 194])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 14))

    # Section 5: Preprocessing & Serialization
    story.append(Paragraph("5. Preprocessing, Encoding & Serialization", h1_style))
    story.append(Paragraph(
        "After feature creation, data undergoes rigid statistical formatting before being saved as reusable pipeline artifacts:",
        body_style
    ))
    
    preproc_points = [
        "<b>Variance Thresholding:</b> <code>VarianceThreshold(threshold=0.0)</code> drops zero-variance constant features.",
        "<b>One-Hot Encoding:</b> <code>OneHotEncoder(handle_unknown='ignore', sparse_output=False)</code> converts categorical strings to dense binary vectors.",
        "<b>Standard Scaling:</b> <code>StandardScaler()</code> normalizes continuous lab values to zero mean and unit variance ($Z = \\frac{X - \\mu}{\\sigma}$).",
        "<b>Artifact Serialization:</b> Imputers, encoders, and scalers are saved using <code>joblib.dump()</code> to <code>data/stage1_ml/models/</code>."
    ]
    for pt in preproc_points:
        story.append(Paragraph(f"• {pt}", bullet_style))
        
    story.append(Spacer(1, 12))

    # Section 6: API & Serving Integration
    story.append(Paragraph("6. API & Real-Time Production Integration", h1_style))
    story.append(Paragraph(
        "The Data Engineer designed the real-time inference input transformer embedded inside <code>OncologyPredictionPipeline</code> ([prediction.py](file:///c:/Users/ADMIN/Desktop/oncology/personalized_precision_oncology/stage1_ml/prediction/prediction.py)):",
        body_style
    ))
    
    story.append(Paragraph(
        "1. <b>Real-Time API Payload Processing:</b> Receives raw single-patient JSON via FastAPI endpoint <code>POST /predict</code>, runs identical imputation & feature engineering transformations in-memory under <b>5 ms latency</b>.<br/>"
        "2. <b>Batch CSV Pipeline:</b> Processes multi-patient CSV files for batch clinical decision support in Streamlit dashboard (`integration/dashboard/app.py`).",
        body_style
    ))
    story.append(Spacer(1, 12))

    # Section 7: Data Quality & QA Assertions
    story.append(Paragraph("7. Data Quality Assurance & Unit Tests", h1_style))
    story.append(Paragraph(
        "To prevent pipeline breakage in production, automated test suites verify data contracts ([test_stage1.py](file:///c:/Users/ADMIN/Desktop/oncology/personalized_precision_oncology/stage1_ml/tests/test_stage1.py)):",
        body_style
    ))
    
    qa_items = [
        "<b>Schema Contract Assertion:</b> Validates all 66 target feature columns exist in processed DataFrames.",
        "<b>Zero Null Assertion:</b> Asserts zero NaN or infinite values after pipeline execution.",
        "<b>Type Checking:</b> Verifies floating-point precision alignment for PyTorch/XGBoost inputs.",
        "<b>Boundary Validation:</b> Ensures probabilities remain bounded in $[0.0, 1.0]$."
    ]
    for qa in qa_items:
        story.append(Paragraph(f"• {qa}", bullet_style))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {filename}")

if __name__ == "__main__":
    out_pdf = "c:/Users/ADMIN/Desktop/oncology/Stage1_Data_Engineering_Detailed_Report.pdf"
    build_pdf(out_pdf)
