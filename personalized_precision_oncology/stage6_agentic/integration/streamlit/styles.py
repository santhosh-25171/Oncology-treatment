"""
Custom CSS styles for Stage 6: Oncology Command Center Workstation.
Provides dark navy header, dark navy sidebar, crisp medical cards,
status badges, circular progress rings, and responsive layout matching the reference image.
"""

def get_command_center_css() -> str:
    return """
<style>
/* ----------------------------------------------------
   GLOBAL RESETS & TYPOGRAPHY
---------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Remove default Streamlit top padding */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* High-Contrast Main Workspace Typography */
.main, .main p, .main span {
    color: #1e293b;
}

.main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
    color: #0f172a !important;
    font-weight: 700 !important;
}

.main b, .main strong {
    color: #0f172a !important;
}

.main [data-testid="stCaptionContainer"] p {
    color: #475569 !important;
    font-size: 13px !important;
}

/* ----------------------------------------------------
   SIDEBAR STYLING (Ultra High-Contrast Dark Navy #0B192C)
---------------------------------------------------- */
section[data-testid="stSidebar"],
[data-testid="stSidebar"] {
    background-color: #0b192c !important;
    color: #ffffff !important;
    border-right: 1px solid #1e293b !important;
}

[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
}

/* Force all general text in sidebar to bright high-contrast white */
section[data-testid="stSidebar"] *,
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] a {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Radio navigation group in sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
    margin-top: 4px !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div,
[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}

/* Each navigation item card */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    margin-bottom: 3px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover,
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(37, 99, 235, 0.25) !important;
    border-color: #3b82f6 !important;
}

/* Active / Selected Radio item (Solid Royal Blue Pill matching reference image) */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: 1px solid #60a5fa !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.45) !important;
}

/* Text inside all radio labels (CRITICAL: High contrast white font) */
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] label span,
[data-testid="stSidebar"] [data-testid="stRadio"] label div,
[data-testid="stSidebar"] div[role="radiogroup"] label p,
[data-testid="stSidebar"] div[role="radiogroup"] label span,
[data-testid="stSidebar"] div[role="radiogroup"] label div,
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    margin: 0 !important;
    line-height: 1.4 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
}

/* Sidebar Radio Dot Styling */
[data-testid="stSidebar"] div[data-testid="stRadioCircle"] {
    border-color: #94a3b8 !important;
    background-color: transparent !important;
}

[data-testid="stSidebar"] label:has(input:checked) div[data-testid="stRadioCircle"],
[data-testid="stSidebar"] label[data-checked="true"] div[data-testid="stRadioCircle"] {
    border-color: #ffffff !important;
    background-color: #ffffff !important;
}

[data-testid="stSidebar"] label:has(input:checked) div[data-testid="stRadioCircle"] > div,
[data-testid="stSidebar"] label[data-checked="true"] div[data-testid="stRadioCircle"] > div {
    background-color: #2563eb !important;
}

/* Sidebar Primary Button (+ Add New Patient) - Coral/Red Gradient matching reference image */
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 18px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4) !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
}

[data-testid="stSidebar"] button[kind="primary"]:hover,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover,
[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(239, 68, 68, 0.55) !important;
}

[data-testid="stSidebar"] button[kind="primary"] *,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Sidebar Secondary Buttons (Run Deliberation Panel, etc.) */
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background-color: rgba(37, 99, 235, 0.15) !important;
    color: #38bdf8 !important;
    -webkit-text-fill-color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    width: 100% !important;
}

[data-testid="stSidebar"] button[kind="secondary"]:hover,
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover,
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background-color: #2563eb !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: #60a5fa !important;
}

/* Sidebar Selectbox & Inputs */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #1e293b !important;
    border-color: #334155 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] * {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
}

/* ----------------------------------------------------
   MAIN WORKSPACE INPUTS & SELECTBOXES (Crisp White & High-Contrast)
---------------------------------------------------- */
/* Search bar, text inputs, text areas on main workspace */
.main [data-testid="stTextInput"] div[data-baseweb="input"],
.main [data-testid="stTextInput"] input,
.main [data-testid="stTextArea"] div[data-baseweb="textarea"],
.main [data-testid="stTextArea"] textarea,
.main [data-testid="stNumberInput"] div[data-baseweb="input"],
.main [data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

.main [data-testid="stTextInput"] input::placeholder,
.main [data-testid="stTextArea"] textarea::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
    font-weight: 400 !important;
}

/* Selectbox on main workspace (Switch Case, Filters, Form fields) */
.main [data-testid="stSelectbox"] div[data-baseweb="select"],
.main [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

.main [data-testid="stSelectbox"] * {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 600 !important;
}

/* Dropdown popover list items */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="menu"],
div[data-baseweb="menu"] * {
    background-color: #ffffff !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}

/* Main workspace labels */
.main [data-testid="stWidgetLabel"] p,
.main [data-testid="stWidgetLabel"] span {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ----------------------------------------------------
   TOP HEADER BAR
---------------------------------------------------- */
.occ-header-bar {
    background: linear-gradient(90deg, #0a192f 0%, #0f2b48 100%);
    color: #ffffff;
    padding: 14px 24px;
    border-radius: 10px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
}

.occ-brand-title {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.occ-brand-sub {
    font-size: 12px;
    color: #38bdf8;
    font-weight: 500;
    letter-spacing: 0.2px;
}

.occ-header-meta {
    display: flex;
    align-items: center;
    gap: 20px;
}

.occ-system-online {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: #34d399;
    font-weight: 600;
}

.occ-pulse-dot {
    width: 9px;
    height: 9px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-green 2s infinite;
}

@keyframes pulse-green {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.occ-doc-profile {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255, 255, 255, 0.08);
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

/* ----------------------------------------------------
   PATIENT HEADER CARD
---------------------------------------------------- */
.patient-header-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.patient-info-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.patient-avatar-circle {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: #e0f2fe;
    color: #0284c7;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: 700;
    border: 2px solid #bae6fd;
}

.patient-title-line {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
}

.patient-meta-line {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
}

.badge-alert-red {
    background-color: #fee2e2;
    color: #dc2626;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid #fca5a5;
}

.badge-alert-green {
    background-color: #d1fae5;
    color: #059669;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid #6ee7b7;
}

/* ----------------------------------------------------
   TOP METRIC CARDS
---------------------------------------------------- */
.card-risk-severe {
    background: #fff5f5;
    border: 1px solid #fed7d7;
    border-radius: 12px;
    padding: 16px;
    height: 100%;
    box-shadow: 0 2px 5px rgba(229, 62, 62, 0.05);
}

.card-confidence {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    height: 100%;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
}

.card-rec-status {
    background: #fffbeb;
    border: 1px solid #fef3c7;
    border-radius: 12px;
    padding: 16px;
    height: 100%;
    box-shadow: 0 2px 5px rgba(217, 119, 6, 0.05);
}

.card-action-required {
    background: #ffffff;
    border: 1.5px solid #f87171;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.08);
}

/* ----------------------------------------------------
   SECOND ROW MINI-METRIC CARDS (with chevrons)
---------------------------------------------------- */
.metric-row-card {
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(0, 0, 0, 0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-row-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
}

.card-bg-red { background-color: #fff1f2; border-color: #fecdd3; }
.card-bg-blue { background-color: #f0f9ff; border-color: #bae6fd; }
.card-bg-yellow { background-color: #fefce8; border-color: #fef08a; }
.card-bg-purple { background-color: #faf5ff; border-color: #e9d5ff; }

/* ----------------------------------------------------
   PANELS & SECTIONS
---------------------------------------------------- */
.occ-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    height: 100%;
}

.occ-panel-title {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 8px;
}

/* ----------------------------------------------------
   KEY DATA AT A GLANCE ROWS
---------------------------------------------------- */
.key-data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #f8fafc;
    font-size: 13px;
}

.key-data-row:last-child {
    border-bottom: none;
}

.key-data-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #475569;
    font-weight: 500;
}

.key-data-val {
    font-weight: 700;
    color: #0f172a;
}

/* ----------------------------------------------------
   RECOMMENDED NEXT STEPS
---------------------------------------------------- */
.next-step-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #f1f5f9;
}

.step-circle {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background-color: #0284c7;
    color: #ffffff;
    font-weight: 700;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

/* ----------------------------------------------------
   LIVE WORKFLOW STATUS TIMELINE
---------------------------------------------------- */
.timeline-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 7px 0;
    font-size: 12px;
}

.timeline-dot-done {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background-color: #10b981;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    flex-shrink: 0;
    margin-top: 2px;
}

.timeline-dot-active {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background-color: #0284c7;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    flex-shrink: 0;
    margin-top: 2px;
    animation: pulse-blue 1.5s infinite;
}

@keyframes pulse-blue {
    0% { box-shadow: 0 0 0 0 rgba(2, 132, 199, 0.6); }
    70% { box-shadow: 0 0 0 5px rgba(2, 132, 199, 0); }
    100% { box-shadow: 0 0 0 0 rgba(2, 132, 199, 0); }
}

/* ----------------------------------------------------
   HUMAN IN THE LOOP CARD (SIDEBAR)
---------------------------------------------------- */
.hitl-card {
    background: rgba(15, 32, 66, 0.7);
    border: 1px solid #1e3a8a;
    border-radius: 8px;
    padding: 12px;
    margin-top: 20px;
}

.hitl-title {
    font-size: 13px;
    font-weight: 700;
    color: #60a5fa;
    display: flex;
    align-items: center;
    gap: 6px;
}

.hitl-text {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 4px;
    line-height: 1.4;
}

/* ----------------------------------------------------
   CIRCULAR PROGRESS RING
---------------------------------------------------- */
.confidence-ring-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
}

/* ----------------------------------------------------
   FOOTER
---------------------------------------------------- */
.occ-footer {
    text-align: center;
    padding: 18px 0 10px 0;
    color: #64748b;
    font-size: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}
</style>
"""
