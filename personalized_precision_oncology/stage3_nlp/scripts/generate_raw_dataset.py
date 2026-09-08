import os
import csv
import random
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_PATH = os.path.join(OUT_DIR, "clinical_text_raw.csv")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
TOTAL_ROWS = 10000

NOTE_TYPE_COUNTS = {
    "physician_progress_note": 3000,
    "patient_symptom_log": 2000,
    "nurse_intake": 2000,
    "pathology_report": 2000,
    "treatment_note": 1000,
}

NOTE_TYPE_TO_SOURCE = {
    "physician_progress_note": "physician_note",
    "patient_symptom_log": "patient_report",
    "nurse_intake": "nursing_note",
    "pathology_report": "pathology_report",
    "treatment_note": "treatment_record",
}

NOTE_TYPE_TO_DEPARTMENTS = {
    "physician_progress_note": ["medical_oncology", "surgical_oncology", "radiation_oncology", "outpatient_oncology"],
    "patient_symptom_log": ["outpatient_oncology", "medical_oncology"],
    "nurse_intake": ["oncology_nursing", "outpatient_oncology"],
    "pathology_report": ["pathology"],
    "treatment_note": ["medical_oncology", "radiation_oncology", "outpatient_oncology"],
}

LANGUAGE = "en"

N_PATIENTS = 3500

# --------------------------------------------------------------------------
# CLINICAL VOCABULARY POOLS
# --------------------------------------------------------------------------
CANCER_TYPES = [
    "breast cancer", "lung cancer", "colorectal cancer", "ovarian cancer",
    "prostate cancer", "pancreatic cancer", "gastric cancer", "melanoma",
    "leukemia", "lymphoma",
]

TREATMENT_TYPES = [
    "chemotherapy", "immunotherapy", "targeted therapy",
    "radiation therapy", "surgery", "hormonal therapy",
]

DRUGS = [
    "cisplatin", "carboplatin", "paclitaxel", "docetaxel", "doxorubicin",
    "cyclophosphamide", "pembrolizumab", "nivolumab", "trastuzumab",
    "osimertinib", "erlotinib", "gefitinib",
]

GENES = [
    "EGFR", "EGFR exon 19 deletion", "EGFR L858R", "KRAS G12C", "BRAF V600E",
    "ALK rearrangement", "ROS1 fusion", "HER2 amplification",
    "BRCA1 mutation", "BRCA2 mutation", "TP53 mutation", "PIK3CA mutation",
]

ADVERSE_EVENTS = [
    "nausea", "vomiting", "diarrhea", "fatigue", "fever", "rash",
    "neutropenia", "anemia", "thrombocytopenia", "neuropathy", "mucositis",
    "dyspnea", "pneumonitis", "febrile neutropenia", "abdominal pain",
]

DOSAGES = [
    "50 mg", "100 mg", "200 mg", "5 mg daily", "10 mg/kg",
    "50 mg/m2", "500 mg twice daily",
]

SEVERITIES = ["mild", "moderate", "severe", "grade 1", "grade 2", "grade 3", "intermittent", "persistent"]
CYCLES = ["cycle one", "cycle two", "cycle three", "cycle four", "cycle five", "cycle six"]
STAGES = ["stage I", "stage II", "stage IIIA", "stage IIIB", "stage IV", "early-stage", "locally advanced", "metastatic"]
BODY_SITES = ["liver", "lung", "bone", "lymph nodes", "brain", "peritoneum", "adrenal gland"]
VITALS = [
    "BP 118/76, HR 82, Temp 98.6F, RR 16",
    "BP 132/84, HR 96, Temp 100.2F, RR 20",
    "BP 108/70, HR 74, Temp 98.1F, RR 14",
    "BP 140/90, HR 102, Temp 101.4F, RR 22",
    "BP 122/78, HR 88, Temp 99.0F, RR 18",
]
ECOG = ["ECOG 0", "ECOG 1", "ECOG 2", "ECOG 3"]

# --------------------------------------------------------------------------
# PHRASE BUILDERS
# --------------------------------------------------------------------------

def p_negation():
    templates = [
        "Patient denies {ae}.",
        "No evidence of {gene}.",
        "Patient reports no {ae}.",
        "Denies {ae} or {ae2} since last visit.",
        "No signs of {ae} on exam.",
    ]
    t = random.choice(templates)
    return t.format(ae=random.choice(ADVERSE_EVENTS), ae2=random.choice(ADVERSE_EVENTS), gene=random.choice(GENES))


def p_history():
    templates = [
        "History of {sev} {ae} during the previous chemotherapy cycle.",
        "Prior {gene} was documented on earlier molecular testing.",
        "Patient previously experienced {ae} following {drug}.",
        "Past medical history notable for {ae} and {ae2}.",
        "Previously treated with {drug} with {sev} {ae} reported at that time.",
    ]
    t = random.choice(templates)
    return t.format(
        sev=random.choice(SEVERITIES), ae=random.choice(ADVERSE_EVENTS),
        ae2=random.choice(ADVERSE_EVENTS), gene=random.choice(GENES),
        drug=random.choice(DRUGS),
    )


def p_current_symptom():
    templates = [
        "Patient currently reports worsening {ae}.",
        "Patient reports {sev} {ae} today.",
        "Today, patient describes new-onset {ae}.",
        "Currently experiencing {sev} {ae} and {ae2}.",
        "On today's visit, {ae} is noted to be {sev}.",
    ]
    t = random.choice(templates)
    return t.format(sev=random.choice(SEVERITIES), ae=random.choice(ADVERSE_EVENTS), ae2=random.choice(ADVERSE_EVENTS))


def p_multi_entity():
    templates = [
        "Patient developed {ae} after receiving {drug} {dose} for {gene}-mutated {cancer}.",
        "Patient's {cancer} was treated with {drug} {dose}; subsequently developed {ae}.",
        "Following {tx} with {drug}, patient reported {sev} {ae}.",
        "Molecular profile positive for {gene} in the setting of {cancer}; started on {drug} {dose}.",
    ]
    t = random.choice(templates)
    return t.format(
        ae=random.choice(ADVERSE_EVENTS), drug=random.choice(DRUGS), dose=random.choice(DOSAGES),
        gene=random.choice(GENES), cancer=random.choice(CANCER_TYPES), tx=random.choice(TREATMENT_TYPES),
        sev=random.choice(SEVERITIES),
    )


def p_treatment_plan():
    templates = [
        "Plan: continue {drug} {dose} on schedule, reassess in {cycle}.",
        "Plan is to proceed with {tx} given {gene} status.",
        "Will initiate {drug} {dose} pending final labs.",
        "Recommend dose adjustment of {drug} due to {sev} {ae}.",
        "Discussed risks and benefits of {tx}; patient agrees to proceed.",
    ]
    t = random.choice(templates)
    return t.format(
        drug=random.choice(DRUGS), dose=random.choice(DOSAGES), cycle=random.choice(CYCLES),
        tx=random.choice(TREATMENT_TYPES), gene=random.choice(GENES),
        sev=random.choice(SEVERITIES), ae=random.choice(ADVERSE_EVENTS),
    )


def p_staging():
    return "Staging consistent with {stage} {cancer}, with possible involvement of the {site}.".format(
        stage=random.choice(STAGES), cancer=random.choice(CANCER_TYPES), site=random.choice(BODY_SITES)
    )


def p_pathology_finding():
    templates = [
        "Molecular testing identified {gene}.",
        "Immunohistochemistry consistent with {cancer}.",
        "Biopsy of the {site} reveals findings consistent with {cancer}.",
        "Next-generation sequencing panel positive for {gene}.",
        "Histologic evaluation confirms {cancer}, {stage}.",
    ]
    t = random.choice(templates)
    return t.format(
        gene=random.choice(GENES), cancer=random.choice(CANCER_TYPES),
        site=random.choice(BODY_SITES), stage=random.choice(STAGES),
    )

def gen_physician_note():
    cancer = random.choice(CANCER_TYPES)
    stage = random.choice(STAGES)
    cycle = random.choice(CYCLES)
    opener_choices = [
        "Patient presents for follow-up after {cycle} of {tx} for {stage} {cancer}.",
        "Seen today in clinic for routine oncology follow-up regarding {stage} {cancer}.",
        "{ecog}. Patient returns for reassessment following recent {tx}.",
        "Follow-up visit for management of {stage} {cancer}, currently on {tx}.",
    ]
    opener = random.choice(opener_choices).format(
        cycle=cycle, tx=random.choice(TREATMENT_TYPES), stage=stage, cancer=cancer, ecog=random.choice(ECOG)
    )

    body_pool = [p_current_symptom, p_negation, p_history, p_multi_entity, p_treatment_plan, p_staging]
    n_sentences = random.randint(2, 5)
    body_funcs = random.sample(body_pool, k=min(n_sentences, len(body_pool)))
    body = " ".join(f() for f in body_funcs)

    vitals = random.choice(VITALS)
    closer_choices = [
        f"Vitals: {vitals}.",
        f"Exam otherwise unremarkable. Vitals: {vitals}.",
        "Will continue to monitor closely.",
        "Discussed plan of care with patient; questions answered.",
    ]
    closer = random.choice(closer_choices)

    parts = [opener, body, closer]
    random.shuffle_seed = None
    return " ".join(parts)


def gen_patient_note():
    ae = random.choice(ADVERSE_EVENTS)
    ae2 = random.choice(ADVERSE_EVENTS)
    drug = random.choice(DRUGS)
    openers = [
        "I have been feeling very tired since the last treatment and had {ae} yesterday.",
        "Since my last dose of {drug}, I've noticed some {ae}.",
        "I wanted to let my doctor know I've had {ae} for the past couple of days.",
        "My {ae} has gotten a little better but I still feel {sev} {ae2}.",
        "Just checking in - after this round of treatment I've had {sev} {ae}.",
    ]
    opener = random.choice(openers).format(ae=ae, ae2=ae2, drug=drug, sev=random.choice(SEVERITIES))

    extra_pool = [p_negation, p_current_symptom, p_history]
    n = random.randint(1, 3)
    extras = [f() for f in random.sample(extra_pool, k=min(n, len(extra_pool)))]

    closers = [
        "Not sure if this is normal, wanted to check.",
        "It's not too bad, just wanted to update the team.",
        "Please let me know if I need to come in.",
        "",
        "I'm managing okay overall but wanted to report this.",
    ]
    parts = [opener] + extras + [random.choice(closers)]
    return " ".join(p for p in parts if p)


def gen_nurse_note():
    drug = random.choice(DRUGS)
    dose = random.choice(DOSAGES)
    openers = [
        "Pt arrived for scheduled infusion of {drug} {dose}.",
        "Pt checked in for {tx} appointment, tolerating well.",
        "Intake assessment completed prior to {tx}.",
        "Pt arrived on time, IV access obtained without difficulty.",
    ]
    opener = random.choice(openers).format(drug=drug, dose=dose, tx=random.choice(TREATMENT_TYPES))

    symptom_pool = [p_negation, p_current_symptom, p_history]
    n = random.randint(1, 3)
    symptom_lines = [f() for f in random.sample(symptom_pool, k=min(n, len(symptom_pool)))]
    # nursing shorthand style adjustments
    symptom_lines = [s.replace("Patient", "Pt") for s in symptom_lines]

    vitals = random.choice(VITALS)
    closers = [
        f"VS stable, {vitals}.",
        f"Vitals within normal limits: {vitals}.",
        "Infusion started without complication.",
        "Pt tolerating infusion well at this time.",
        "Will continue to monitor per protocol.",
    ]
    parts = [opener] + symptom_lines + [random.choice(closers)]
    return " ".join(parts)


def gen_pathology_note():
    cancer = random.choice(CANCER_TYPES)
    site = random.choice(BODY_SITES)
    stage = random.choice(STAGES)
    openers = [
        "Specimen: core needle biopsy of the {site}.",
        "Specimen: surgical resection, {site}.",
        "Gross description: tan-white tissue fragments received from the {site}.",
        "Final diagnosis pending correlation with clinical findings, {site} biopsy.",
    ]
    opener = random.choice(openers).format(site=site)

    finding_pool = [p_pathology_finding, p_negation, p_staging]
    n = random.randint(2, 4)
    findings = [f() for f in random.sample(finding_pool, k=min(n, len(finding_pool)))]

    closers = [
        "Findings were discussed with the treating oncology team.",
        "Recommend correlation with clinical and radiologic findings.",
        "Additional molecular testing pending.",
        f"Overall consistent with {cancer}, {stage}.",
    ]
    parts = [opener] + findings + [random.choice(closers)]
    return " ".join(parts)


def gen_treatment_note():
    drug = random.choice(DRUGS)
    dose = random.choice(DOSAGES)
    cycle = random.choice(CYCLES)
    openers = [
        "Administered {drug} {dose} today, {cycle} of planned regimen.",
        "Treatment note: patient received {drug} {dose} without immediate infusion reaction.",
        "Completed {cycle} of {tx} today with {drug} {dose}.",
        "Chemotherapy administration record: {drug} {dose} infused over scheduled duration.",
    ]
    opener = random.choice(openers).format(drug=drug, dose=dose, cycle=cycle, tx=random.choice(TREATMENT_TYPES))

    ae_pool = [p_current_symptom, p_negation, p_multi_entity, p_treatment_plan]
    n = random.randint(1, 3)
    ae_lines = [f() for f in random.sample(ae_pool, k=min(n, len(ae_pool)))]

    closers = [
        "Next cycle scheduled in three weeks pending labs.",
        "Patient tolerated treatment overall.",
        "Will reassess response with imaging after next cycle.",
        "No immediate adverse reactions observed during infusion.",
    ]
    parts = [opener] + ae_lines + [random.choice(closers)]
    return " ".join(parts)


def fix_capitalization(text):
    sentences = text.split(". ")
    fixed = []
    for s in sentences:
        s = s.strip()
        if s:
            s = s[0].upper() + s[1:]
        fixed.append(s)
    return ". ".join(fixed)


NOTE_TYPE_GENERATORS = {
    "physician_progress_note": gen_physician_note,
    "patient_symptom_log": gen_patient_note,
    "nurse_intake": gen_nurse_note,
    "pathology_report": gen_pathology_note,
    "treatment_note": gen_treatment_note,
}

START_DATE = datetime(2025, 1, 1, 7, 0, 0)
END_DATE = datetime(2026, 8, 31, 19, 0, 0)
DATE_RANGE_SECONDS = int((END_DATE - START_DATE).total_seconds())

def random_timestamp():
    offset = random.randint(0, DATE_RANGE_SECONDS)
    ts = START_DATE + timedelta(seconds=offset)
    return ts.strftime("%Y-%m-%d %H:%M:%S")

def build_note_type_sequence():
    seq = []
    for note_type, count in NOTE_TYPE_COUNTS.items():
        seq.extend([note_type] * count)
    random.shuffle(seq)
    return seq

def build_patient_ids(n_rows, n_patients):
    patient_ids = [f"P_{i:05d}" for i in range(1, n_patients + 1)]
    assigned = [random.choice(patient_ids) for _ in range(n_rows)]
    return assigned

def generate_dataset():
    note_type_seq = build_note_type_sequence()
    patient_seq = build_patient_ids(TOTAL_ROWS, N_PATIENTS)

    rows = []
    for i in range(TOTAL_ROWS):
        note_type = note_type_seq[i]
        record_id = f"CLN_{i+1:06d}"
        patient_id = patient_seq[i]
        source = NOTE_TYPE_TO_SOURCE[note_type]
        department = random.choice(NOTE_TYPE_TO_DEPARTMENTS[note_type])
        timestamp = random_timestamp()
        text = fix_capitalization(NOTE_TYPE_GENERATORS[note_type]())

        rows.append({
            "record_id": record_id,
            "patient_id": patient_id,
            "note_type": note_type,
            "timestamp": timestamp,
            "department": department,
            "text": text,
            "source": source,
            "language": LANGUAGE,
        })

    # Add a few duplicate rows and messy data to simulate real conditions
    # Duplicate some rows
    rows.extend(random.sample(rows, 15))

    # Add a few empty texts
    for _ in range(10):
        rows[random.randint(0, TOTAL_ROWS-1)]['text'] = "   "

    # Add some messy whitespace
    for _ in range(50):
        idx = random.randint(0, TOTAL_ROWS-1)
        if rows[idx]['text'].strip():
            rows[idx]['text'] = " \t  \n " + rows[idx]['text'] + "  \n\n  "

    return rows

def validate(rows):
    return []

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = generate_dataset()
    fieldnames = ["record_id", "patient_id", "note_type", "timestamp", "department", "text", "source", "language"]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
