# Stage 3 NLP: Comprehensive Data Cleaning & Dataset Architecture Report

**Project:** Personalized Precision Oncology Platform  
**Stage:** Stage 03 — Clinical NLP & Text Processing  
**Scope:** Comparative Analysis between Raw (`clinical_text_raw.csv`) and Cleaned (`clinical_text_cleaned.csv`) Datasets, Architectural Classification, and Clinical Rationale

---

## Executive Summary

This report documents the end-to-end data cleaning, sanitization, and normalization pipeline applied to the Stage 3 clinical notes dataset. The objective was to eliminate text encoding artifacts, irregular multi-line breaks, and whitespace anomalies while strictly preserving clinical semantics, medical entity cases, numerical dosages, negation tokens, and 100% record alignment across multimodal stages.

---

## 1. Dataset Nature: Structured, Unstructured, or Semi-Structured?

This dataset is architecturally classified as **Semi-Structured** — specifically, **unstructured free-text clinical narratives encapsulated within structured relational metadata**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        SEMI-STRUCTURED ONCOLOGY DATASET CONTAINER                      │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│         STRUCTURED RELATIONAL EHR DATA    │       UNSTRUCTURED CLINICAL TEXT PAYLOAD   │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│  • record_id    : CLN_009484              │  • text:                                   │
│  • patient_id   : P_01459                 │    "Follow-up visit for management of      │
│  • note_type    : physician_progress_note │     metastatic pancreatic cancer, currently│
│  • timestamp    : 2025-06-26 16:48:56     │     on hormonal therapy. Molecular profile │
│  • department   : medical_oncology        │     positive for BRCA2 mutation... started │
│  • source       : physician_note          │     on osimertinib 50 mg. Staging          │
│  • language     : en                      │     consistent with stage I lung cancer.   │
│                                           │     Patient denies nausea..."              │
│  [Indexed, Queryable, Strict Enum Types]  │  [Free-text Narrative, Bio-Entities, Neg]  │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

* **The Structured Dimension:** The columns `record_id`, `patient_id`, `note_type`, `timestamp`, `department`, `source`, and `language` adhere to fixed relational schemas, strict categorical domains, and ISO timestamps.
* **The Unstructured Dimension:** The `text` column contains high-density, free-form medical language with doctor shorthand, vital signs, genomic mutation profiles, drug regimens, and symptom descriptions.

---

## 2. Why We Chose This Dataset (The 5 Core Pillars)

In oncology, **over 80% of actionable clinical intelligence is trapped in unstructured doctor notes and pathology reports**, rather than in tabular EHR spreadsheets. This dataset was selected based on five critical pillars:

```
                      WHY THIS DATASET WAS CHOSEN
                                   │
     ┌───────────────────┬─────────┴─────────┬───────────────────┐
     ▼                   ▼                   ▼                   ▼
1. Deep Clinical    2. 5-Class Named    3. Contextual Negation   4. Multimodal Fusion
   Nuances & Causal    Entity Extraction   Detection & Patient      with Stages 1 & 2
   Reasoning           (Biomarkers/Drugs)  Safety Guardrail         via Patient ID
```

### Pillar 1: Capturing Nuances Missing from Structured Tables
Structured lab values only provide isolated numbers (e.g., *"Platelets: 45,000/µL"*). The unstructured text provides the critical clinical context:
> *"Since my last dose of osimertinib, patient developed grade 3 thrombocytopenia, but denies abdominal pain. Plan: withhold therapy for 7 days."*
It captures **treatment causality**, **adverse event severity**, **physician rationale**, and **reassessment timelines**.

### Pillar 2: Training High-Precision Biomedical NER Models
The text enables fine-tuned Transformer models (e.g., BioBERT, PubMedBERT, ClinicalBERT) to identify 5 essential oncology entity classes:
1. **Cancer Types & Histology:** *Metastatic melanoma, Stage IIIB breast cancer, Colorectal adenocarcinoma*
2. **Genetic Biomarkers & Mutations:** *KRAS G12C, EGFR L858R, BRCA2, ROS1 fusion, TP53, HER2*
3. **Chemotherapy & Targeted Agents:** *Osimertinib 50 mg, Pembrolizumab 100 mg, Paclitaxel, Nivolumab*
4. **Adverse Events & Toxicities:** *Febrile neutropenia, Pneumonitis, Mucositis, Thrombocytopenia*
5. **Anatomical Sites & Metastases:** *Liver, Bone, Adrenal gland, Brain, Peritoneum*

### Pillar 3: Contextual Negation Parsing for Patient Safety
In clinical care, extracting a word in isolation can lead to fatal misinterpretations:
* *"Patient **has** severe dyspnea"* $\longrightarrow$ **Immediate emergency / drug toxicity escalation**.
* *"Patient **denies** dyspnea"* $\longrightarrow$ **Normal tolerance / routine continuation**.
This dataset provides realistic clinical negation phrases (`"denies"`, `"no signs of"`, `"no evidence of"`, `"unremarkable"`) to train robust negation-aware classifiers.

### Pillar 4: Multi-Source Clinical Perspectives
The dataset integrates **5 distinct medical viewpoints**:
* **Physician Progress Notes:** Holistic assessments, ECOG performance statuses, and long-term treatment strategies.
* **Pathology Reports:** Histologic tissue confirmations, gross descriptions, and NGS mutation panels.
* **Nurse Intake Notes:** Vital sign checks (`BP`, `HR`, `Temp`, `RR`), IV infusion tolerance, and immediate acute observations.
* **Treatment Records:** Administered cycle counts, exact dosages, and acute infusion reactions.
* **Patient Symptom Logs:** Real-world patient-reported outcomes (PROs) and outpatient toxicities.

### Pillar 5: Seamless Multimodal Integration Across Pipeline Stages
In our unified Precision Oncology Platform, patient records are linked across modalities via `patient_id`:
* **Stage 1 (Machine Learning):** Predicts baseline risk from structured tabular lab & demographic data.
* **Stage 2 (Deep Learning):** Models survival curves and complex non-linear clinical interactions.
* **Stage 3 (Clinical NLP):** Extracts extracted biomarker entities and computes acute urgency triage scores.
* **Final Agentic Stage:** Synthesizes all three stages to generate personalized, guideline-adherent treatment plans.

---

## 3. Dataset Dimensions & Line Count Explanation

A common source of confusion in clinical text processing is the difference between **physical lines in a text editor** and **actual parsed records (rows)**.

| Metric | Raw (Uncleaned) Dataset | Cleaned Dataset | Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Total Parsed Records (Rows)** | **10,015** | **10,015** | **0 rows dropped** (100% data retention) |
| **Header Row** | 1 | 1 | Exact same 8 schema columns |
| **Physical Lines in Editor** | **10,163 lines** (+1 empty EOF) | **10,016 lines** | Cleaned file has exactly 1 line per record |
| **Multi-line Text Records** | **49 records** | **0 records** | All multi-line records collapsed to single line |
| **Embedded Newlines (`\r\n` / `\n`)** | **147 extra linebreaks** | **0** | Eliminated all erratic line breaks |
| **Blank / Whitespace Records** | 10 (contained `'   '`) | 10 (standardized to `""`) | Clean empty string representations |
| **Total Records Modified** | — | **59 records** | 49 formatted + 10 whitespace standardized |
| **Min Text Length (non-empty)** | 3 characters (`'   '`) | 80 characters | Non-empty records have valid clinical content |
| **Max Text Length** | 591 characters | 591 characters | No truncation of patient records |
| **Average Text Length** | 274.56 characters | 274.75 characters | Reflects removal of whitespace artifacts |

### Mathematical Line Count Proof:
$$\text{Raw File Lines} = 1\text{ (Header)} + 10,015\text{ (Records)} + 147\text{ (Embedded Newlines inside quotes)} = \mathbf{10,163}\text{ lines}$$
$$\text{Cleaned File Lines} = 1\text{ (Header)} + 10,015\text{ (Records)} + 0\text{ (Embedded Newlines)} = \mathbf{10,016}\text{ lines}$$

---

## 4. Dataset Schema

Both files maintain an identical 8-column structure:

1. `record_id` (string, unique ID, e.g., `CLN_000001`)
2. `patient_id` (string, e.g., `P_01234`)
3. `note_type` (categorical: `physician_progress_note`, `nurse_intake`, `pathology_report`, `patient_symptom_log`, `treatment_note`)
4. `timestamp` (datetime ISO string: `YYYY-MM-DD HH:MM:SS`)
5. `department` (categorical: `medical_oncology`, `radiation_oncology`, `pathology`, `oncology_nursing`, `outpatient_oncology`)
6. `text` (free-text clinical narrative)
7. `source` (categorical: `physician_note`, `nursing_note`, `pathology_report`, `patient_report`, `treatment_record`)
8. `language` (string: `en`)

---

## 5. Step-by-Step Data Cleaning Pipeline

The cleaning operations are executed deterministically through `clean_text.py`:

```
                                  STAGE 3 CLEANING WORKFLOW
                                  
  ┌─────────────────────────┐
  │  clinical_text_raw.csv  │ ──► Compute Initial MD5 Checksum (Preserve Raw File)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 1. Empty/Null Handling  │ ──► Converts NaN, None, and pure whitespace ('   ') to ""
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 2. Unicode NFKC Norm    │ ──► Standardizes smart quotes (“ ” ‘ ’), ligatures & dashes
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 3. Regex Whitespace Fix │ ──► Collapses \r\n, \t, and \s+ into single space ' '
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 4. Metadata Guard       │ ──► Replaces null categorical fields with 'UNKNOWN'
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 5. PII Validation       │ ──► Scans for SSN, phone, email patterns (0 found)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 6. Integrity & Export   │ ──► Verify Raw MD5 unchanged, Save clinical_text_cleaned.csv
  └─────────────────────────┘
```

### Detailed Process Notes:

1. **Raw File Immutability (MD5 Checksum):**
   - The raw file hash is verified before and after the script runs to prevent in-place corruption or accidental truncation.
2. **Safe Empty & NaN Handling:**
   - Any missing or pure-whitespace strings are safely converted to an empty string `""` without dropping the row.
3. **Unicode Normalization (`NFKC`):**
   - Resolves formatting discrepancies from EHR copy-pastes, such as curly quotes, en-dashes, and special symbols.
4. **Whitespace & Multi-line Collapse:**
   - Leading and trailing whitespace is stripped via `.strip()`.
   - Repeated whitespace, tabs, and newline sequences (`\s+`) are replaced with a single standard ASCII space.
5. **Categorical Metadata Imputation:**
   - Missing entries in metadata columns are filled with `"UNKNOWN"` to avoid pipeline crashes in downstream classifiers.
6. **Personally Identifiable Information (PII) Screening:**
   - Scanned text using regex for phone numbers, SSNs, and emails. **0 violations found**.
7. **Cross-Stage ID Preservation:**
   - Kept all 10,015 record IDs and patient IDs intact to guarantee seamless joins with Stage 1 (Tabular ML) and Stage 2 (Deep Learning).

---

## 6. Clinical Safety Protocols (What Was Intentionally Preserved)

Clinical NLP models require precise semantic and syntactic structure. The following standard NLP transformations were **strictly prohibited**:

* ❌ **NO Lowercasing:** Uppercase text is essential for recognizing gene mutations (*e.g., `KRAS G12C`, `EGFR L858R`, `TP53`*), medical acronyms (*e.g., `ECOG`, `IV`, `CT`, `MRI`*), and cancer stage numerals (*e.g., `Stage IIIB`*).
* ❌ **NO Punctuation Removal:** Dosages (*e.g., `50 mg/m2`, `10 mg/kg`*), vitals (*`BP 120/80`*), and tumor measurements (*`3.5 cm`*) rely on punctuation.
* ❌ **NO Stopword Removal:** Negation terms (*"no"*, *"not"*, *"denies"*) are vital. Stripping stopwords would invert clinical meaning (*e.g., "denies fever" $\rightarrow$ "fever"*).
* ❌ **NO Stemming or Lemmatization:** Preserves exact medical terminology and inflectional endings.

---

## 7. Concrete Before & After Examples

### Case 1: Multi-line Breaking & Control Characters (Record `CLN_009502`, Raw Lines 9641–9644)
* **Raw (Uncleaned):**
  ```csv
  CLN_009502,P_02448,pathology_report,2025-07-11 22:32:34,pathology," 	  
   Gross description: tan-white tissue fragments received from the lung. Histologic evaluation confirms lung cancer, locally advanced. Staging consistent with metastatic melanoma, with possible involvement of the adrenal gland. No signs of neutropenia on exam. Additional molecular testing pending.  

    ",pathology_report,en
  ```
* **Cleaned:**
  ```csv
  CLN_009502,P_02448,pathology_report,2025-07-11 22:32:34,pathology,Gross description: tan-white tissue fragments received from the lung. Histologic evaluation confirms lung cancer, locally advanced. Staging consistent with metastatic melanoma, with possible involvement of the adrenal gland. No signs of neutropenia on exam. Additional molecular testing pending.,pathology_report,en
  ```

### Case 2: Embedded Carriage Returns & Tabs (Record `CLN_007891`)
* **Raw (Uncleaned):**
  ```text
  " \t  \r\n Since my last dose of osimertinib, I've noticed some thrombocytopenia. Past medical history notable for abdominal pain and rash.  \r\n\r\n  "
  ```
* **Cleaned:**
  ```text
  "Since my last dose of osimertinib, I've noticed some thrombocytopenia. Past medical history notable for abdominal pain and rash."
  ```

### Case 3: Whitespace-Only Record Standardization
* **Raw (Uncleaned):** `'   '` (3 whitespace characters)
* **Cleaned:** `""` (clean empty string)

---

## 8. Verification and Readiness

- **Downstream Readiness:** The cleaned dataset is 100% synchronized and validated for Named Entity Recognition (NER), BioBERT embeddings, and urgency classification models.
- **Audit Trail:** Preserved raw files and automated verification scripts guarantee full reproducibility under clinical AI governance standards.
