# Stage 4 SLM — Raw Dataset Quality & Inconsistency Audit Report

> **DISCLAIMER**: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.  
> Evaluates the raw uncleaned synthetic dataset (`oncology_stage4_raw_10000.csv`) for Stage 4 SLM fine-tuning.

---

## 1. Dataset Dimensions & Completeness
- **Source File**: `oncology_stage4_raw_10000.csv`
- **Total Raw Records**: `10,000`
- **Total Columns**: `7`
- **Exact Duplicate Rows**: `115`
- **Missing Non-Critical Values**:
  - `source_type`: `214` missing
  - `stage1_context`: `201` missing
  - `stage2_context`: `215` missing
  - `stage3_context`: `201` missing
  - `patient_id`: `0 missing (100% complete)`
  - `clinical_report`: `0 missing (100% complete)`
  - `target_summary`: `0 missing (100% complete)`

---

## 2. Patient Cohort & Multi-Record Distribution
- **Unique Synthetic Patients**: `3,200`
- **Records per Patient Mean**: `3.12` (std: `1.54`)
- **Records per Patient Range**: `1` to `8` records (median: `3.0`)
- **Multi-Record Patients**: `Longitudinal consultation notes per patient require grouped patient-level splitting to prevent data leakage.`

---

## 3. Raw Clinical Inconsistencies Identified
The uncleaned dataset contains realistic clinical documentation irregularities:
1. **Clinical Shorthand & Abbreviations**:
   - Patient reference shorthand (`pt`): `4,932` mentions
   - Lab assessment shorthand (`WNL`): `291` mentions
   - Dosing intervals (`bid`, `qd`, `q3w`, `weekly`): `3,253` mentions
2. **Heterogeneous Date Formatting**:
   - Slash format (`MM/DD/YYYY`): `1,161`
   - Dash format (`YYYY-MM-DD`): `1,231`
   - Word format (`DD Mon YYYY`): `1,208`
3. **Inconsistent Context Serialization**:
   - Context fields are serialized across authors in multiple string formats (`(synthetic stage1) key: val`, `synthetic_stage1: key = val`, `[SYNTHETIC-STAGE1] key = val`, `SYNTHETIC STAGE OUTPUT - key:val`) rather than clean JSON.
4. **Source Type Capitalization & Spacing**:
   - `41` raw distinct representations (e.g. `treatment note` vs `Treatment Note`, `Imaging Report` vs `imaging_report`).

---

## 4. Text Length Characteristics

| Text Field | Min Chars | Max Chars | Mean Chars | Min Words | Max Words | Mean Words |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clinical Consultation Report** | `84` | `337` | `180.9` | `12` | `51` | `26.8` |
| **Target Oncology Summary** | `87` | `170` | `124.1` | `13` | `24` | `16.9` |

---

## 5. Synthetic Safety & PII Audit
- **Personally Identifiable Information (PII) Check**:
  - Email Addresses Found: `0`
  - Phone Numbers Found: `0`
  - Social Security Numbers Found: `0`
  - Patient ID Syntax: `100% match standard synthetic regex ^SYN-\d{6}$ (10,000 / 10,000)`
- **Synthetic Flag Verification**: All parsed contexts explicitly assert `"SYNTHETIC": true`.
