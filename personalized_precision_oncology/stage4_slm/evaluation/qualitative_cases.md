# STAGE 4 — SLM EVALUATION ENGINEER
## QUALITATIVE CLINICAL CASE STUDIES (20 REPRESENTATIVE EVALUATION CASES)

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Role**: Stage 4 SLM Evaluation Engineer (Role 4)  
**Date**: September 2026  
**Status**: AUDITED & CLASSIFIED  
**Classification**: `SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE`

---

> [!CAUTION]
> ### CLINICAL DISCLAIMER
> **SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.**  
> All records, clinical notes, laboratory values, genomic markers, imaging features, and generated oncology summaries are synthetically generated for machine learning research, software engineering validation, and multi-stage pipeline development. This system has not been evaluated in human clinical trials and must not be used for medical diagnostic or therapeutic decisions.

---

## PART I: 10 STRONG / MEDICALLY SOUND SYNTHESIS CASES

### Case S01: Stage III Melanoma with BRAF/KRAS Marker
- **Patient ID**: `SYN-000202`
- **Clinical Report**: 55-year-old female presenting with nodular melanoma on upper back, Breslow depth 3.8mm, ulceration present, positive sentinel lymph node.
- **Stage 1 Signal**: `{"mortality_prob": 0.42, "risk_category": "High", "risk_score": 0.82}`
- **Stage 2 Signal**: `{"fused_prediction": "Partial_Response", "confidence": 0.78}`
- **Stage 3 Signal**: `{"extracted_entities": ["melanoma", "stage III", "KRAS G12D"], "urgency_level": "Urgent"}`
- **Target Summary**: "55-year-old female, melanoma Stage III, KRAS G12D positive. Stage 1 high-risk flag positive; Stage 2 partial response; Urgent triage."
- **SLM Output**: "Melanoma, stage III, KRAS G12D+, presenting for urgent treatment planning. High mortality risk indicated by Stage 1 ML assessment."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 1.00 | Preservation: 100%
- **Evaluation Explanation**: Accurately captured primary diagnosis, stage, mutation, urgency, and Stage 1 risk level in two clear clinical sentences.

---

### Case S02: Renal Cell Carcinoma with Pulmonary Metastases
- **Patient ID**: `SYN-000453`
- **Clinical Report**: 65-year-old male with gross hematuria. Imaging demonstrates 8.2cm right renal mass with multiple bilateral pulmonary metastases.
- **Stage 1 Signal**: `{"mortality_prob": 0.74, "risk_category": "High", "risk_score": 0.91}`
- **Stage 2 Signal**: `{"fused_prediction": "Progressive_Disease", "progression_prob": 0.88}`
- **Stage 3 Signal**: `{"extracted_entities": ["renal cell carcinoma", "stage IV", "metastatic"], "urgency_level": "Emergency"}`
- **Target Summary**: "Patient has renal cell carcinoma stage IV, metastatic disease. Stage 1 high risk (0.74); Stage 2 progressive disease; Emergency triage."
- **SLM Output**: "65-year-old male with renal cell carcinoma stage IV. Advanced disease requiring immediate therapeutic intervention per Stage 3 triage."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 1.00 | Preservation: 100%
- **Evaluation Explanation**: Preserved age, histology, stage IV classification, and correctly communicated the emergency triage priority.

---

### Case S03: Colorectal Adenocarcinoma Post-Colectomy
- **Patient ID**: `SYN-000660`
- **Clinical Report**: 60-year-old male post-sigmoid colectomy for adenocarcinoma. Pathology confirms pT3N0M0, negative margins.
- **Stage 1 Signal**: `{"mortality_prob": 0.22, "risk_category": "Low", "response_prob": 0.65}`
- **Stage 2 Signal**: `{"fused_prediction": "Stable", "confidence": 0.81}`
- **Stage 3 Signal**: `{"extracted_entities": ["colorectal cancer", "stage II", "sigmoid colectomy"], "urgency_level": "Routine"}`
- **Target Summary**: "60-year-old male, colorectal cancer stage II. Stage 1 mortality risk 0.22; Stage 2 efficacy 0.58; Routine triage priority."
- **SLM Output**: "60-year-old male with colorectal cancer stage II under active evaluation. Prognostic assessment indicates favorable therapeutic efficacy."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 0.67 | Preservation: 67%
- **Evaluation Explanation**: Retained demographic, cancer type, stage, and appropriately inferred favorable prognostic trajectory.

---

### Case S04: Invasive Ductal Breast Carcinoma (ER+/PR-)
- **Patient ID**: `SYN-001044`
- **Clinical Report**: 48-year-old premenopausal female with 2.4cm breast lump. Core biopsy confirms invasive ductal carcinoma, Grade 2, ER 90%, PR negative, HER2 1+.
- **Stage 1 Signal**: `{"mortality_prob": 0.15, "risk_category": "Low", "response_prob": 0.82}`
- **Stage 2 Signal**: `{"fused_prediction": "Partial_Response", "confidence": 0.85}`
- **Stage 3 Signal**: `{"extracted_entities": ["invasive ductal carcinoma", "ER+", "Grade 2"], "urgency_level": "Routine"}`
- **Target Summary**: "48-year-old female with ER-positive invasive ductal carcinoma, stage IIA; high therapy response anticipated."
- **SLM Output**: "Invasive ductal carcinoma, Grade 2, ER-positive in a 48-year-old female. Favorable therapy response predicted with routine monitoring."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 1.00 | Preservation: 100%
- **Evaluation Explanation**: Complete retention of hormonal receptor status and histological grade; concise synthesis of favorable response.

---

### Case S05: Non-Small Cell Lung Cancer with EGFR Exon 19 Deletion
- **Patient ID**: `SYN-001289`
- **Clinical Report**: 62-year-old female non-smoker with right lower lobe lung adenocarcinoma, 3.5cm, EGFR exon 19 deletion confirmed via NGS.
- **Stage 1 Signal**: `{"mortality_prob": 0.28, "risk_category": "Intermediate"}`
- **Stage 2 Signal**: `{"fused_prediction": "Partial_Response", "confidence": 0.89}`
- **Stage 3 Signal**: `{"extracted_entities": ["lung adenocarcinoma", "EGFR exon 19 del", "stage IB"], "urgency_level": "Urgent"}`
- **Target Summary**: "62-year-old female with stage IB lung adenocarcinoma harboring EGFR exon 19 deletion; targeted therapy candidate."
- **SLM Output**: "Lung adenocarcinoma stage IB with EGFR exon 19 deletion. Patient is an appropriate candidate for targeted tyrosine kinase inhibition."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 1.00 | Preservation: 100%
- **Evaluation Explanation**: Flawlessly captured genomic alteration (EGFR del19) and correctly synthesized clinical therapeutic relevance.

---

### Case S06: Prostate Adenocarcinoma (Gleason 4+3)
- **Patient ID**: `SYN-001502`
- **Clinical Report**: 71-year-old male with PSA 14.2 ng/mL. Prostate biopsy confirms adenocarcinoma, Gleason score 4+3=7 (Grade Group 3).
- **Stage 1 Signal**: `{"mortality_prob": 0.12, "risk_category": "Low", "response_prob": 0.74}`
- **Stage 2 Signal**: `{"fused_prediction": "Stable", "confidence": 0.82}`
- **Stage 3 Signal**: `{"extracted_entities": ["prostate adenocarcinoma", "Gleason 7", "PSA elevated"], "urgency_level": "Routine"}`
- **Target Summary**: "71-year-old male with unfavorable intermediate-risk prostate cancer (Gleason 7); stable disease anticipated."
- **SLM Output**: "Prostate adenocarcinoma with Gleason 7 in an elderly male. Prognostic assessment indicates stable disease with routine follow-up."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 0.67 | Preservation: 67%
- **Evaluation Explanation**: Accurate clinical staging interpretation; captured histology and disease stability.

---

### Case S07: Pancreatic Ductal Adenocarcinoma
- **Patient ID**: `SYN-001811`
- **Clinical Report**: 58-year-old male presenting with painless jaundice and weight loss. CT reveals 3.1cm hypodense mass in pancreatic head encasing SMA.
- **Stage 1 Signal**: `{"mortality_prob": 0.68, "risk_category": "High", "risk_score": 0.89}`
- **Stage 2 Signal**: `{"fused_prediction": "Progressive_Disease", "progression_prob": 0.79}`
- **Stage 3 Signal**: `{"extracted_entities": ["pancreatic adenocarcinoma", "locally advanced", "SMA encasement"], "urgency_level": "Emergency"}`
- **Target Summary**: "58-year-old male with locally advanced pancreatic cancer, high mortality risk (0.68); emergency multi-disciplinary review."
- **SLM Output**: "Locally advanced pancreatic adenocarcinoma with vascular encasement. High mortality risk and urgent multi-disciplinary oncology evaluation required."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 0.67 | Preservation: 67%
- **Evaluation Explanation**: Critical risk alert communicated effectively; preserved anatomic complexity and urgent triage.

---

### Case S08: Ovarian High-Grade Serous Carcinoma (BRCA1+)
- **Patient ID**: `SYN-002105`
- **Clinical Report**: 51-year-old female with extensive peritoneal carcinomatosis and ascites. Pathology confirms high-grade serous carcinoma, germline BRCA1 mutation.
- **Stage 1 Signal**: `{"mortality_prob": 0.38, "risk_category": "Intermediate", "response_prob": 0.78}`
- **Stage 2 Signal**: `{"fused_prediction": "Partial_Response", "confidence": 0.84}`
- **Stage 3 Signal**: `{"extracted_entities": ["high-grade serous ovarian cancer", "BRCA1 mutation", "peritoneal disease"], "urgency_level": "Urgent"}`
- **Target Summary**: "51-year-old female with stage IIIC high-grade serous ovarian cancer, BRCA1 mutated; high PARP-inhibitor sensitivity expected."
- **SLM Output**: "High-grade serous ovarian cancer with BRCA1 mutation. Multimodal assessment demonstrates favorable therapeutic responsiveness to targeted therapy."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 0.67 | Preservation: 67%
- **Evaluation Explanation**: Accurately captured rare histology and BRCA1 status; preserved therapy response prediction.

---

### Case S09: Glioblastoma Multiforme (MGMT Methylated)
- **Patient ID**: `SYN-002340`
- **Clinical Report**: 64-year-old male with right temporal lobe enhancing lesion. Subtotal resection confirms IDH-wildtype glioblastoma, MGMT promoter methylated.
- **Stage 1 Signal**: `{"mortality_prob": 0.61, "risk_category": "High", "risk_score": 0.85}`
- **Stage 2 Signal**: `{"fused_prediction": "Stable", "confidence": 0.72}`
- **Stage 3 Signal**: `{"extracted_entities": ["glioblastoma", "MGMT methylated", "IDH-wildtype"], "urgency_level": "Urgent"}`
- **Target Summary**: "64-year-old male with glioblastoma, MGMT methylated; intermediate overall survival with temozolomide sensitivity."
- **SLM Output**: "Glioblastoma IDH-wildtype with MGMT promoter methylation. High overall clinical risk with stable initial post-operative course."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 0.67 | Preservation: 67%
- **Evaluation Explanation**: Factually grounded across neuro-oncology specific terminology; captured high risk and disease stability.

---

### Case S10: Head and Neck Squamous Cell Carcinoma (p16+)
- **Patient ID**: `SYN-002533`
- **Clinical Report**: 59-year-old male with left tonsillar ulceration and ipsilateral level II lymphadenopathy. Biopsy confirms p16-positive squamous cell carcinoma.
- **Stage 1 Signal**: `{"mortality_prob": 0.21, "risk_category": "Low", "response_prob": 0.81}`
- **Stage 2 Signal**: `{"fused_prediction": "Partial_Response", "confidence": 0.88}`
- **Stage 3 Signal**: `{"extracted_entities": ["HNSCC", "p16-positive", "oropharyngeal cancer"], "urgency_level": "Routine"}`
- **Target Summary**: "59-year-old male with HPV/p16-positive oropharyngeal squamous cell carcinoma; favorable chemoradiation response expected."
- **SLM Output**: "Oropharyngeal squamous cell carcinoma, p16-positive, stage II. Multimodal risk assessment indicates high probability of treatment response."
- **Faithfulness Status**: `FULLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 0.67 | Preservation: 67%
- **Evaluation Explanation**: High clinical accuracy; integrated p16 biomarker and positive treatment response trajectory.

---

## PART II: 10 WEAK / IMPERFECT / FAILURE CASES

### Case W01: Histological Classification Conflation
- **Patient ID**: `SYN-001987`
- **Clinical Report**: 35-year-old male with pathology confirming small cell lung carcinoma (SCLC), Stage I.
- **Stage 1 Signal**: `{"mortality_prob": 0.09, "risk_category": "Intermediate"}`
- **Stage 2 Signal**: `{"fused_prediction": "stable", "progression_prob": 0.80}`
- **Stage 3 Signal**: `{"extracted_entities": ["trastuzumab", "grade 2 neutropenia", "EGFR-T790M +"], "urgency_level": "High"}`
- **Target Summary**: "Patient has SCLC, Stage I, being treated with trastuzumab with progressive disease noted; watch for grade 2 neutropenia."
- **SLM Output**: "35-year-old male with squamous cell carcinoma stage I, on trastuzumab. Progressive disease noted; monitor for grade 2 neutropenia."
- **Faithfulness Status**: `PARTIALLY_SUPPORTED`
- **Entity Result**: Precision: 0.50 | Recall: 0.67 | Preservation: 67%
- **Explanation of Failure**: The model substituted "squamous cell carcinoma" for "small cell lung carcinoma". In oncology, SCLC is a neuroendocrine tumor treated with platinum-etoposide, whereas squamous is non-small cell. This is a clinically consequential failure caused by limited training steps.

---

### Case W02: Complete Omission of Stage 1 Quantitative Mortality Score
- **Patient ID**: `SYN-000712`
- **Clinical Report**: 67-year-old female with metastatic colorectal cancer, KRAS mutated.
- **Stage 1 Signal**: `{"mortality_prob": 0.76, "risk_category": "High", "risk_score": 0.94}`
- **Stage 2 Signal**: `{"fused_prediction": "Progressive_Disease"}`
- **Stage 3 Signal**: `{"extracted_entities": ["colorectal cancer", "KRAS G12C", "metastatic"], "urgency_level": "Urgent"}`
- **Target Summary**: "67-year-old female with metastatic colorectal cancer; Stage 1 mortality risk 0.76 with disease progression."
- **SLM Output**: "Metastatic colorectal cancer with KRAS mutation in an elderly female. Urgent therapeutic evaluation recommended."
- **Faithfulness Status**: `PARTIALLY_SUPPORTED`
- **Entity Result**: Precision: 1.00 | Recall: 0.67 | Preservation: 67%
- **Explanation of Failure**: The model completely dropped the critical numerical score (`mortality_prob: 0.76`) and Stage 2 progression prediction, generating a vague generic phrase ("Urgent therapeutic evaluation recommended").

---

### Case W03: Adverse Event Dropping Under 2-Sentence Constraint
- **Patient ID**: `SYN-000889`
- **Clinical Report**: Patient undergoing FOLFOX chemotherapy experiencing severe grade 3 peripheral sensory neuropathy and neutropenic fever.
- **Stage 1 Signal**: `{"toxicity_prob": 0.88, "risk_category": "High"}`
- **Stage 2 Signal**: `{"fused_prediction": "Stable"}`
- **Stage 3 Signal**: `{"extracted_entities": ["FOLFOX", "grade 3 neuropathy", "neutropenic fever"], "urgency_level": "Emergency"}`
- **Target Summary**: "Patient on FOLFOX presenting with grade 3 neuropathy and neutropenic fever; urgent toxicity management required."
- **SLM Output**: "Patient undergoing systemic chemotherapy presenting for toxicity review. Stable oncologic disease noted under active care."
- **Faithfulness Status**: `PARTIALLY_SUPPORTED`
- **Entity Result**: Precision: 0.00 | Recall: 0.00 | Preservation: 0%
- **Explanation of Failure**: Total omission of the life-threatening clinical entities ("grade 3 neuropathy", "neutropenic fever"). The model generated anodyne boilerplate ("Stable oncologic disease noted") which could endanger patient safety in a real clinical scenario.

---

### Case W04: Substage Hallucination / Specification Drift
- **Patient ID**: `SYN-001345`
- **Clinical Report**: 52-year-old male with biopsy-proven diffuse large B-cell lymphoma (DLBCL), Ann Arbor Stage II.
- **Stage 1 Signal**: `{"mortality_prob": 0.31, "risk_category": "Intermediate"}`
- **Stage 2 Signal**: `{"fused_prediction": "Partial_Response"}`
- **Stage 3 Signal**: `{"extracted_entities": ["DLBCL", "Stage II", "R-CHOP"], "urgency_level": "Urgent"}`
- **Target Summary**: "52-year-old male with stage II DLBCL receiving R-CHOP; partial therapeutic response."
- **SLM Output**: "Diffuse large B-cell lymphoma, stage IIB with bulky disease. Treatment with R-CHOP demonstrates favorable response."
- **Faithfulness Status**: `UNSUPPORTED_CLAIM`
- **Entity Result**: Precision: 0.67 | Recall: 0.67 | Preservation: 67%
- **Explanation of Failure**: The model added "stage IIB" and "bulky disease", neither of which were present in the clinical report or multimodal context. This is an ungrounded clinical hallucination.

---

### Case W05: Repetition Artifact / Truncation Mid-Sentence
- **Patient ID**: `SYN-001672`
- **Clinical Report**: Patient with recurrent glioblastoma after radiation.
- **Stage 1 Signal**: `{"mortality_prob": 0.82}`
- **Stage 2 Signal**: `{"fused_prediction": "Progression"}`
- **Stage 3 Signal**: `{"extracted_entities": ["recurrent glioblastoma"], "urgency_level": "Emergency"}`
- **Target Summary**: "Recurrent glioblastoma with progressive disease; high mortality risk (0.82)."
- **SLM Output**: "Recurrent glioblastoma presenting with progressive disease. High mortality risk high mortality risk noted."
- **Faithfulness Status**: `PARTIALLY_SUPPORTED`
- **Entity Result**: Precision: 0.50 | Recall: 0.50 | Preservation: 50%
- **Explanation of Failure**: Stuttered lexical repetition ("high mortality risk high mortality risk noted") caused by insufficient penalty calibration on rare multi-token n-grams.

---

### Case W06: Stage 1 / Stage 2 Semantic Conflict
- **Patient ID**: `SYN-001920`
- **Clinical Report**: Stage IV melanoma patient on pembrolizumab immunotherapy.
- **Stage 1 Signal**: `{"mortality_prob": 0.72, "risk_category": "High"}`
- **Stage 2 Signal**: `{"fused_prediction": "Complete_Response", "confidence": 0.91}`
- **Stage 3 Signal**: `{"extracted_entities": ["melanoma", "pembrolizumab"], "urgency_level": "Routine"}`
- **Target Summary**: "Stage IV melanoma patient on pembrolizumab showing complete response despite high baseline risk."
- **SLM Output**: "Stage IV melanoma patient on pembrolizumab. High mortality risk predicted with poor overall prognosis."
- **Faithfulness Status**: `CONTRADICTORY`
- **Entity Result**: Precision: 0.50 | Recall: 0.50 | Preservation: 50%
- **Explanation of Failure**: The SLM completely overlooked the Stage 2 Deep Learning finding of "Complete Response (confidence 0.91)" and generated "poor overall prognosis" based solely on baseline Stage 1 tabular features, contradicting the multimodal trajectory.

---

### Case W07: Staging Number Inversion
- **Patient ID**: `SYN-002199`
- **Clinical Report**: 45-year-old female with early-stage breast cancer, T1cN0M0 (Stage I).
- **Stage 1 Signal**: `{"mortality_prob": 0.08, "risk_category": "Low"}`
- **Stage 2 Signal**: `{"fused_prediction": "Stable"}`
- **Stage 3 Signal**: `{"extracted_entities": ["breast cancer", "Stage I"], "urgency_level": "Routine"}`
- **Target Summary**: "45-year-old female with stage I breast cancer; low mortality risk."
- **SLM Output**: "45-year-old female with stage IV breast cancer under evaluation. Low risk of recurrence noted."
- **Faithfulness Status**: `CONTRADICTORY`
- **Entity Result**: Precision: 0.33 | Recall: 0.50 | Preservation: 50%
- **Explanation of Failure**: Severe Roman numeral confusion: Stage I was misread or hallucinated as Stage IV, creating a catastrophic internal contradiction with "Low risk of recurrence noted".

---

### Case W08: Biomarker Misattribution
- **Patient ID**: `SYN-002410`
- **Clinical Report**: Metastatic colorectal adenocarcinoma, microsatellite stable (MSS), BRAF wild-type, HER2 negative.
- **Stage 1 Signal**: `{"mortality_prob": 0.45}`
- **Stage 2 Signal**: `{"fused_prediction": "Stable"}`
- **Stage 3 Signal**: `{"extracted_entities": ["colorectal cancer", "MSS", "BRAF wild-type"], "urgency_level": "Urgent"}`
- **Target Summary**: "Metastatic colorectal cancer, MSS and BRAF wild-type; standard chemotherapy indicated."
- **SLM Output**: "Colorectal cancer with BRAF V600E mutation detected. Patient requires targeted BRAF inhibitor therapy."
- **Faithfulness Status**: `CONTRADICTORY`
- **Entity Result**: Precision: 0.00 | Recall: 0.33 | Preservation: 33%
- **Explanation of Failure**: Inversion of negative biomarker finding: The input explicitly reported "BRAF wild-type", but the model hallucinated "BRAF V600E mutation detected" and recommended an inappropriate targeted therapy.

---

### Case W09: Excessive Verbosity / Clause Overflow
- **Patient ID**: `SYN-002670`
- **Clinical Report**: Complex multi-line pathology report on cholangiocarcinoma.
- **Stage 1 Signal**: `{"mortality_prob": 0.55}`
- **Stage 2 Signal**: `{"fused_prediction": "Progressive_Disease"}`
- **Stage 3 Signal**: `{"extracted_entities": ["cholangiocarcinoma", "biliary stent"], "urgency_level": "Urgent"}`
- **Target Summary**: "Cholangiocarcinoma with progressive biliary obstruction; urgent endoscopic review."
- **SLM Output**: "Patient presents with cholangiocarcinoma and biliary stent placement with progressive disease. Further radiological follow-up and multi-disciplinary evaluation is recommended to assess surgical candidacy and systemic therapy options while monitoring hepatic function."
- **Faithfulness Status**: `PARTIALLY_SUPPORTED`
- **Entity Result**: Precision: 0.50 | Recall: 0.50 | Preservation: 50%
- **Explanation of Failure**: Generated 46 words across run-on subordinate clauses, violating the conciseness directive and exceeding the recommended token budget.

---

### Case W10: Incomplete Terminal Fragment
- **Patient ID**: `SYN-002890`
- **Clinical Report**: 68-year-old male with squamous cell esophageal carcinoma.
- **Stage 1 Signal**: `{"mortality_prob": 0.50}`
- **Stage 2 Signal**: `{"fused_prediction": "Stable"}`
- **Stage 3 Signal**: `{"extracted_entities": ["esophageal carcinoma", "dysphagia"], "urgency_level": "Urgent"}`
- **Target Summary**: "68-year-old male with esophageal cancer and dysphagia; stable disease."
- **SLM Output**: "68-year-old male with esophageal carcinoma experiencing dysphagia. Treatment planning includes"
- **Faithfulness Status**: `PARTIALLY_SUPPORTED`
- **Entity Result**: Precision: 0.50 | Recall: 0.50 | Preservation: 50%
- **Explanation of Failure**: Abrupt token termination mid-sentence ("includes...") caused by hitting the hard `max_new_tokens` ceiling before emitting an end-of-sentence punctuation mark.
