# Stage 03 NLP - Annotation Report

## Dataset
- **Records Processed (Cleaned Dataset):** 10015
- **Successfully Annotated:** 9990
- **Excluded Records:** 25 (Empty texts: 10, Duplicate IDs: 15)

## Urgency Distribution
- **LOW:** 6185 (61.9%)
- **MODERATE:** 1577 (15.8%)
- **HIGH:** 2228 (22.3%)

## NER Distribution
- **GENE_MUTATION:** 4572
- **DRUG_NAME:** 7213
- **DOSAGE_LEVEL:** 4053
- **ADVERSE_EVENT:** 21331

## Quality Checks
- **Invalid Spans:** 0
- **Annotation Validation Result:** PASS

## Examples
### Example 1
**Text:** My dyspnea has gotten a little better but I still feel moderate pneumonitis. Previously treated with docetaxel with intermittent thrombocytopenia reported at that time. On today's visit, dyspnea is noted to be grade 3. Not sure if this is normal, wanted to check.

**Urgency:** HIGH

**Entities:**
- [ADVERSE_EVENT] `dyspnea` (chars 3:10)
- [ADVERSE_EVENT] `pneumonitis` (chars 64:75)
- [DRUG_NAME] `docetaxel` (chars 101:110)
- [ADVERSE_EVENT] `thrombocytopenia` (chars 129:145)
- [ADVERSE_EVENT] `dyspnea` (chars 187:194)

---
### Example 2
**Text:** Pt checked in for hormonal therapy appointment, tolerating well. History of moderate anemia during the previous chemotherapy cycle. Vitals within normal limits: BP 108/70, HR 74, Temp 98.1F, RR 14.

**Urgency:** LOW

**Entities:**
- [ADVERSE_EVENT] `anemia` (chars 85:91)

---
### Example 3
**Text:** Specimen: surgical resection, brain. Biopsy of the lung reveals findings consistent with ovarian cancer. Staging consistent with stage II prostate cancer, with possible involvement of the lymph nodes. Findings were discussed with the treating oncology team.

**Urgency:** LOW

**Entities:**

---
### Example 4
**Text:** ECOG 3. Patient returns for reassessment following recent targeted therapy. No signs of abdominal pain on exam. Previously treated with nivolumab with mild mucositis reported at that time. Following radiation therapy with cisplatin, patient reported moderate rash. Recommend dose adjustment of gefitinib due to persistent nausea. Vitals: BP 140/90, HR 102, Temp 101.4F, RR 22.

**Urgency:** MODERATE

**Entities:**
- [ADVERSE_EVENT] `abdominal pain` (chars 88:102)
- [DRUG_NAME] `nivolumab` (chars 136:145)
- [ADVERSE_EVENT] `mucositis` (chars 156:165)
- [DRUG_NAME] `cisplatin` (chars 222:231)
- [ADVERSE_EVENT] `rash` (chars 259:263)
- [DRUG_NAME] `gefitinib` (chars 294:303)
- [ADVERSE_EVENT] `nausea` (chars 322:328)

---
### Example 5
**Text:** I wanted to let my doctor know I've had abdominal pain for the past couple of days. Patient denies anemia. Please let me know if I need to come in.

**Urgency:** LOW

**Entities:**
- [ADVERSE_EVENT] `abdominal pain` (chars 40:54)
- [ADVERSE_EVENT] `anemia` (chars 99:105)

---
