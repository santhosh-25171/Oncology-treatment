# Stage 4 SLM — Preprocessing & Data Leakage Prevention Report

> **DISCLAIMER**: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.  
> Documents the cleaning transformations, context normalization, and patient-level splitting results for Stage 4 SLM fine-tuning.

---

## 1. Cleaning & Preprocessing Transformations

1. **Deduplication**:
   - Identified and removed `144` duplicate records (`115` raw exact duplicates + `29` post-normalization duplicates).
   - Preserved all `3,200` unique patient cohorts.
2. **Missing Non-Critical Value Imputation**:
   - Missing `source_type` imputed as standardized category `clinical_note`.
   - Missing Stage 1-3 contexts parsed as explicit `status: "missing_stageX_context"` with `"SYNTHETIC": true`.
3. **Source Type Standardization**:
   - Normalized all source types to lower snake_case (e.g. `Imaging Report` $\to$ `imaging_report`).
4. **Text Cleaning**:
   - Normalized internal spacing and eliminated stray spaces before punctuation in consultation reports and target summaries.
5. **Heterogeneous Context Normalization**:
   - Extracted key-value pairs from heterogeneous string serializations (`(synthetic stage1) key: val`, `[SYNTHETIC-STAGE2] key=val`, etc.) into validated, sorted, compact JSON objects.
6. **Instruction Tuning Prompt Construction**:
   - Formatted `slm_prompt` combining task instruction, normalized clinical report, and multimodal context.

---

## 2. Patient-Level Grouped Splitting (Zero-Leakage Strategy)

A **deterministic patient-level grouped split** was applied with fixed random seed `seed=42`.

### Patient Counts per Partition
- **Total Unique Patients**: `3,200`
- **Training Patients**: `2,560` (`80.00%`)
- **Validation Patients**: `320` (`10.00%`)
- **Test Patients**: `320` (`10.00%`)

### Record Counts per Partition
- **Total Deduplicated Notes**: `9,856`
- **Training Notes**: `7,896` (`80.11%`)
- **Validation Notes**: `1,010` (`10.25%`)
- **Test Notes**: `950` (`9.64%`)

---

## 3. Data Leakage Verification Results

| Leakage Dimension | Overlap Count | Verification Status |
| :--- | :---: | :---: |
| **Patient ID: Train $\cap$ Validation** | `0` | 🟢 **ZERO LEAKAGE** |
| **Patient ID: Train $\cap$ Test** | `0` | 🟢 **ZERO LEAKAGE** |
| **Patient ID: Validation $\cap$ Test** | `0` | 🟢 **ZERO LEAKAGE** |
| **Clinical Report Text: Train $\cap$ Val** | `0` | 🟢 **ZERO LEAKAGE** |
| **Clinical Report Text: Train $\cap$ Test** | `0` | 🟢 **ZERO LEAKAGE** |
| **Clinical Report Text: Val $\cap$ Test** | `0` | 🟢 **ZERO LEAKAGE** |

---

## 4. Generated Artifact Locations
- Raw Dataset: `C:\Users\santh\OneDrive - Rathinam Group Of Institutions\Desktop\Oncology patient prediction\Oncology-treatment\personalized_precision_oncology\stage4_slm\data\raw\oncology_stage4_raw_10000.csv`
- Processed Dataset: `stage4_slm/data/processed/stage4_slm_processed_dataset.csv`
- Training Partition: `stage4_slm/data/splits/train.csv`
- Validation Partition: `stage4_slm/data/splits/validation.csv`
- Testing Partition: `stage4_slm/data/splits/test.csv`
- Oncology Domain Dictionary: `stage4_slm/domain/oncology_dictionary.json`
