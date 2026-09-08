# Stage 1 ML — Data Quality Audit Report

## 1. Dataset Dimensions & Overview
- **Total Records**: 5012
- **Total Features**: 34
- **Targets**: 3 (`overall_patient_risk` [Primary], `toxicity_risk` [Secondary], `therapy_response` [Secondary])
- **Exact Duplicate Rows**: 12 (Removed)

## 2. Missing Values Analysis
Missing values were identified in 7 numerical features:
- `renal_function`: 123 missing (2.45%)
- `liver_function`: 117 missing (2.33%)
- `hemoglobin`: 135 missing (2.69%)
- `ctDNA_level`: 103 missing (2.06%)
- `biomarker_1`: 120 missing (2.39%)
- `biomarker_2`: 120 missing (2.39%)
- `albumin`: 141 missing (2.81%)

## 3. Data Integrity & Outliers
- **Constant Columns**: None
- **Near-Zero Variance Columns (<0.01)**: None
- **Top Outlier Features (IQR 1.5 threshold)**:
  - `ctDNA_level`: 290 potential outliers
  - `inflammatory_marker`: 266 potential outliers
  - `baseline_tumor_volume`: 264 potential outliers
  - `performance_status`: 197 potential outliers
  - `mutation_burden`: 170 potential outliers

## 4. Target Distributions & Definition

### Primary Target: `overall_patient_risk`
Defines overall patient risk by integrating adverse reaction vulnerability (`toxicity_risk`) and therapeutic efficacy (`therapy_response`):
- **High Risk**: `toxicity_risk == 'High'` OR `therapy_response == 'Non-Responder'`
- **Low Risk**: `toxicity_risk == 'Low'` AND `therapy_response == 'Complete Response'`
- **Moderate Risk**: All intermediate clinical trajectories.

**Class Distribution**:
- **High**: 2511 (50.2%)
- **Moderate**: 1775 (35.5%)
- **Low**: 714 (14.3%)

### Secondary Targets:
- **`toxicity_risk`**: Low (2025), High (1954), Moderate (1021)
- **`therapy_response`**: Partial Response (2306), Complete Response (1777), Non-Responder (917)
