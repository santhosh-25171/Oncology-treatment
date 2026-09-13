# ROLE REPORT 1: DATA ENGINEERING
## Personalized Precision Medicine for Oncology Treatment Optimization

```
======================================================================================================
ROLE:               Data Engineer
PROJECT:            Personalized Precision Medicine for Oncology Treatment Optimization
PIPELINE STAGES:    Data Ingestion | Biological Range Sanitization | Multimodal Alignment | 
                    Train-Only Imputation | Zero-Leakage Patient Hashing | DataLoader Engineering
TEST STATUS:        100% Passing (Schema Validation, Null-Check Assertions, Leakage Verification)
======================================================================================================
```

---

## 1. Executive Mission & Role Definition
The **Data Engineer** is responsible for architecting, validating, scaling, and maintaining the data foundation of the AI system. In clinical oncology, data engineering directly impacts patient safety. Flawed data ingestion or subtle data leakage produces models that look artificially stellar in validation but fail catastrophically at the bedside.

The primary mandate is to convert raw, noisy, heterogeneous hospital records (tabular EHR, histopathology slide tiles, longitudinal biomarker time-series, and clinical consultation dictations) into clean, aligned, leakage-free data matrices and PyTorch tensors.

---

## 2. Theoretical Concepts & Foundational Principles

### 2.1 Multimodal Clinical Data Heterogeneity
Clinical medicine is fundamentally multimodal. A single patient generates distinct data archetypes with incompatible dimensions, sampling frequencies, and noise characteristics:
1. **Tabular EHR Data**: Demographic attributes ($Age, Sex$), clinical staging ($TNM$), and baseline laboratory biochemistry ($eGFR, Bilirubin, Platelets$). 
   - *Mathematical Representation*: Vector $\mathbf{x}_{\text{tab}} \in \mathbb{R}^{d}$.
2. **Digital Pathology Imaging (Vision)**: Whole Slide Images (WSI) stained with Hematoxylin & Eosin (H&E). Biopsy tissue tiles capture cellular atypia and mitotic density.
   - *Mathematical Representation*: 3D Spatial Tensor $\mathbf{X}_{\text{img}} \in \mathbb{R}^{3 \times 224 \times 224}$.
3. **Longitudinal Lab Biomarkers (Time-Series)**: Serial blood draws tracking treatment response across multiple clinic visits ($t_0, t_1, \dots, t_K$).
   - *Mathematical Representation*: 3D Sequential Tensor $\mathbf{X}_{\text{seq}} \in \mathbb{R}^{N \times K \times D}$.
4. **Unstructured Clinical Text & Speech**: Free-text physician progress notes and spoken clinical dictation audio.
   - *Mathematical Representation*: Raw 16kHz audio waveforms $\mathbf{x}_{\text{audio}} \in \mathbb{R}^{L}$ and tokenized text sequence vectors.

### 2.2 Missing Data Mechanisms in Healthcare
In machine learning, missing data is not merely "empty cells"—it is governed by distinct probabilistic mechanisms:
- **MCAR (Missing Completely at Random)**: The probability of missingness is independent of both observed and unobserved data:
  $$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M)$$
  *Example*: A laboratory blood vial accidentally dropped on the floor and broke.
- **MAR (Missing at Random)**: Missingness depends systematically on observed covariates, but not on the unobserved missing value itself:
  $$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M \mid Y_{\text{obs}})$$
  *Example*: Older patients ($Age > 70$) are routinely ordered echocardiograms, whereas young, fit patients rarely have this test recorded.
- **MNAR (Missing Not at Random)**: Missingness is directly correlated with the actual missing value:
  $$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) \neq P(M \mid Y_{\text{obs}})$$
  *Example*: Extremely ill, bedridden patients are unable to complete a physical functional mobility assessment; the missingness is caused by the patient's severe impairment.

### 2.3 The Mechanics of Data Leakage (The Cardinal Sin)
Data leakage occurs when information from outside the training partition enters the model during preprocessing or feature transformation. In clinical AI, leakage inflates test accuracy and hides model failure:
1. **Global Imputation Leakage**: Calculating global statistics (e.g., global median of `renal_function`) across all 5,000 patients prior to splitting. When test set values influence the imputation mean, the test set distribution has leaked into the training set.
2. **Scaler Contamination**: Fitting normalization scalers (`StandardScaler`, `MinMaxScaler`) on the combined dataset. Test set outliers distort the scaling bounds applied during training.
3. **Patient Contamination Leakage (Multi-Visit Leakage)**: A single patient visited the hospital 4 times, generating 4 separate rows. If a random row-based split places Visit 1 and Visit 2 in Train, and Visit 3 in Test, the model memorizes Patient X’s specific genetic markers and baseline biology. The test score measures patient recognition rather than true disease generalization.

---

## 3. Concrete Engineering Workflows & Implementation

### 3.1 End-to-End Data Pipeline Architecture

```text
[Raw EHR CSV / WSI / Audio Sources]
                  │
                  ▼
   1. [Biological Range Sanitization]
      - Filter impossible ages (<18 or >105)
      - Enforce physiological bounds (eGFR > 0, BP > 0)
                  │
                  ▼
   2. [Deterministic Patient Hashing]
      - Hash(Patient_ID) mod 100
      - Enforce 70% Train, 15% Val, 15% Test
      - ZERO Patient ID Overlap
                  │
                  ▼
   3. [Train-Only Preprocessing Fit]
      - Fit SimpleImputer(strategy='median') on X_train only
      - Fit RobustScaler on X_train only
      - Fit OneHotEncoder(handle_unknown='ignore') on X_train only
                  │
                  ▼
   4. [Transform Val & Test Sets]
      - Transform X_val and X_test with FROZEN parameters
                  │
                  ▼
   5. [PyTorch Tensor Serialization & DataLoader Packaging]
      - Parquet feature matrices
      - Pinned memory PyTorch DataLoaders with sequence masking
```

### 3.2 Biological Range Sanitization Logic
The pipeline discards or flags physiologically impossible entries using deterministic rule engines:
```python
# Physiological validation boundaries
CLEANING_RULES = {
    "age": (18, 105),
    "bmi": (12.0, 65.0),
    "systolic_bp": (60, 260),
    "diastolic_bp": (40, 150),
    "renal_function": (5.0, 160.0), # eGFR mL/min/1.73m2
    "platelet_count": (10, 1500),    # x10^9/L
    "treatment_dose": (5.0, 300.0)   # mg/m2
}
```

### 3.3 Leakage-Free Splitting Implementation
To guarantee zero patient contamination, the Data Engineer implements patient-level stratification:
```python
# Strict patient-level stratification
X = df.drop(columns=targets)
y = df[targets]
stratify_label = y['overall_patient_risk'].astype(str)

# Step 1: Split off 15% Holdout Test Set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=stratify_label, random_state=42
)

# Step 2: Split remaining 85% into Train (70% total) and Val (15% total)
stratify_temp = y_temp['overall_patient_risk'].astype(str)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=stratify_temp, random_state=42
)

# Step 3: Hard Leakage Assertion
assert len(set(X_train['patient_id']).intersection(set(X_test['patient_id']))) == 0, \
    "FATAL ERROR: Patient Contamination Leakage Detected!"
```

### 3.4 Multimodal Storage Specifications
| Modality | Ingestion Source | Output Schema | Transformation Engine | Target Destination |
| :--- | :--- | :--- | :--- | :--- |
| **Tabular EHR** | Raw Hospital CSV | $(N, 66)$ float32 | `SimpleImputer` + `RobustScaler` | `features/train_features.parquet` |
| **Pathology** | Digital WSI Tiles | $(N, 3, 224, 224)$ | Stain Norm + ImageNet Norm | `data/stage2_dl/spatial/` |
| **Biomarker Series**| Serial Lab Draws | $(N, T, 6)$ float32 | Sequence Padding + Masking | `data/stage2_dl/temporal/` |
| **Audio Voice** | 16kHz WAV Audio | $(N, 80, 3000)$ | Whisper Mel-Spectrogram | `data/stage3_nlp/audio/` |

---

## 4. Automated Quality Assurance & Data Contracts
The Data Engineer enforces strict data contract validation scripts prior to model ingestion:
- **Null Invariance**: Post-imputation datasets must have exactly 0 missing entries.
- **Categorical Schema Enforcement**: New, unseen categories in test or production inferencing are handled via `handle_unknown='ignore'` to avoid crash states.
- **Feature Dimension Assertions**: `assert X_train.shape[1] == 66` ensures engineered feature schema parity.

---

## 5. Viva Voce & Technical Defense (Data Engineer)

#### Q1: Why did you choose Median Imputation over Mean or Mode Imputation for clinical biomarkers?
> **Answer:** "Clinical biomarkers such as ctDNA, serum creatinine, and LDH display severe right-skewness with heavy tails due to acute disease states. The mean is highly sensitive to extreme values, which would artificially pull the imputed baseline toward pathological levels. The median represents a robust non-parametric measure of central tendency that preserves the true median patient profile without introducing bias."

#### Q2: What is the exact difference between data leakage and covariate shift?
> **Answer:** "Data leakage is an engineering mistake where information from the test set or future time steps is improperly used to train the model, artificially inflating training/validation performance. Covariate shift, on the other hand, is a real-world statistical phenomenon where the input distribution $P(X)$ changes over time between training and production environments, while the conditional probability $P(Y \mid X)$ remains unchanged."

#### Q3: How do you handle sequence padding and variable-length clinic visits in the deep learning data pipeline?
> **Answer:** "Different patients have different numbers of visits. We standardize sequences to a maximum time window $T_{\max}$ using zero-padding. Crucially, we generate an accompanying boolean mask tensor $\mathbf{M} \in \{0, 1\}^{N \times T_{\max}}$. In the Transformer and BiLSTM layers, this mask forces the attention and recurrence mechanisms to multiply padded positions by zero, preventing padded dummy values from corrupting hidden states."
