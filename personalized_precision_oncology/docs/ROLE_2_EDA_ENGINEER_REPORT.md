# ROLE REPORT 2: EXPLORATORY DATA ANALYSIS (EDA)
## Personalized Precision Medicine for Oncology Treatment Optimization

```
======================================================================================================
ROLE:               Exploratory Data Analysis (EDA) Engineer
PROJECT:            Personalized Precision Medicine for Oncology Treatment Optimization
PIPELINE STAGES:    Statistical Profiling | Normality Testing | Multicollinearity Audits | 
                    Outlier Boundary Detection | Target Balance Diagnostics | Feature Discovery Memos
KEY FINDINGS:       Heavy Right-Skew (ctDNA γ1 > 2.4) | Multicollinearity (VIF > 8.5 in BP metrics) | 
                    Minority Non-Responders (18.3%) | 6 Actionable Feature Recommendations
======================================================================================================
```

---

## 1. Executive Mission & Role Definition
The **EDA Engineer** acts as the diagnostic scientist of the machine learning lifecycle. Before any model is trained or hyperparameter is tuned, the EDA Engineer inspects data distributions, calculates statistical properties, uncovers hidden correlations, evaluates class imbalance, and identifies biophysical boundaries.

The core objective is to ensure that all downstream modeling decisions (such as model families, scaling methods, loss functions, and probability thresholds) are guided by empirical data evidence rather than arbitrary trial and error.

---

## 2. Theoretical Concepts & Foundational Principles

### 2.1 Distributional Profiling & Moments
Every feature is characterized by four statistical moments that dictate how preprocessing and algorithms should behave:
1. **First Moment (Mean / Central Tendency $\mu$)**:
   $$\mu = \frac{1}{N}\sum_{i=1}^N x_i$$
2. **Second Moment (Variance / Dispersion $\sigma^2$)**:
   $$\sigma^2 = \frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2$$
3. **Third Moment (Skewness $\gamma_1$)**: Evaluates asymmetry around the mean:
   $$\gamma_1 = \frac{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^3}{\left(\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2\right)^{3/2}}$$
   - $\gamma_1 = 0$: Symmetric Gaussian distribution.
   - $\gamma_1 > +1.0$: Highly right-skewed (concentrated around zero with rare, massive spikes).
4. **Fourth Moment (Kurtosis $\beta_2$)**: Evaluates the propensity for extreme outliers (tail heaviness):
   $$\beta_2 = \frac{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^4}{\left(\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2\right)^2}$$
   - A standard normal distribution has $\beta_2 = 3.0$ (excess kurtosis $= 0$). A $\beta_2 > 5.0$ indicates leptokurtic behavior with fat tails.

### 2.2 Multicollinearity & Variance Inflation Factor (VIF)
Multicollinearity occurs when two or more predictor features are highly linearly correlated. While tree ensembles are somewhat robust to collinearity, it causes significant problems:
- Feature importance splits become randomly divided across collinear features (split dilution).
- Linear and logistic baselines exhibit erratic, unstable weights with inverted signs.
- **Variance Inflation Factor**: Measures how much the variance of an estimated regression coefficient increases due to collinearity:
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
  Where $R_j^2$ is the coefficient of determination when feature $x_j$ is regressed on all other predictors.
  - $\text{VIF} = 1$: Completely independent.
  - $\text{VIF} > 5.0$: Moderate collinearity requiring investigation.
  - $\text{VIF} > 10.0$: Severe collinearity requiring feature elimination or dimensionality reduction.

### 2.3 Outlier Detection Theory in Clinical Medicine
In traditional data science, outliers are frequently stripped using automated filters. In clinical oncology, this is dangerous:
- **Tukey's IQR Rule**:
  $$\text{IQR} = Q_3 - Q_1, \quad [\text{Lower}, \text{Upper}] = [Q_1 - 1.5 \cdot \text{IQR}, \; Q_3 + 1.5 \cdot \text{IQR}]$$
- **The Clinical Distinction**: An extreme value can be either:
  1. *Measurement Artifact*: Negative age, diastolic blood pressure $> 250\, \text{mmHg}$, or zero platelets due to clotted sample $\implies$ **DROP OR SANITIZE**.
  2. *True Acute Pathology*: ctDNA spike 50x above median or white blood cell count indicative of leukostasis $\implies$ **RETAIN & CREATE RISK INDICATOR FLAG**.

---

## 3. Concrete EDA Findings on Oncology Dataset ($N=5,000$)

### 3.1 Numerical Feature Distribution Summary
| Feature Name | Mean | Median | Std Dev | Skewness ($\gamma_1$) | Kurtosis ($\beta_2$) | Diagnostic Action Taken |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Age** | 56.80 | 56.80 | 12.65 | -0.04 | 2.12 | Normal distribution; bin into `<50, 50-65, >65` |
| **BMI** | 26.42 | 26.10 | 4.82 | +0.48 | 3.15 | Categorize into WHO categories (underweight/obese) |
| **Tumor Size (cm)** | 3.84 | 3.60 | 1.82 | +0.65 | 3.42 | Discretize into TNM T-stages (`T1, T2, T3`) |
| **Treatment Dose** | 75.67 | 75.33 | 18.01 | +0.12 | 2.85 | Standard continuous parameter |
| **Duration (wks)** | 15.93 | 15.90 | 6.77 | +0.08 | 2.90 | Calculate treatment intensity: $\frac{\text{Dose}}{\text{Duration}}$ |
| **Renal (eGFR)** | 69.62 | 69.20 | 18.41 | -0.22 | 2.78 | Key toxicity driver; flag critical threshold $<30$ |
| **ctDNA Level** | 14.82 | 4.10 | 28.60 | **+3.14** | **14.80** | Extreme right-skew; apply $\log(1+x)$ transform |

### 3.2 Target Variable Class Balance Analysis
```text
1. Primary Target: overall_patient_risk
   ├── Low Risk:       2,000 (40.0%)
   ├── High Risk:      1,865 (37.3%)
   └── Moderate Risk:  1,135 (22.7%)
   Ratio: 1.76 : 1.64 : 1.00 (Mild Imbalance)

2. Secondary Target: toxicity_risk
   ├── Low Toxicity:   2,025 (40.5%)
   ├── High Toxicity:  1,954 (39.1%)
   └── Moderate:       1,021 (20.4%)
   Ratio: 1.98 : 1.91 : 1.00 (Moderate Imbalance)

3. Secondary Target: therapy_response
   ├── Partial Response:   2,306 (46.1%)
   ├── Complete Response:  1,777 (35.5%)
   └── Non-Responder:        917 (18.3%)  <-- SEVERE MINORITY CLASS
   Ratio: 2.51 : 1.94 : 1.00 (Critical Class Imbalance!)
```

### 3.3 Collinearity Analysis & VIF Findings
- Bivariate correlation revealed high collinearity between `systolic_bp` and `mean_arterial_pressure` ($r = 0.91$, $\text{VIF} = 9.8$).
- High collinearity was observed between `treatment_dose` and `cumulative_exposure` ($r = 0.86$, $\text{VIF} = 7.9$).
- *Feedback to ML Team*: Drop redundant raw metrics in favor of composite engineered metrics (`treatment_intensity` = `dose` / `duration`).

---

## 4. Actionable Feedback Provided to Downstream Roles
The EDA Engineer generated 6 formal recommendations that directly shaped the ML pipeline:
1. **Logarithmic Transformation**: Mandated $\log(1 + x)$ transformations on `ctDNA` and `biomarker_interaction` to stabilize tree split thresholds and linear weight learning.
2. **Clinical Bins**: Mandated creation of discrete oncology cohorts (`age_group`, `bmi_category`, `tumor_size_category`).
3. **Biomarker Interaction Feature**: Observed that when both `biomarker_1` and `biomarker_2` are elevated simultaneously, mortality increases multiplicatively rather than additively $\implies$ engineered `biomarker_interaction = biomarker_1 * biomarker_2`.
4. **Clinical Risk Index**: Identified that patients with both high comorbidity ($> \text{median}$) and poor performance status ($> \text{median}$) have an $82\%$ probability of adverse outcomes $\implies$ created `high_clinical_risk` boolean flag.
5. **Class Imbalance Intervention**: Flagged the $18.3\%$ minority non-responder class; mandated class-weighted loss (`class_weight='balanced'`) and Platt-scaled threshold tuning.
6. **Robust Scaling**: Recommended `RobustScaler` over `StandardScaler` to prevent extreme outliers in inflammatory markers from compressing non-outlier features.

---

## 5. Viva Voce & Technical Defense (EDA Engineer)

#### Q1: How do you mathematically differentiate between an outlier and a distinct sub-population cluster?
> **Answer:** "An isolated outlier is typically an observation falling more than $3\sigma$ or $1.5 \times \text{IQR}$ away from the distribution mode with very low local density. A sub-population cluster, by contrast, exhibits multi-modal distribution characteristics where Gaussian Mixture Models (GMM) or kernel density estimation (KDE) reveal secondary density peaks with consistent multivariate relationships across multiple features (e.g., elderly patients with comorbid diabetes and hypertension forming a distinct high-risk cluster)."

#### Q2: Why is Spearman rank correlation often preferred over Pearson correlation for clinical lab biomarkers?
> **Answer:** "Pearson correlation measures strictly linear relationships and assumes bivariate normality. Clinical biomarkers often follow power-law or exponential kinetics where disease progression causes non-linear biomarker elevation. Spearman rank correlation evaluates monotonic relationships by computing Pearson's coefficient on the ranked values of the variables, making it non-parametric and invariant to monotonic non-linear distortions."

#### Q3: What happens to a decision tree when two features exhibit a correlation of 0.95?
> **Answer:** "When two features are collinear ($r = 0.95$), both features carry nearly identical split information. At any given tree split, the algorithm selects one feature with a marginally higher gain. In subsequent trees or runs, slight data perturbations cause the alternative feature to be picked instead. This dilutes the computed feature importance scores (e.g., Gini importance or gain) across both variables, misleading clinicians about which biomarker is the true causal driver."
