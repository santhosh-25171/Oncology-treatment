# Stage 1 ML Model Effectiveness & Calibration Report

## 1. Executive Summary
This report provides an empirical evaluation of the Stage 1 Machine Learning pipeline for **Oncology Treatment — Patient Risk Prediction System**. 

The goal of Stage 1 ML is to provide clinical decision support by predicting:
1. **Overall Patient Risk** (`High`, `Moderate`, `Low`) — Primary Target uniting therapy non-response and severe adverse drug reaction risk.
2. **Toxicity Risk** (`High`, `Moderate`, `Low`) — Secondary Target predicting treatment adverse event risk.
3. **Therapy Response** (`Complete Response`, `Partial Response`, `Non-Responder`) — Secondary Target predicting clinical outcome.

---

## 2. Dataset Baseline & Split
- **Total Patient Dataset**: 5,000 raw patient records -> 3,750 clean processed feature matrix ($75\%$).
- **Data Split**: 2,625 Train (70%), 375 Validation (10%), 750 Unseen Holdout Test (20%).
- **Feature Space**: 66 engineered features (16 numerical, 50 one-hot encoded categorical). Zero test-set leakage.

---

## 3. 5-Model Benchmarking Results (Unseen Holdout Set, $n=750$)

### Primary Target: `overall_patient_risk`
| Model | High-Risk Recall | Macro F1 | Accuracy | Brier Score (Calibration) | ROC-AUC | Optimization / Decision Strategy | Champion Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **XGBoost (Calibrated + Opt Threshold)** | **78.25%** | **0.3359** | **48.80%** | **0.1978** | **0.5636** | Platt Scaling ($t = 0.48$) | CHAMPION |
| **Random Forest** | 62.07% | 0.3650 | 46.40% | 0.1994 | 0.5434 | Class-Balanced Weighting | Runner-Up |
| **LightGBM** | 60.21% | 0.3664 | 44.93% | 0.2141 | 0.5295 | Gradient Boosting | Benchmark |
| **CatBoost** | 54.91% | 0.3606 | 42.00% | 0.2193 | 0.5456 | Ordered Boosting | Benchmark |
| **Logistic Regression** | 49.87% | 0.3678 | 40.80% | 0.2172 | 0.5746 | Linear Baseline | Linear Baseline |

### Secondary Target: `toxicity_risk`
| Model | High-Risk Recall | Macro F1 | Accuracy | Brier Score | Optimization / Decision Strategy | Champion Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **CatBoost** | **53.31%** | **0.4124** | **42.53%** | **0.2089** | Platt Scaling ($L_2=1$, Depth=6) | CHAMPION |
| **Logistic Regression** | 48.01% | 0.3923 | 40.00% | 0.2168 | $L_2$ Regularized ($C=0.01$) | Linear Baseline |
| **Random Forest** | 54.97% | 0.3776 | 40.40% | 0.2145 | Class-Balanced Trees | Benchmark |
| **LightGBM** | 46.03% | 0.3772 | 38.93% | 0.2254 | Leaf-wise Tree Growth | Benchmark |
| **XGBoost** | 53.97% | 0.3584 | 44.13% | 0.2385 | Subsampled Boosted Trees | Benchmark |

### Secondary Target: `therapy_response`
| Model | High-Risk Recall | Macro F1 | Accuracy | Brier Score | Optimization / Decision Strategy | Champion Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Random Forest** | **51.69%** | **0.3832** | **42.80%** | **0.2056** | Platt Scaling ($N_{\text{est}}=100$) | CHAMPION |
| **CatBoost** | 47.19% | 0.3758 | 40.27% | 0.2395 | Ordered Categorical Boosting | Runner-Up |
| **LightGBM** | 40.82% | 0.3477 | 37.07% | 0.2340 | Balanced Leaf-wise Boosting | Benchmark |
| **Logistic Regression** | 46.44% | 0.3183 | 33.07% | 0.2251 | $L_2$ Regularized | Linear Baseline |
| **XGBoost** | 25.84% | 0.2947 | 43.60% | 0.2427 | Subsampled Trees | Benchmark |

---

## 4. Probability Calibration & Decision Threshold Tuning
- **Calibration Method**: Platt Scaling (Sigmoidal Logistic Calibration) fitted via 5-fold cross-validation (`CalibratedClassifierCV`).
- **Brier Score Improvement**: Reduced probability Brier loss down to **0.1978** per-class mean for Calibrated XGBoost.
- **Decision Threshold Optimization**:
  - High-Risk Recall Threshold: **0.48** for `overall_patient_risk`, yielding **78.25% High-Risk Recall** on unseen holdout.
  - The threshold of $t = 0.48$ **prioritizes high-risk sensitivity in this research prototype**, ensuring that potentially high-risk oncology cases receive proper clinical attention.

---

## 5. SHAP Biomarker Explainability Drivers
Global SHAP feature importance analysis identified key clinical drivers:
1. `comorbidity_score` (Charlson Comorbidity Index)
2. `ctDNA_level` (Circulating Tumor DNA Level)
3. `performance_status` (ECOG Performance Score)
4. `biomarker_interaction` (`biomarker_1` * `biomarker_2`)
5. `treatment_intensity` (`treatment_dose` / `treatment_duration`)

---

## 6. Deployment & System Integration
- **Model Pipeline**: `OncologyPredictionPipeline` in [`stage1_ml/prediction/prediction.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage1_ml/prediction/prediction.py).
- **Automated Validation**: Automated test suites in [`stage1_ml/evaluation/final_validation.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage1_ml/evaluation/final_validation.py) and [`test_risk_decision.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage1_ml/evaluation/test_risk_decision.py) pass 100% of assertion checks.
- **FastAPI Endpoints**: `/predict` and `/health` endpoints in [`integration/api/main.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/integration/api/main.py).
- **Streamlit Dashboard**: Enhanced interactive UI in [`integration/dashboard/app.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/integration/dashboard/app.py).
