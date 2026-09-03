# Personalized Precision Oncology
# Stage 1 Machine Learning Final Report

## 1. Problem Statement
**Why oncology treatment prediction is needed**: Oncology involves complex, high-risk treatment decisions. Not all patients respond equally to therapies, and some are at severe risk of debilitating toxicity. Predictive modeling allows clinicians to foresee adverse events and anticipate treatment efficacy, ultimately personalizing care and improving survival rates.
**What the ML system predicts**: The Stage 1 Machine Learning system is designed to predict two critical patient outcomes:
1. `toxicity_risk`: The probability that a patient will suffer severe side effects from the prescribed treatment.
2. `therapy_response`: The likelihood that the patient's cancer will positively respond to the treatment regimen.

## 2. Dataset Description
- **Dataset Size**: 5,000 synthetic patient records (developed for academic ML training).
- **Number of Features**: 34 original clinical features including demographics, tumor characteristics, treatments, and biomarkers.
- **Target Variables**:
    - `toxicity_risk` (Classes: Low, Moderate, High)
    - `therapy_response` (Classes: Complete Response, Partial Response, Non-Responder)

## 3. Complete ML Pipeline
The pipeline was architected linearly to ensure reproducibility and data integrity:

Raw Dataset 
↓ 
Data Cleaning 
↓ 
EDA 
↓ 
Feature Engineering 
↓ 
Feature Selection 
↓ 
Train Validation Test Split 
↓ 
Model Training 
↓ 
Hyperparameter Tuning 
↓ 
Evaluation 
↓ 
Explainability 
↓ 
Prediction Pipeline

## 4. Data Processing
- **Missing Value Handling**: Categorical features missing values were imputed with a constant `'unknown'`. Numerical features were imputed using the strictly training-fitted `median` to protect against outliers.
- **Outlier Handling**: Outliers were analyzed during EDA; severe distortions were mitigated using robust scaler paradigms and tree-based modeling which is naturally resilient to outliers.
- **Data Validation**: Exact duplicate rows and leakage-prone features were removed prior to model exposure.

## 5. Exploratory Data Analysis
- **Feature Distributions**: Analyzed numerical feature distributions (e.g., Age, Tumor Size) and categorical proportions.
- **Correlation Analysis**: Generated heatmaps to identify multicollinearity. Redundant highly-correlated variables were identified for subsequent dropping.
- **Class Distribution**: Evaluated the target variables to understand the baseline prevalence of each risk and response class.

## 6. Feature Engineering
- **Created Features**: `age_group`, `bmi_category`, `tumor_size_category`, `treatment_intensity`, `high_clinical_risk`, `biomarker_interaction`.
- **Feature Selection Method**: Applied VarianceThreshold to remove near-zero variance features, followed by a Pearson Correlation filter (> 0.90) to remove redundant variables.
- **Leakage Prevention**: All imputation, scaling (StandardScaler), and encoding (OneHotEncoder) were strictly fit on the `train` split only.

## 7. Model Training
Four core models were robustly tested for both multi-class targets:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## 8. Final Models
**Toxicity Risk:**
- **Final model**: Tuned XGBoost
- **Hyperparameters**: `{'subsample': 1.0, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 1.0}`
- **Why selected**: Yielded the highest Macro F1-score on the unseen test set, properly balancing precision and recall across all severity classes.

**Therapy Response:**
- **Final model**: Tuned XGBoost
- **Hyperparameters**: `{'subsample': 1.0, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 1.0}`
- **Why selected**: XGBoost narrowly outperformed the Logistic Regression baseline after hyperparameter tuning via RandomizedSearchCV, offering the best overall Macro F1-score.

## 9. Evaluation
Using the holdout test dataset, the models were evaluated comprehensively:
- **Accuracy**: Overall correct classification rate.
- **Precision & Recall**: Tracked via Macro-averaging to ensure minority classes were not ignored.
- **Macro F1**: Used as the primary optimization metric.
- **ROC-AUC**: Tracked the models' ability to probabilistically distinguish between classes.
- **Confusion Matrix**: Generated to observe exact misclassifications across clinical boundaries.

## 10. Explainability
- **SHAP (SHapley Additive exPlanations)**: Utilized `TreeExplainer` to break down both global and local feature importance.
- **Feature Importance**: XGBoost split weight and SHAP values identified `age`, `cancer_stage`, and specific physiological readings as top drivers.
- **Clinical Interpretation**: Prevents the model from being a "black box" by allowing oncologists to see exactly which biomarkers are flagging a high toxicity risk for a specific patient.

## 11. Prediction Pipeline
Input patient data 
↓ 
Preprocessing (applies historically fitted scalers/encoders)
↓ 
Model prediction (runs XGBoost inference)
↓ 
Risk output (discrete class prediction)
↓ 
Confidence score (maximum softmax probability)
↓ 
Important factors (dynamically calculated top 3 SHAP drivers)

## 12. Limitations
- **Dataset limitations**: Uses synthetic 5,000-patient dataset; lacks longitudinal time-series data.
- **Need for clinical validation**: Models require rigorous validation against external, real-world cohorts before actual deployment.
- **AI assists doctors, does not replace doctors**: Predictions are meant to augment clinical decision-making, not override it.

## 13. Future Work
Stage 1 Machine Learning lays the groundwork. Future expansion involves **Stage 2 Deep Learning (DL)**:
- **Medical image analysis**: CNNs for radiology/pathology scans.
- **Treatment sequence modelling**: LSTMs/Transformers for longitudinal patient trajectories.
- **Multi-modal AI**: Merging tabular clinical data with imaging and genomic sequences into a unified predictive network.
