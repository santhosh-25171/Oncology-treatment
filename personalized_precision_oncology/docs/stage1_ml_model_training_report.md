# Stage 1 ML — Model Training Report

## 1. Objective
The goal of this stage is to train and evaluate multiple machine learning classification algorithms to predict two key oncology outcomes: `toxicity_risk` and `therapy_response`. By comparing different algorithms, we can select the most performant model for integration into a clinical decision-support tool.

## 2. Models Selected
To ensure robust evaluation, we trained a diverse set of classifiers:
- **Logistic Regression (Baseline)**: Selected for its high interpretability and efficiency, establishing a performance baseline.
- **Random Forest**: Selected for its ability to capture non-linear interactions without severe overfitting, using an ensemble of decision trees.
- **XGBoost**: Selected for its high predictive accuracy and gradient boosting capabilities, commonly yielding top performance on tabular medical data.
- **LightGBM**: Selected for its highly efficient gradient boosting implementation, often handling complex, sparse features rapidly.

## 3. Training Process
- **Data Split**: Models were trained on the training split (70%) and evaluated on the holdout test split (15%) defined during feature engineering.
- **Preprocessing**: All predictors were standardized and encoded appropriately in the prior stage. Target variables were label-encoded into integers prior to model fitting to satisfy tree-based libraries (XGBoost/LightGBM).
- **Predictions**: For evaluation, models output both class predictions (for precision, recall, F1, accuracy) and probability predictions (for ROC-AUC multiclass).

## 4. Metrics Comparison
### Target: toxicity_risk
- **Logistic Regression** | F1 (Macro): 0.3459 | Accuracy: 0.4427 | ROC-AUC: 0.5890
- **Random Forest** | F1 (Macro): 0.3140 | Accuracy: 0.4160 | ROC-AUC: 0.5459
- **XGBoost** | F1 (Macro): 0.3462 | Accuracy: 0.4227 | ROC-AUC: 0.5499
- **LightGBM** | F1 (Macro): 0.3434 | Accuracy: 0.4173 | ROC-AUC: 0.5628

### Target: therapy_response
- **Logistic Regression** | F1 (Macro): 0.3236 | Accuracy: 0.4587 | ROC-AUC: 0.5453
- **Random Forest** | F1 (Macro): 0.3090 | Accuracy: 0.4520 | ROC-AUC: 0.5219
- **XGBoost** | F1 (Macro): 0.3153 | Accuracy: 0.4253 | ROC-AUC: 0.5189
- **LightGBM** | F1 (Macro): 0.3131 | Accuracy: 0.4160 | ROC-AUC: 0.5052

## 5. Best Model Selection Criteria
The best model for each target was selected based on the **Macro F1-Score**. In medical datasets where moderate class imbalance is present, Macro F1 prevents the model from achieving a high score merely by over-predicting the majority class. It ensures that minority but critical classes (e.g., 'Complete Response' or 'Critical Toxicity') are predicted accurately.

- **toxicity_risk**: The best model is **XGBoost** with an F1-Score of 0.3462.
- **therapy_response**: The best model is **Logistic Regression** with an F1-Score of 0.3236.

## 6. Medical Interpretation of Results
The model's predictions provide a probabilistic assessment based on historically observed patient traits. 
- **Therapy Response Prediction**: Helps oncologists stratify patients into likely responders vs. non-responders, potentially avoiding aggressive but futile treatments.
- **Toxicity Risk Prediction**: Identifies patients highly vulnerable to severe side-effects, allowing care teams to preemptively adjust dosages or increase monitoring frequency.

*Note: These models function strictly as decision-support systems. Final clinical decisions must incorporate holistic patient context uncaptured by the tabular variables alone.*
