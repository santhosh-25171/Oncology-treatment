# Stage 1 ML — Prediction Pipeline Report

## 1. Purpose
The prediction pipeline serves as the central orchestration module. It combines data preprocessing, feature engineering, predictive inference, and model explainability into a single robust endpoint.

## 2. Patient Data Flow
1. **Input**: A JSON dictionary representing raw clinical features.
2. **Validation**: Enforces the presence and correct typing of 34 expected medical fields.
3. **Preprocessing**: Missing values are imputed, categorical variables are one-hot encoded, and numerical features are standardized utilizing the exact statistical boundaries learned during training.
4. **Engineering**: Derived fields like BMI category, Age groups, and Biomarker interaction scores are computed dynamically.
5. **Inference**: High-performance XGBoost models execute the prediction.
6. **Explanation**: SHAP TreeExplainer identifies exactly which features pushed the model's confidence for that specific patient.

## 3. Probability Interpretation
The prediction output includes a structured probability dictionary. This helps clinicians understand if a "High Toxicity Risk" prediction is borderline (e.g., 51% High, 49% Moderate) or highly confident (e.g., 90% High).

## 4. Explainability Integration
By natively integrating SHAP, the pipeline ensures no prediction is a "black box". The top 3 factors driving each prediction are always returned alongside the clinical classification.

## 5. Sample Output
```json
{
    "toxicity_risk": {
        "prediction": "Moderate",
        "confidence": 0.397933691740036,
        "probabilities": {
            "High": 0.21673548221588135,
            "Low": 0.38533082604408264,
            "Moderate": 0.397933691740036
        },
        "important_factors": [
            "renal_function",
            "lymphocyte_count",
            "hemoglobin"
        ]
    },
    "therapy_response": {
        "prediction": "Partial Response",
        "confidence": 0.5799921154975891,
        "probabilities": {
            "Complete Response": 0.30703702569007874,
            "Non-Responder": 0.11297084391117096,
            "Partial Response": 0.5799921154975891
        },
        "important_factors": [
            "cancer_stage_III",
            "cancer_stage_IV",
            "prior_treatment_count"
        ]
    }
}
```

## 6. Clinical Importance & Limitations
While this AI system provides highly accurate historical pattern matching, it **cannot** replace clinical judgment. Factors such as patient preference, unrecorded complex comorbidities, or sudden physiological changes are not captured by the tabular inputs. This tool should only function as a decision support auxiliary.
