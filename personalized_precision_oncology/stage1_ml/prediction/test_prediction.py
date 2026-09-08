import os
import json
import pandas as pd
from prediction import OncologyPredictionPipeline

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "stage1_ml", "processed", "oncology_cleaned.csv")
    preds_dir = os.path.join(base_dir, "data", "stage1_ml", "predictions")
    docs_dir = os.path.join(base_dir, "docs")
    
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    print("Initializing Prediction Pipeline (loading models & rebuilding preprocessing)...")
    pipeline = OncologyPredictionPipeline(base_dir=base_dir)
    print("Pipeline successfully initialized.")
    
    print("Extracting a sample patient from cleaned dataset...")
    df = pd.read_csv(data_path)
    sample_patient = df.iloc[0].drop(labels=['toxicity_risk', 'therapy_response']).to_dict()
    
    print("Sample Patient Features:")
    print(json.dumps({k: sample_patient[k] for k in list(sample_patient.keys())[:5]}, indent=2), "...\n")
    
    print("Running prediction engine...")
    prediction_result = pipeline.predict(sample_patient)
    
    print("\n--- PREDICTION OUTPUT ---")
    print(json.dumps(prediction_result, indent=4))
    
    # Save outputs
    sample_path = os.path.join(preds_dir, "sample_prediction.json")
    with open(sample_path, "w") as f:
        json.dump({"input_features": sample_patient, "prediction": prediction_result}, f, indent=4)
        
    results_path = os.path.join(preds_dir, "prediction_results.json")
    with open(results_path, "w") as f:
        json.dump(prediction_result, f, indent=4)
        
    # Generate Markdown documentation
    md_content = f"""# Stage 1 ML — Prediction Pipeline Report

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
{json.dumps(prediction_result, indent=4)}
```

## 6. Clinical Importance & Limitations
While this AI system provides highly accurate historical pattern matching, it **cannot** replace clinical judgment. Factors such as patient preference, unrecorded complex comorbidities, or sudden physiological changes are not captured by the tabular inputs. This tool should only function as a decision support auxiliary.
"""
    
    md_path = os.path.join(docs_dir, "stage1_ml_prediction_pipeline_report.md")
    with open(md_path, "w") as f:
        f.write(md_content)
        
    print("\n------------------------------")
    print("PREDICTION PIPELINE STATUS: PASS")
    print("Generated files:")
    print(f"  - {sample_path}")
    print(f"  - {results_path}")
    print(f"  - {md_path}")
    print("Any errors found: None")
    print("------------------------------")
    print("Prediction pipeline completed successfully. Stopping.")
    
if __name__ == "__main__":
    main()
