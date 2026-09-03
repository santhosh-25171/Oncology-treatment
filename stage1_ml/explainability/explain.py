import pandas as pd
import numpy as np
import json
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(base_dir, "data", "stage1_ml", "features", "processed_features.csv")
    models_dir = os.path.join(base_dir, "data", "stage1_ml", "models")
    out_dir = os.path.join(base_dir, "data", "stage1_ml", "explainability")
    docs_dir = os.path.join(base_dir, "docs")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    print("Loading datasets...")
    df = pd.read_csv(features_path)
    # We use the test split for explainability to see how it behaves on unseen data
    # (though training data can also be used for global explanations)
    test_df = df[df['dataset_split'] == 'test']
    
    targets = ['toxicity_risk', 'therapy_response']
    X_test = test_df.drop(columns=targets + ['dataset_split'])
    # Fix names as in training
    X_test.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in X_test.columns]
    feature_names = X_test.columns.tolist()
    
    # Load best models info
    with open(os.path.join(models_dir, "best_model.json"), "r") as f:
        best_models = json.load(f)
        
    feature_importance_dict = {}
    shap_summaries = {}
    patient_explanations = {}
    
    # Analyze Toxicity Risk (XGBoost)
    target = 'toxicity_risk'
    model_name = best_models[target]["best_model"]
    print(f"Analyzing {target} ({model_name})...")
    
    model_path = os.path.join(models_dir, f"{target}_{model_name.replace(' ', '_').lower()}.joblib")
    model_tox = joblib.load(model_path)
    
    # 1. Feature Importance (XGBoost)
    importance_tox = model_tox.feature_importances_
    tox_ranking = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_tox
    }).sort_values('importance', ascending=False)
    
    feature_importance_dict[target] = tox_ranking.head(20).to_dict(orient='records')
    
    # SHAP for XGBoost
    print("Computing SHAP for toxicity_risk...")
    explainer_tox = shap.TreeExplainer(model_tox)
    # limit to 100 for speed
    X_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
    shap_values_tox = explainer_tox.shap_values(X_sample)
    
    # Handle list or 3D array for multiclass
    if isinstance(shap_values_tox, list):
        mean_shap_tox = np.abs(np.array(shap_values_tox)).mean(axis=(0, 1))
    elif len(shap_values_tox.shape) == 3:
        # shape is (n_samples, n_features, n_classes)
        mean_shap_tox = np.abs(shap_values_tox).mean(axis=(0, 2))
    else:
        mean_shap_tox = np.abs(shap_values_tox).mean(axis=0)
        
    shap_summaries[target] = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_shap_tox
    }).sort_values('mean_abs_shap', ascending=False).head(20).to_dict(orient='records')
    
    # Individual Patient
    pat_idx = 0
    pat_data = X_sample.iloc[pat_idx:pat_idx+1]
    pat_shap = explainer_tox.shap_values(pat_data)
    
    if isinstance(pat_shap, list):
        pat_shap_impact = pat_shap[0][0]
    elif len(pat_shap.shape) == 3:
        pat_shap_impact = pat_shap[0, :, 0] # feature impact for class 0
    else:
        pat_shap_impact = pat_shap[0]
        
    patient_explanations[target] = {
        'patient_index': int(pat_data.index[0]),
        'top_features_pushing_prediction': pd.DataFrame({
            'feature': feature_names,
            'shap_value': pat_shap_impact,
            'feature_value': pat_data.iloc[0].values
        }).sort_values('shap_value', key=abs, ascending=False).head(5).to_dict(orient='records')
    }

    # Analyze Therapy Response (Logistic Regression)
    target = 'therapy_response'
    model_name = best_models[target]["best_model"]
    print(f"Analyzing {target} ({model_name})...")
    
    model_path = os.path.join(models_dir, f"{target}_{model_name.replace(' ', '_').lower()}.joblib")
    model_ther = joblib.load(model_path)
    
    # 1. Feature Importance (Logistic Regression)
    # LogReg coef_ shape is (n_classes, n_features) for multiclass
    coefs = np.abs(model_ther.coef_).mean(axis=0)
    ther_ranking = pd.DataFrame({
        'feature': feature_names,
        'importance': coefs
    }).sort_values('importance', ascending=False)
    
    feature_importance_dict[target] = ther_ranking.head(20).to_dict(orient='records')
    
    # SHAP for LogReg
    print("Computing SHAP for therapy_response...")
    # LinearExplainer works well with standard scaler features
    masker = shap.maskers.Independent(data=X_test)
    explainer_ther = shap.LinearExplainer(model_ther, masker)
    shap_values_ther = explainer_ther.shap_values(X_sample)
    
    if isinstance(shap_values_ther, list):
        mean_shap_ther = np.abs(np.array(shap_values_ther)).mean(axis=(0, 1))
    elif len(shap_values_ther.shape) == 3:
        mean_shap_ther = np.abs(shap_values_ther).mean(axis=(0, 2))
    else:
        mean_shap_ther = np.abs(shap_values_ther).mean(axis=0)
        
    shap_summaries[target] = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_shap_ther
    }).sort_values('mean_abs_shap', ascending=False).head(20).to_dict(orient='records')
    
    pat_shap_ther = explainer_ther.shap_values(pat_data)
    if isinstance(pat_shap_ther, list):
        pat_shap_impact_ther = pat_shap_ther[0][0]
    elif len(pat_shap_ther.shape) == 3:
        pat_shap_impact_ther = pat_shap_ther[0, :, 0]
    else:
        pat_shap_impact_ther = pat_shap_ther[0]
        
    patient_explanations[target] = {
        'patient_index': int(pat_data.index[0]),
        'top_features_pushing_prediction': pd.DataFrame({
            'feature': feature_names,
            'shap_value': pat_shap_impact_ther,
            'feature_value': pat_data.iloc[0].values
        }).sort_values('shap_value', key=abs, ascending=False).head(5).to_dict(orient='records')
    }
    
    # Biomarker Leaderboard
    print("Generating Biomarker Leaderboard...")
    biomarker_keywords = ['biomarker', 'mutation', 'ctdna', 'genetic', 'tumor', 'lymph']
    leaderboard = []
    
    for t, ranking in feature_importance_dict.items():
        for feat in ranking:
            if any(k in feat['feature'].lower() for k in biomarker_keywords):
                leaderboard.append({
                    'target': t,
                    'biomarker': feat['feature'],
                    'importance_score': feat['importance']
                })
                
    leaderboard = sorted(leaderboard, key=lambda x: x['importance_score'], reverse=True)
    
    # Save JSON files
    with open(os.path.join(out_dir, "feature_importance.json"), "w") as f:
        json.dump(feature_importance_dict, f, indent=4)
        
    with open(os.path.join(out_dir, "shap_values_summary.json"), "w") as f:
        json.dump(shap_summaries, f, indent=4)
        
    with open(os.path.join(out_dir, "biomarker_leaderboard.json"), "w") as f:
        json.dump(leaderboard, f, indent=4)
        
    with open(os.path.join(out_dir, "patient_explanation.json"), "w") as f:
        json.dump(patient_explanations, f, indent=4)
        
    # Generate Plots
    print("Generating plots...")
    
    # Plot feature importance (bar plot for Toxicity)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tox_ranking.head(10), x='importance', y='feature', palette='viridis')
    plt.title("Top 10 Feature Importances (Toxicity Risk - XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance_plot.png"))
    plt.close()
    
    # Plot SHAP summary (Therapy response)
    # we just use a generic bar plot of SHAP importances since plotting raw SHAP summary for multiclass can throw plotting errors in script
    plt.figure(figsize=(10, 6))
    shap_ther_df = pd.DataFrame(shap_summaries['therapy_response']).head(10)
    sns.barplot(data=shap_ther_df, x='mean_abs_shap', y='feature', palette='magma')
    plt.title("SHAP Global Summary (Therapy Response - Logistic Regression)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"))
    plt.close()

    # Markdown Report
    md_content = f"""# Stage 1 ML — Explainability Report

## 1. Why Explainability is Important in Medical AI
In oncology, AI acts as a decision-support tool. Clinicians cannot blindly trust "black box" models when patient lives are at stake. Explainability validates that the model is making predictions based on clinically sound logic (e.g., higher tumor sizes and specific mutations driving risk) rather than spurious statistical artifacts.

## 2. How Feature Importance Works
Feature importance measures how frequently a feature is used to split the data across all trees (in XGBoost) or the magnitude of the learned weight (in Logistic Regression). High importance indicates that the model relies heavily on that feature to differentiate classes.

## 3. How SHAP Explains Predictions
SHAP (SHapley Additive exPlanations) is based on cooperative game theory. It breaks down a prediction to show the marginal contribution of every single feature. While global feature importance tells us what matters overall, SHAP tells us exactly *why* a specific prediction was made for an individual patient.

## 4. Important Clinical Features Discovered
Our analysis revealed critical drivers for both targets.

**Top Features for Toxicity Risk (XGBoost):**
"""
    for item in feature_importance_dict['toxicity_risk'][:5]:
        md_content += f"- `{item['feature']}` (Score: {item['importance']:.4f})\n"
        
    md_content += "\n**Top Features for Therapy Response (Logistic Regression):**\n"
    for item in feature_importance_dict['therapy_response'][:5]:
        md_content += f"- `{item['feature']}` (Score: {item['importance']:.4f})\n"

    md_content += """
## 5. Biomarker Leaderboard
The most influential biological and molecular features driving predictions across both models are:
"""
    for item in leaderboard[:5]:
        md_content += f"- **{item['biomarker']}** -> Predicts: {item['target']} (Score: {item['importance_score']:.4f})\n"

    md_content += """
## 6. Individual Patient Explanation Example
Consider Patient ID #{pat_idx} from our holdout dataset.

**Toxicity Risk Prediction Drivers**:
"""
    for item in patient_explanations['toxicity_risk']['top_features_pushing_prediction']:
        md_content += f"- `{item['feature']}` (Value: {item['feature_value']:.2f}) pushed the model's confidence by SHAP value: {item['shap_value']:.4f}\n"

    md_content += "\n**Therapy Response Prediction Drivers**:\n"
    for item in patient_explanations['therapy_response']['top_features_pushing_prediction']:
        md_content += f"- `{item['feature']}` (Value: {item['feature_value']:.2f}) pushed the model's confidence by SHAP value: {item['shap_value']:.4f}\n"

    with open(os.path.join(docs_dir, "stage1_ml_explainability_report.md"), "w") as f:
        f.write(md_content)

    print("\n------------------------------")
    print("EXPLAINABILITY SUMMARY:")
    print("Models analyzed:")
    print("  - toxicity_risk: XGBoost")
    print("  - therapy_response: Logistic Regression")
    print("Top features overall:")
    print(f"  - Toxicity: {feature_importance_dict['toxicity_risk'][0]['feature']}")
    print(f"  - Therapy: {feature_importance_dict['therapy_response'][0]['feature']}")
    print("Generated files:")
    print(f"  - {os.path.join(out_dir, 'feature_importance.json')}")
    print(f"  - {os.path.join(out_dir, 'shap_values_summary.json')}")
    print(f"  - {os.path.join(out_dir, 'biomarker_leaderboard.json')}")
    print(f"  - {os.path.join(out_dir, 'patient_explanation.json')}")
    print(f"  - {os.path.join(out_dir, 'shap_summary.png')}")
    print(f"  - {os.path.join(out_dir, 'feature_importance_plot.png')}")
    print(f"  - {os.path.join(docs_dir, 'stage1_ml_explainability_report.md')}")
    print("Any errors: None")
    print("------------------------------")
    print("Explainability completed successfully. Stopping.")

if __name__ == "__main__":
    main()
