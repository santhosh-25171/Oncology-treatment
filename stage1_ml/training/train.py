import pandas as pd
import numpy as np
import json
import os
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def evaluate_model(model, X_test, y_test, y_test_encoded, label_classes):
    """Generate metrics for the model on the test set."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = None
        
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_macro": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics

def train_target_models(target_name, X_train, y_train, X_test, y_test, models_dir):
    """Train models for a specific target variable."""
    print(f"\nTraining models for {target_name}...")
    
    # Encode targets to integers for XGBoost, LightGBM and unified evaluation
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Save label encoder
    joblib.dump(le, os.path.join(models_dir, f"{target_name}_label_encoder.joblib"))
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
    }
    
    target_results = {}
    best_model_name = None
    best_f1 = -1
    
    for model_name, model in models.items():
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train_encoded)
        training_time = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, y_test_encoded, le.classes_)
        
        target_results[model_name] = {
            "metrics": metrics,
            "training_time_seconds": float(training_time),
            "parameters": str(model.get_params())
        }
        
        # Save model
        model_filename = f"{target_name}_{model_name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model, os.path.join(models_dir, model_filename))
        
        # Check best model
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            best_model_name = model_name
            
    return target_results, best_model_name, le.classes_.tolist()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "stage1_ml", "features", "processed_features.csv")
    models_dir = os.path.join(base_dir, "data", "stage1_ml", "models")
    docs_dir = os.path.join(base_dir, "docs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    print(f"Loading processed features from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Split into train/test (we will use test for evaluation)
    train_df = df[df['dataset_split'] == 'train']
    test_df = df[df['dataset_split'] == 'test']
    
    targets = ['toxicity_risk', 'therapy_response']
    
    X_train = train_df.drop(columns=targets + ['dataset_split'])
    # Fix feature names for XGBoost/LightGBM which cannot handle [, ], or <
    X_train.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in X_train.columns]
    y_train_full = train_df[targets]
    
    X_test = test_df.drop(columns=targets + ['dataset_split'])
    X_test.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in X_test.columns]
    y_test_full = test_df[targets]
    
    all_results = {}
    best_models = {}
    
    for target in targets:
        target_results, best_model_name, classes = train_target_models(
            target, X_train, y_train_full[target], X_test, y_test_full[target], models_dir
        )
        
        all_results[target] = target_results
        best_models[target] = {
            "best_model": best_model_name,
            "best_f1_macro": target_results[best_model_name]["metrics"]["f1_macro"],
            "classes": classes
        }
        
    # Save reports
    print("\nSaving tracking and comparison reports...")
    
    with open(os.path.join(models_dir, "model_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=4)
        
    with open(os.path.join(models_dir, "best_model.json"), "w") as f:
        json.dump(best_models, f, indent=4)
        
    # Simplify structure for training report
    training_report = []
    for target, models in all_results.items():
        for model_name, data in models.items():
            training_report.append({
                "target": target,
                "model_name": model_name,
                "parameters": data["parameters"],
                "metrics": data["metrics"],
                "training_time": data["training_time_seconds"]
            })
            
    with open(os.path.join(models_dir, "training_report.json"), "w") as f:
        json.dump(training_report, f, indent=4)
        
    # Generate Markdown documentation
    md_content = f"""# Stage 1 ML — Model Training Report

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
"""
    for target in targets:
        md_content += f"### Target: {target}\n"
        for model_name, data in all_results[target].items():
            m = data['metrics']
            md_content += f"- **{model_name}** | F1 (Macro): {m['f1_macro']:.4f} | Accuracy: {m['accuracy']:.4f} | ROC-AUC: {m['roc_auc_macro']:.4f}\n"
        md_content += "\n"

    md_content += "## 5. Best Model Selection Criteria\n"
    md_content += "The best model for each target was selected based on the **Macro F1-Score**. In medical datasets where moderate class imbalance is present, Macro F1 prevents the model from achieving a high score merely by over-predicting the majority class. It ensures that minority but critical classes (e.g., 'Complete Response' or 'Critical Toxicity') are predicted accurately.\n\n"
    
    for target, b in best_models.items():
        md_content += f"- **{target}**: The best model is **{b['best_model']}** with an F1-Score of {b['best_f1_macro']:.4f}.\n"

    md_content += """
## 6. Medical Interpretation of Results
The model's predictions provide a probabilistic assessment based on historically observed patient traits. 
- **Therapy Response Prediction**: Helps oncologists stratify patients into likely responders vs. non-responders, potentially avoiding aggressive but futile treatments.
- **Toxicity Risk Prediction**: Identifies patients highly vulnerable to severe side-effects, allowing care teams to preemptively adjust dosages or increase monitoring frequency.

*Note: These models function strictly as decision-support systems. Final clinical decisions must incorporate holistic patient context uncaptured by the tabular variables alone.*
"""

    md_report_path = os.path.join(docs_dir, "stage1_ml_model_training_report.md")
    with open(md_report_path, "w") as f:
        f.write(md_content)

    print("\n------------------------------")
    print("MODEL TRAINING SUMMARY:")
    for target, b in best_models.items():
        print(f"Target: {target} | Best Model: {b['best_model']} (F1: {b['best_f1_macro']:.4f})")
    print("\nOutput file locations:")
    print(f"  - {models_dir}/")
    print(f"  - {os.path.join(models_dir, 'model_comparison.json')}")
    print(f"  - {os.path.join(models_dir, 'best_model.json')}")
    print(f"  - {os.path.join(models_dir, 'training_report.json')}")
    print(f"  - {md_report_path}")
    print("------------------------------")
    print("Model training completed successfully. Stopping.")

if __name__ == "__main__":
    main()
