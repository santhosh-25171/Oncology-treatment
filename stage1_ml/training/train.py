import pandas as pd
import numpy as np
import json
import os
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    brier_score_loss, classification_report
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold

def compute_multiclass_brier(y_true_encoded, y_proba):
    """Compute Brier Score for multi-class predictions using One-vs-Rest strategy."""
    n_classes = y_proba.shape[1]
    y_true_onehot = label_binarize(y_true_encoded, classes=list(range(n_classes)))
    brier_scores = []
    for c in range(n_classes):
        brier_scores.append(brier_score_loss(y_true_onehot[:, c], y_proba[:, c]))
    return float(np.mean(brier_scores))

def compute_multiclass_pr_auc(y_true_encoded, y_proba):
    """Compute Macro Area Under Precision-Recall Curve (PR-AUC)."""
    n_classes = y_proba.shape[1]
    y_true_onehot = label_binarize(y_true_encoded, classes=list(range(n_classes)))
    pr_aucs = []
    for c in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, c], y_proba[:, c])
        pr_aucs.append(auc(recall, precision))
    return float(np.mean(pr_aucs))

def evaluate_comprehensive_metrics(model, X_eval, y_eval_encoded, label_classes):
    """Generate exhaustive clinical & statistical metrics for a model."""
    y_pred = model.predict(X_eval)
    
    # Ensure 1D prediction array
    if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
        
    y_proba = model.predict_proba(X_eval)
    n_classes = len(label_classes)
    
    acc = accuracy_score(y_eval_encoded, y_pred)
    bal_acc = balanced_accuracy_score(y_eval_encoded, y_pred)
    p_macro = precision_score(y_eval_encoded, y_pred, average='macro', zero_division=0)
    r_macro = recall_score(y_eval_encoded, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_eval_encoded, y_pred, average='macro', zero_division=0)
    
    p_weighted = precision_score(y_eval_encoded, y_pred, average='weighted', zero_division=0)
    r_weighted = recall_score(y_eval_encoded, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_eval_encoded, y_pred, average='weighted', zero_division=0)
    
    try:
        roc_auc = float(roc_auc_score(y_eval_encoded, y_proba, multi_class='ovr', average='macro'))
    except Exception:
        roc_auc = None
        
    pr_auc = compute_multiclass_pr_auc(y_eval_encoded, y_proba)
    brier = compute_multiclass_brier(y_eval_encoded, y_proba)
    cm = confusion_matrix(y_eval_encoded, y_pred)
    
    # Class-wise metrics
    p_per_class = precision_score(y_eval_encoded, y_pred, average=None, zero_division=0).tolist()
    r_per_class = recall_score(y_eval_encoded, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_eval_encoded, y_pred, average=None, zero_division=0).tolist()
    
    class_metrics = {}
    high_risk_idx = None
    for idx, c_name in enumerate(label_classes):
        class_metrics[str(c_name)] = {
            "precision": float(p_per_class[idx]),
            "recall": float(r_per_class[idx]),
            "f1": float(f1_per_class[idx])
        }
        if str(c_name).lower() == 'high':
            high_risk_idx = idx
            
    high_risk_recall = float(r_per_class[high_risk_idx]) if high_risk_idx is not None else float(r_macro)
    high_risk_precision = float(p_per_class[high_risk_idx]) if high_risk_idx is not None else float(p_macro)
    high_risk_f1 = float(f1_per_class[high_risk_idx]) if high_risk_idx is not None else float(f1_macro)
    
    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro": roc_auc,
        "pr_auc_macro": pr_auc,
        "brier_score": brier,
        "high_risk_recall": high_risk_recall,
        "high_risk_precision": high_risk_precision,
        "high_risk_f1": high_risk_f1,
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist()
    }
    return metrics

def get_candidate_models():
    """Return dictionary of the 5 benchmark candidate models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
    }
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(iterations=200, random_state=42, verbose=0, auto_class_weights='Balanced')
    return models

def run_5fold_cross_validation(model_factory, X_train, y_train_encoded, label_classes):
    """Run 5-fold Stratified Cross-Validation on training split."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    acc_scores = []
    roc_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train_encoded):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train_encoded[train_idx], y_train_encoded[val_idx]
        
        m = model_factory()
        m.fit(X_tr, y_tr)
        preds = m.predict(X_va)
        if hasattr(preds, "ndim") and preds.ndim > 1:
            preds = preds.ravel()
            
        f1_scores.append(f1_score(y_va, preds, average='macro', zero_division=0))
        acc_scores.append(accuracy_score(y_va, preds))
        try:
            probas = m.predict_proba(X_va)
            roc_scores.append(roc_auc_score(y_va, probas, multi_class='ovr', average='macro'))
        except Exception:
            pass
            
    return {
        "cv_f1_macro_mean": float(np.mean(f1_scores)),
        "cv_f1_macro_std": float(np.std(f1_scores)),
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_accuracy_std": float(np.std(acc_scores)),
        "cv_roc_auc_mean": float(np.mean(roc_scores)) if roc_scores else None
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "stage1_ml", "features", "processed_features.csv")
    models_dir = os.path.join(base_dir, "data", "stage1_ml", "models")
    docs_dir = os.path.join(base_dir, "docs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    print(f"Loading processed dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    targets = ['overall_patient_risk', 'toxicity_risk', 'therapy_response']
    
    train_df = df[df['dataset_split'] == 'train']
    test_df = df[df['dataset_split'] == 'test']
    
    X_train = train_df.drop(columns=targets + ['dataset_split'])
    X_train.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in X_train.columns]
    y_train_full = train_df[targets]
    
    X_test = test_df.drop(columns=targets + ['dataset_split'])
    X_test.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in X_test.columns]
    y_test_full = test_df[targets]
    
    all_results = {}
    best_models = {}
    baseline_results = {}
    
    for target in targets:
        print(f"\n==========================================")
        print(f"BENCHMARKING MODELS FOR TARGET: {target}")
        print(f"==========================================")
        
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train_full[target])
        y_ts_enc = le.transform(y_test_full[target])
        
        # Save LabelEncoder
        joblib.dump(le, os.path.join(models_dir, f"{target}_label_encoder.joblib"))
        
        # Phase 6: Baseline Model (Majority Class Dummy Classifier)
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_tr_enc)
        dummy_metrics = evaluate_comprehensive_metrics(dummy, X_test, y_ts_enc, le.classes_)
        baseline_results[target] = {
            "strategy": "most_frequent_majority_class",
            "metrics": dummy_metrics
        }
        print(f"Baseline (Majority Class) Accuracy: {dummy_metrics['accuracy']:.4f} | Macro F1: {dummy_metrics['f1_macro']:.4f}")
        
        candidate_models = get_candidate_models()
        target_results = {}
        best_model_name = None
        best_f1 = -1
        
        for model_name, model in candidate_models.items():
            print(f"Training {model_name}...")
            start_time = time.time()
            model.fit(X_train, y_tr_enc)
            train_time = time.time() - start_time
            
            # Evaluate on Test Set
            metrics = evaluate_comprehensive_metrics(model, X_test, y_ts_enc, le.classes_)
            
            # Run 5-fold CV on Training split
            def model_factory():
                return get_candidate_models()[model_name]
                
            cv_results = run_5fold_cross_validation(model_factory, X_train, y_tr_enc, le.classes_)
            
            target_results[model_name] = {
                "metrics": metrics,
                "cv_results": cv_results,
                "training_time_seconds": float(train_time),
                "parameters": str(model.get_params())
            }
            
            # Save base model joblib
            filename = f"{target}_{model_name.replace(' ', '_').lower()}.joblib"
            joblib.dump(model, os.path.join(models_dir, filename))
            
            print(f"  - {model_name}: Test Macro F1: {metrics['f1_macro']:.4f} | Acc: {metrics['accuracy']:.4f} | High-Risk Recall: {metrics['high_risk_recall']:.4f} | 5-Fold CV F1: {cv_results['cv_f1_macro_mean']:.4f} ± {cv_results['cv_f1_macro_std']:.4f}")
            
            if metrics['f1_macro'] > best_f1:
                best_f1 = metrics['f1_macro']
                best_model_name = model_name
                
        all_results[target] = target_results
        best_models[target] = {
            "best_model": best_model_name,
            "best_f1_macro": target_results[best_model_name]["metrics"]["f1_macro"],
            "classes": le.classes_.tolist()
        }
        
    # Save baseline_results.json
    baseline_path = os.path.join(models_dir, "baseline_results.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline_results, f, indent=4)
        
    # Save model comparison & best model JSONs
    with open(os.path.join(models_dir, "model_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=4)
        
    with open(os.path.join(models_dir, "best_model.json"), "w") as f:
        json.dump(best_models, f, indent=4)
        
    print("\nBenchmark completed. Outputs saved to data/stage1_ml/models/")

if __name__ == "__main__":
    main()
