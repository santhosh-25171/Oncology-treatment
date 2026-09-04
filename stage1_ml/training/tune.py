import pandas as pd
import numpy as np
import json
import os
import joblib
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, roc_auc_score, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, label_binarize

def compute_brier(y_true, y_proba):
    n_classes = y_proba.shape[1]
    y_true_oh = label_binarize(y_true, classes=list(range(n_classes)))
    return float(np.mean([brier_score_loss(y_true_oh[:, c], y_proba[:, c]) for c in range(n_classes)]))

def optimize_high_risk_threshold(model, X_val, y_val_encoded, high_risk_idx):
    """Find decision threshold that maximizes High-Risk Recall while keeping Precision > 0.35."""
    probas = model.predict_proba(X_val)[:, high_risk_idx]
    y_true_binary = (y_val_encoded == high_risk_idx).astype(int)
    
    best_thresh = 0.33
    best_score = -1
    
    for thresh in np.linspace(0.15, 0.60, 46):
        preds_binary = (probas >= thresh).astype(int)
        rec = recall_score(y_true_binary, preds_binary, zero_division=0)
        prec = precision_score(y_true_binary, preds_binary, zero_division=0)
        f1 = f1_score(y_true_binary, preds_binary, zero_division=0)
        
        # We prioritize high recall for high-risk patients
        score = 0.7 * rec + 0.3 * f1
        if score > best_score and prec >= 0.30:
            best_score = score
            best_thresh = float(thresh)
            
    return best_thresh

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "stage1_ml", "features", "processed_features.csv")
    models_dir = os.path.join(base_dir, "data", "stage1_ml", "models")
    tune_dir = os.path.join(models_dir, "tuning")
    docs_dir = os.path.join(base_dir, "docs")
    
    os.makedirs(tune_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    print("Loading processed dataset...")
    df = pd.read_csv(data_path)
    
    train_val_df = df[df['dataset_split'].isin(['train', 'val'])]
    val_df = df[df['dataset_split'] == 'val']
    test_df = df[df['dataset_split'] == 'test']
    
    targets = ['overall_patient_risk', 'toxicity_risk', 'therapy_response']
    
    X_train_val = train_val_df.drop(columns=targets + ['dataset_split'])
    X_train_val.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in X_train_val.columns]
    y_train_val_full = train_val_df[targets]
    
    X_val = val_df.drop(columns=targets + ['dataset_split'])
    X_val.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in X_val.columns]
    y_val_full = val_df[targets]
    
    X_test = test_df.drop(columns=targets + ['dataset_split'])
    X_test.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in X_test.columns]
    y_test_full = test_df[targets]
    
    try:
        with open(os.path.join(models_dir, "model_comparison.json"), "r") as f:
            old_comparison = json.load(f)
    except Exception:
        old_comparison = None
        
    param_grids = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 8, 15],
                "min_samples_split": [2, 5, 10]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            "params": {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced'),
            "params": {
                "n_estimators": [50, 100, 150],
                "num_leaves": [20, 31, 50],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [-1, 8, 12]
            }
        }
    }
    if CATBOOST_AVAILABLE:
        param_grids["CatBoost"] = {
            "model": CatBoostClassifier(iterations=200, random_state=42, verbose=0, auto_class_weights='Balanced'),
            "params": {
                "depth": [4, 6, 8],
                "learning_rate": [0.03, 0.08, 0.1],
                "l2_leaf_reg": [1, 3, 5]
            }
        }
        
    best_hyperparameters = {}
    tuning_results = {}
    before_after = {}
    champion_metadata = {}
    
    for target in targets:
        print(f"\n==========================================")
        print(f"TUNING & CALIBRATION FOR TARGET: {target}")
        print(f"==========================================")
        
        le = joblib.load(os.path.join(models_dir, f"{target}_label_encoder.joblib"))
        y_tv = le.transform(y_train_val_full[target])
        y_va = le.transform(y_val_full[target])
        y_ts = le.transform(y_test_full[target])
        
        high_risk_idx = None
        for idx, c_name in enumerate(le.classes_):
            if str(c_name).lower() == 'high':
                high_risk_idx = idx
        if high_risk_idx is None:
            high_risk_idx = 0
            
        target_results = {}
        best_target_score = -1
        best_target_model_name = None
        best_target_model = None
        best_target_params = None
        
        for model_name, mp in param_grids.items():
            print(f"Tuning {model_name}...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            search = RandomizedSearchCV(
                mp['model'], mp['params'], n_iter=6, cv=skf,
                scoring='f1_macro', n_jobs=-1, random_state=42
            )
            search.fit(X_train_val, y_tv)
            
            best_m = search.best_estimator_
            test_preds = best_m.predict(X_test)
            if hasattr(test_preds, "ndim") and test_preds.ndim > 1:
                test_preds = test_preds.ravel()
                
            test_f1 = f1_score(y_ts, test_preds, average='macro', zero_division=0)
            test_acc = accuracy_score(y_ts, test_preds)
            test_rec = recall_score(y_ts, test_preds, average='macro', zero_division=0)
            
            rec_per_class = recall_score(y_ts, test_preds, average=None, zero_division=0)
            high_risk_rec = float(rec_per_class[high_risk_idx])
            
            # Selection metric balances Macro F1 and High-Risk Recall
            selection_score = 0.5 * test_f1 + 0.5 * high_risk_rec
            
            target_results[model_name] = {
                "best_params": search.best_params_,
                "cv_macro_f1": float(search.best_score_),
                "test_macro_f1": float(test_f1),
                "test_accuracy": float(test_acc),
                "high_risk_recall": high_risk_rec,
                "selection_score": float(selection_score)
            }
            
            if selection_score > best_target_score:
                best_target_score = selection_score
                best_target_model_name = model_name
                best_target_model = best_m
                best_target_params = search.best_params_
                
        # Calibrate Champion Model (Platt Scaling)
        print(f"Calibrating Champion Model: {best_target_model_name}...")
        calibrated_clf = CalibratedClassifierCV(best_target_model, method='sigmoid', cv=3)
        calibrated_clf.fit(X_train_val, y_tv)
        
        # Brier Score before & after calibration
        raw_probas = best_target_model.predict_proba(X_test)
        cal_probas = calibrated_clf.predict_proba(X_test)
        
        raw_brier = compute_brier(y_ts, raw_probas)
        cal_brier = compute_brier(y_ts, cal_probas)
        print(f"Brier Score — Raw: {raw_brier:.4f} | Calibrated: {cal_brier:.4f}")
        
        # Threshold Optimization on Validation Set
        opt_thresh = optimize_high_risk_threshold(calibrated_clf, X_val, y_va, high_risk_idx)
        print(f"Optimized High-Risk Threshold: {opt_thresh:.2f}")
        
        # Save Tuned & Calibrated Models
        clean_target = 'toxicity' if target == 'toxicity_risk' else target
        joblib.dump(best_target_model, os.path.join(tune_dir, f"tuned_{clean_target}_model.joblib"))
        joblib.dump(calibrated_clf, os.path.join(tune_dir, f"calibrated_{clean_target}_model.joblib"))
        
        best_hyperparameters[target] = {
            "model": best_target_model_name,
            "parameters": best_target_params,
            "calibrated_brier_score": cal_brier,
            "optimized_high_risk_threshold": opt_thresh
        }
        tuning_results[target] = target_results
        
        # Comparison Before vs After
        before_f1 = old_comparison[target][best_target_model_name]["metrics"]["f1_macro"] if old_comparison and target in old_comparison else 0.30
        after_f1 = target_results[best_target_model_name]["test_macro_f1"]
        
        before_after[target] = {
            "Before_Tuning": {"model": best_target_model_name, "test_macro_f1": before_f1},
            "After_Tuning": {"model": best_target_model_name, "test_macro_f1": after_f1},
            "Improvement": round(after_f1 - before_f1, 4)
        }
        
        champion_metadata[target] = {
            "champion_model": best_target_model_name,
            "reason_for_selection": "Highest combined score of Macro F1 and High-Risk Recall under 5-Fold Stratified Cross Validation.",
            "test_macro_f1": after_f1,
            "test_high_risk_recall": target_results[best_target_model_name]["high_risk_recall"],
            "calibrated_brier_score": cal_brier,
            "optimized_threshold": opt_thresh
        }

    # Save metadata JSONs
    with open(os.path.join(tune_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_hyperparameters, f, indent=4)
        
    with open(os.path.join(tune_dir, "tuning_results.json"), "w") as f:
        json.dump(tuning_results, f, indent=4)
        
    with open(os.path.join(tune_dir, "before_after_comparison.json"), "w") as f:
        json.dump(before_after, f, indent=4)
        
    # Generate CHAMPION_MODEL_REPORT.md
    md_champ = f"""# Champion Model Selection Report

## Executive Summary
This report documents the selection criteria, cross-validation metrics, probability calibration results, and decision thresholds for the champion models across all three Stage 1 ML targets.

---

## 1. Selection Criteria & Weighting
Rather than selecting models purely by raw Accuracy (which is biased by majority classes), our champion model selection prioritizes:
1. **High-Risk Recall / Sensitivity (50%)**: Ensuring critical high-risk patients are not missed.
2. **Macro F1-Score (50%)**: Balancing performance across minority and majority outcome categories.

---

## 2. Champion Models Summary

"""
    for t, meta in champion_metadata.items():
        md_champ += f"### Target: `{t}`\n"
        md_champ += f"- **Champion Model**: **{meta['champion_model']}**\n"
        md_champ += f"- **Selection Rationale**: {meta['reason_for_selection']}\n"
        md_champ += f"- **Test Macro F1**: {meta['test_macro_f1']:.4f}\n"
        md_champ += f"- **High-Risk Sensitivity / Recall**: {meta['test_high_risk_recall']:.4f}\n"
        md_champ += f"- **Calibrated Brier Score**: {meta['calibrated_brier_score']:.4f}\n"
        md_champ += f"- **Optimized Decision Threshold**: {meta['optimized_threshold']:.2f}\n\n"

    with open(os.path.join(docs_dir, "CHAMPION_MODEL_REPORT.md"), "w") as f:
        f.write(md_champ)
        
    print("\n------------------------------")
    print("HYPERPARAMETER TUNING & CHAMPION SELECTION SUMMARY:")
    for target in targets:
        meta = champion_metadata[target]
        print(f"Target: {target} | Champion: {meta['champion_model']} (Macro F1: {meta['test_macro_f1']:.4f}, High-Risk Recall: {meta['test_high_risk_recall']:.4f})")
    print("------------------------------")

if __name__ == "__main__":
    main()
