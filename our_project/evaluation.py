"""
Stage 01 — Patient Risk Score Evaluation
========================================

Evaluation flow for the ML implementation in pipeline.py:

1. Load historical patient data
2. Clean data using the same rules as the ML pipeline
3. Select the same leakage-safe features
4. Recreate the same deterministic train/test split
5. Load the trained winning ML model + label encoder
6. Generate predictions on the untouched test set
7. Evaluate predictions
8. Perform error analysis
9. Check class balance and high-risk recall
10. Produce a final evaluation decision/report

IMPORTANT:
- This file does NOT train or tune a new model.
- It evaluates the model produced by pipeline.py.
- Run pipeline.py first so these files exist:
      risk_score_model.joblib
      label_encoder.joblib
- The evaluation uses the same RANDOM_STATE=42 and 20% stratified test split
  as pipeline.py, so the test set is reproduced deterministically.

Usage:
    python evaluation.py --data oncology_cleaned_datasets.csv

Outputs:
    evaluation_outputs/
        classification_report.csv
        confusion_matrix.csv
        error_analysis.csv
        class_distribution.csv
        evaluation_report.json
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_COL = "risk_level"
TEST_SIZE = 0.20

MODEL_FILE = "risk_score_model.joblib"
ENCODER_FILE = "label_encoder.joblib"
OUTPUT_DIR = Path("evaluation_outputs")


# ---------------------------------------------------------------------------
# STEP 1: Load historical patient data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    print("=" * 70)
    print("STEP 1 — LOAD HISTORICAL PATIENT DATA")
    print("=" * 70)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(f"Target  : {TARGET_COL}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    return df


# ---------------------------------------------------------------------------
# STEP 2: Clean data — same logic as pipeline.py
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[TARGET_COL] != "Not Recorded"].copy()

    if "survival_status" in df.columns:
        df["survival_status"] = df["survival_status"].replace(
            {"Not Recorded": "Unknown"}
        )

    if "cancer_stage" in df.columns:
        df["cancer_stage"] = df["cancer_stage"].str.strip().str.upper()

    print("\n" + "=" * 70)
    print("STEP 2 — CLEAN DATA")
    print("=" * 70)
    print(f"Rows before cleaning : {len(df) + 0}")  # final cleaned row count shown below
    print(f"Rows after cleaning  : {len(df)}")
    print("Unrecorded risk_level rows removed.")
    print("No missing values are expected in the supplied cleaned dataset.")

    return df


# ---------------------------------------------------------------------------
# STEP 3: Select features — same logic as pipeline.py
# ---------------------------------------------------------------------------
def select_features(df: pd.DataFrame):
    leakage_cols = ["survival_status", "followup_months"]
    id_cols = ["patient_id", "diagnosis_date"]

    feature_cols = [
        c for c in df.columns
        if c not in id_cols + leakage_cols + [TARGET_COL]
    ]

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    categorical_cols = [
        c for c in feature_cols
        if c not in numeric_cols
    ]

    X = df[feature_cols]
    y = df[TARGET_COL]

    print("\n" + "=" * 70)
    print("STEP 3 — SELECT FEATURES")
    print("=" * 70)
    print(f"Number of features : {len(feature_cols)}")
    print(f"Numeric features   : {numeric_cols}")
    print(f"Categorical        : {categorical_cols}")
    print(f"Excluded IDs       : {id_cols}")
    print(f"Excluded leakage   : {leakage_cols}")

    return X, y, feature_cols


# ---------------------------------------------------------------------------
# STEP 4: Recreate the same train/test split as pipeline.py
# ---------------------------------------------------------------------------
def recreate_test_split(X, y, encoder):
    y_encoded = encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print("\n" + "=" * 70)
    print("STEP 4 — RECREATE TRAIN & TEST SETS")
    print("=" * 70)
    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print("Stratified split  : YES")
    print(f"Random state      : {RANDOM_STATE}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# STEP 5: Load trained ML model
# ---------------------------------------------------------------------------
def load_model():
    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            f"{MODEL_FILE} not found. Run pipeline.py first."
        )

    if not Path(ENCODER_FILE).exists():
        raise FileNotFoundError(
            f"{ENCODER_FILE} not found. Run pipeline.py first."
        )

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)

    print("\n" + "=" * 70)
    print("STEP 5 — LOAD TRAINED ML MODEL")
    print("=" * 70)
    print(f"Model  : {MODEL_FILE}")
    print(f"Encoder: {ENCODER_FILE}")
    print("Model loaded successfully.")

    return model, encoder


# ---------------------------------------------------------------------------
# STEP 6: Generate predictions
# ---------------------------------------------------------------------------
def make_predictions(model, X_test):
    preds = model.predict(X_test)

    print("\n" + "=" * 70)
    print("STEP 6 — MAKE PREDICTIONS")
    print("=" * 70)
    print(f"Predictions generated: {len(preds)}")

    return preds


# ---------------------------------------------------------------------------
# STEP 7: Evaluate predictions
# ---------------------------------------------------------------------------
def evaluate_predictions(y_test, preds, class_names):
    accuracy = accuracy_score(y_test, preds)
    precision_macro = precision_score(
        y_test, preds, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_test, preds, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        y_test, preds, average="macro", zero_division=0
    )
    f1_weighted = f1_score(
        y_test, preds, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_test,
        preds,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(
        y_test,
        preds,
        labels=np.arange(len(class_names)),
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )

    print("\n" + "=" * 70)
    print("STEP 7 — EVALUATE PREDICTIONS")
    print("=" * 70)
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall   : {recall_macro:.4f}")
    print(f"Macro F1       : {f1_macro:.4f}")
    print(f"Weighted F1    : {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(report_df.round(4).to_string())

    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
    }

    return metrics, report_df, cm_df


# ---------------------------------------------------------------------------
# STEP 7.5: Overfitting / Underfitting Analysis
# ---------------------------------------------------------------------------
def analyze_overfitting_underfitting(model, X_train, y_train, X_test, y_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    train_f1 = f1_score(y_train, train_preds, average="macro", zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)

    acc_gap = train_acc - test_acc
    f1_gap = train_f1 - test_f1

    if acc_gap > 0.08 or f1_gap > 0.08:
        status = "OVERFITTING (Model performs significantly better on train set than test set)"
        status_code = "OVERFITTING"
    elif train_acc < 0.55 and test_acc < 0.55:
        status = "UNDERFITTING (Model has low performance on both train and test sets)"
        status_code = "UNDERFITTING"
    else:
        status = "WELL-FITTED (Model generalizes well with minimal train-test gap)"
        status_code = "WELL-FITTED"

    print("\n" + "=" * 70)
    print("STEP 7.5 — OVERFITTING / UNDERFITTING ANALYSIS")
    print("=" * 70)
    print(f"Train Accuracy    : {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"Test Accuracy     : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Accuracy Gap      : {acc_gap:.4f} ({acc_gap * 100:.2f}%)")
    print(f"Train Macro F1    : {train_f1:.4f}")
    print(f"Test Macro F1     : {test_f1:.4f}")
    print(f"Macro F1 Gap      : {f1_gap:.4f}")
    print(f"Fit Assessment    : {status}")

    fit_metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "accuracy_gap": float(acc_gap),
        "train_macro_f1": float(train_f1),
        "test_macro_f1": float(test_f1),
        "macro_f1_gap": float(f1_gap),
        "fit_status": status_code,
    }

    return fit_metrics


# ---------------------------------------------------------------------------
# STEP 8: Error analysis
# ---------------------------------------------------------------------------
def error_analysis(X_test, y_test, preds, encoder):
    actual = encoder.inverse_transform(y_test)
    predicted = encoder.inverse_transform(preds)

    error_df = X_test.copy()
    error_df["actual_risk"] = actual
    error_df["predicted_risk"] = predicted
    error_df["correct"] = error_df["actual_risk"] == error_df["predicted_risk"]

    errors_only = error_df[~error_df["correct"]].copy()

    error_summary = (
        errors_only
        .groupby(["actual_risk", "predicted_risk"])
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
    )

    print("\n" + "=" * 70)
    print("STEP 8 — ERROR ANALYSIS")
    print("=" * 70)
    print(f"Total test samples : {len(error_df)}")
    print(f"Incorrect samples  : {len(errors_only)}")
    print(
        f"Error rate         : "
        f"{len(errors_only) / len(error_df):.4f}"
    )

    if len(error_summary) > 0:
        print("\nMost common prediction errors:")
        print(error_summary.to_string(index=False))
    else:
        print("No prediction errors found.")

    return error_df, error_summary


# ---------------------------------------------------------------------------
# STEP 9: Class balance + high-risk evaluation
# ---------------------------------------------------------------------------
def risk_class_evaluation(y_test, preds, encoder):
    actual = encoder.inverse_transform(y_test)
    predicted = encoder.inverse_transform(preds)

    class_distribution = (
        pd.Series(actual, name="risk_level")
        .value_counts()
        .rename_axis("risk_level")
        .reset_index(name="test_samples")
    )

    per_class_recall = recall_score(
        y_test,
        preds,
        labels=np.arange(len(encoder.classes_)),
        average=None,
        zero_division=0,
    )

    recall_table = pd.DataFrame({
        "risk_level": encoder.classes_,
        "recall": per_class_recall,
    })

    high_risk_recall = None
    if "High" in encoder.classes_:
        high_index = list(encoder.classes_).index("High")
        high_risk_recall = float(per_class_recall[high_index])

    critical_recall = None
    if "Critical" in encoder.classes_:
        critical_index = list(encoder.classes_).index("Critical")
        critical_recall = float(per_class_recall[critical_index])

    print("\n" + "=" * 70)
    print("STEP 9 — RISK-CLASS EVALUATION")
    print("=" * 70)

    print("\nTest-set class distribution:")
    print(class_distribution.to_string(index=False))

    print("\nRecall by risk class:")
    print(recall_table.round(4).to_string(index=False))

    if high_risk_recall is not None:
        print(f"\nHigh Risk Recall : {high_risk_recall:.4f}")

    if critical_recall is not None:
        print(f"Critical Recall  : {critical_recall:.4f}")

    return class_distribution, recall_table, high_risk_recall, critical_recall


# ---------------------------------------------------------------------------
# STEP 10: Final evaluation decision
# ---------------------------------------------------------------------------
def final_decision(metrics, high_risk_recall, critical_recall):
    """
    These are project-level evaluation thresholds, not medical/regulatory
    thresholds. They are used only to demonstrate a PASS/REVIEW decision.

    PASS conditions:
      - Macro F1 >= 0.80
      - Macro Recall >= 0.75
      - High Risk Recall >= 0.75 when High exists
      - Critical Recall >= 0.75 when Critical exists

    If the model does not meet them, return REVIEW rather than FAIL because
    this evaluation alone cannot establish clinical safety.
    """

    checks = {
        "macro_f1": metrics["macro_f1"] >= 0.80,
        "macro_recall": metrics["macro_recall"] >= 0.75,
    }

    if high_risk_recall is not None:
        checks["high_risk_recall"] = high_risk_recall >= 0.75

    if critical_recall is not None:
        checks["critical_recall"] = critical_recall >= 0.75

    decision = "PASS" if all(checks.values()) else "REVIEW"

    print("\n" + "=" * 70)
    print("STEP 10 — FINAL EVALUATION DECISION")
    print("=" * 70)

    for metric, passed in checks.items():
        print(f"{metric:20s}: {'PASS' if passed else 'REVIEW'}")

    print(f"\nFINAL DECISION: {decision}")

    return decision, checks


# ---------------------------------------------------------------------------
# Save evaluation artifacts
# ---------------------------------------------------------------------------
def save_outputs(
    metrics,
    report_df,
    cm_df,
    error_summary,
    class_distribution,
    recall_table,
    decision,
    checks,
    high_risk_recall,
    critical_recall,
    feature_cols,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )
    cm_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )
    error_summary.to_csv(
        OUTPUT_DIR / "error_analysis.csv",
        index=False,
    )
    class_distribution.to_csv(
        OUTPUT_DIR / "class_distribution.csv",
        index=False,
    )
    recall_table.to_csv(
        OUTPUT_DIR / "risk_class_recall.csv",
        index=False,
    )

    evaluation_report = {
        "evaluation_type": "Stage 01 Patient Risk Score Evaluation",
        "model_file": MODEL_FILE,
        "dataset": "oncology_cleaned_datasets.csv",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "features_used": feature_cols,
        "metrics": metrics,
        "high_risk_recall": high_risk_recall,
        "critical_recall": critical_recall,
        "decision": decision,
        "decision_checks": checks,
        "note": (
            "PASS/REVIEW thresholds are project evaluation thresholds only; "
            "they are not clinical or regulatory acceptance criteria."
        ),
    }

    with open(
        OUTPUT_DIR / "evaluation_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(evaluation_report, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION OUTPUTS SAVED")
    print("=" * 70)

    for file in sorted(OUTPUT_DIR.iterdir()):
        print(file)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(data_path: str):
    # Step 1
    df = load_data(data_path)

    # Step 2
    rows_before = len(df)
    df = clean_data(df)
    rows_after = len(df)

    # Correct the displayed cleaning count.
    print(f"Rows removed during cleaning: {rows_before - rows_after}")

    # Step 3
    X, y, feature_cols = select_features(df)

    # Step 5 must happen before Step 4 because the label encoder is required
    # to reproduce the exact encoded target used by pipeline.py.
    model, encoder = load_model()

    # Step 4
    X_train, X_test, y_train, y_test = recreate_test_split(X, y, encoder)

    # Step 6
    preds = make_predictions(model, X_test)

    # Step 7
    metrics, report_df, cm_df = evaluate_predictions(
        y_test, preds, encoder.classes_
    )

    # Step 7.5
    fit_metrics = analyze_overfitting_underfitting(
        model, X_train, y_train, X_test, y_test
    )
    metrics.update(fit_metrics)

    # Step 8
    _, error_summary = error_analysis(
        X_test, y_test, preds, encoder
    )

    # Step 9
    (
        class_distribution,
        recall_table,
        high_risk_recall,
        critical_recall,
    ) = risk_class_evaluation(y_test, preds, encoder)

    # Step 10
    decision, checks = final_decision(
        metrics,
        high_risk_recall,
        critical_recall,
    )

    save_outputs(
        metrics=metrics,
        report_df=report_df,
        cm_df=cm_df,
        error_summary=error_summary,
        class_distribution=class_distribution,
        recall_table=recall_table,
        decision=decision,
        checks=checks,
        high_risk_recall=high_risk_recall,
        critical_recall=critical_recall,
        feature_cols=feature_cols,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 01 Patient Risk Score ML model"
    )
    parser.add_argument(
        "--data",
        default="oncology_cleaned_datasets.csv",
        help="Path to the evaluation dataset",
    )

    args = parser.parse_args()
    main(args.data)
