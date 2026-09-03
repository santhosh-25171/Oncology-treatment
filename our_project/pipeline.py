"""
Stage 01 — Patient Risk Score Pipeline
========================================
Follows this exact flow:

Historical patient data -> Clean data -> Select features -> Train/Test split
-> Train ML model -> Make predictions -> Evaluate predictions -> Tune model
-> Select best model

Usage:
    python pipeline.py --data oncology_cleaned_datasets.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

RANDOM_STATE = 42
TARGET_COL = "risk_level"


# ---------------------------------------------------------------------------
# STEP 1: Historical patient data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"STEP 1 — Loaded historical patient data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# STEP 2: Clean data
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with no recorded target, if any slipped through upstream cleaning.
    # (Safe no-op if the EDA-checked dataset has already removed these.)
    df = df[df[TARGET_COL] != "Not Recorded"].copy()

    # "Not Recorded" survival status is a real state (e.g. lost to follow-up),
    # not a true missing value — keep it as its own category rather than dropping.
    # (Also a no-op if upstream cleaning already removed these rows.)
    if "survival_status" in df.columns:
        df["survival_status"] = df["survival_status"].replace({"Not Recorded": "Unknown"})

    # Normalize casing/whitespace so "Stage Ii" and "STAGE II" are treated as
    # the same category rather than silently becoming two separate one-hot columns.
    if "cancer_stage" in df.columns:
        df["cancer_stage"] = df["cancer_stage"].str.strip().str.upper()

    print(f"STEP 2 — Cleaned data: {df.shape[0]} rows remain after removing unrecorded targets")
    return df


# ---------------------------------------------------------------------------
# STEP 3: Select features
# ---------------------------------------------------------------------------
def select_features(df: pd.DataFrame):
    # Excluded on purpose: outcome-adjacent fields not known at patient intake.
    # Using them would leak future information into a real-time risk scorer.
    leakage_cols = ["survival_status", "followup_months"]
    id_cols = ["patient_id", "diagnosis_date"]

    feature_cols = [c for c in df.columns if c not in id_cols + leakage_cols + [TARGET_COL]]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    print(f"STEP 3 — Selected {len(feature_cols)} features")
    print(f"          Numeric: {numeric_cols}")
    print(f"          Categorical: {categorical_cols}")
    print(f"          Excluded (leakage risk): {leakage_cols}")

    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y, numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# STEP 4: Train/test split
# ---------------------------------------------------------------------------
def split_data(X, y_enc):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    print(f"STEP 4 — Train/test split: {len(X_train)} train / {len(X_test)} test")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# STEP 5: Train ML model
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    print("STEP 5 — Trained initial Random Forest model")
    return model


# ---------------------------------------------------------------------------
# STEP 6: Make predictions
# ---------------------------------------------------------------------------
def make_predictions(model, X_test):
    preds = model.predict(X_test)
    print("STEP 6 — Generated predictions on the test set")
    return preds


# ---------------------------------------------------------------------------
# STEP 7: Evaluate predictions
# ---------------------------------------------------------------------------
def evaluate_predictions(y_test, preds, class_names, label=""):
    f1 = f1_score(y_test, preds, average="macro")
    print(f"STEP 7 — Evaluation {label}(macro F1 = {f1:.3f})")
    print(classification_report(y_test, preds, target_names=class_names))

    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm,
                          index=[f"true_{c}" for c in class_names],
                          columns=[f"pred_{c}" for c in class_names])
    print(cm_df)
    return f1, cm_df


# ---------------------------------------------------------------------------
# STEP 8: Tune model
# ---------------------------------------------------------------------------
def tune_model(X_train, y_train, numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")),
    ])

    param_grid = {
        "clf__n_estimators": [100],
        "clf__max_depth": [6, 10],
        "clf__min_samples_leaf": [1, 5],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv)
    search.fit(X_train, y_train)

    print(f"STEP 8 — Tuned model. Best params: {search.best_params_}")
    print(f"          Best CV macro F1: {search.best_score_:.3f}")
    return search.best_estimator_, search.best_score_


# ---------------------------------------------------------------------------
# STEP 9: Select best model
# ---------------------------------------------------------------------------
def select_best_model(candidates: dict):
    best_name = max(candidates, key=lambda k: candidates[k][1])
    best_model, best_score = candidates[best_name]
    print(f"STEP 9 — Selected best model: {best_name} (macro F1 = {best_score:.3f})")
    return best_name, best_model, best_score


def main(data_path: str):
    # Step 1
    df = load_data(data_path)

    # Step 2
    df = clean_data(df)

    # Step 3
    X, y, numeric_cols, categorical_cols = select_features(df)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_

    # Step 4
    X_train, X_test, y_train, y_test = split_data(X, y_enc)

    # Step 5
    baseline_model = train_model(X_train, y_train, numeric_cols, categorical_cols)

    # Step 6
    baseline_preds = make_predictions(baseline_model, X_test)

    # Step 7
    baseline_f1, _ = evaluate_predictions(y_test, baseline_preds, class_names, label="(baseline) ")

    # Step 8
    tuned_model, tuned_cv_score = tune_model(X_train, y_train, numeric_cols, categorical_cols)
    tuned_preds = make_predictions(tuned_model, X_test)
    tuned_f1, cm_df = evaluate_predictions(y_test, tuned_preds, class_names, label="(tuned) ")

    # Step 9
    candidates = {
        "Baseline Random Forest": (baseline_model, baseline_f1),
        "Tuned Random Forest": (tuned_model, tuned_f1),
    }
    best_name, best_model, best_score = select_best_model(candidates)

    # Save the winning model
    joblib.dump(best_model, "risk_score_model.joblib")
    joblib.dump(le, "label_encoder.joblib")
    cm_df.to_csv("confusion_matrix.csv")

    metadata = {
        "best_model": best_name,
        "test_macro_f1": float(best_score),
        "classes": list(class_names),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved: risk_score_model.joblib, label_encoder.joblib, confusion_matrix.csv, model_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient risk score pipeline")
    parser.add_argument("--data", default="oncology_cleaned_datasets.csv", help="Path to input CSV")
    args = parser.parse_args()
    main(args.data)
