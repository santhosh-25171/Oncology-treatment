import pandas as pd
import numpy as np
import json
import os
import sys

def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset not found at {file_path}")
    return pd.read_csv(file_path)

def verify_columns(df: pd.DataFrame, expected_features: int = 34, expected_targets: int = 2):
    """Verify that the dataset has the exact expected number of features and targets."""
    target_cols = ['toxicity_risk', 'therapy_response']
    
    # Check if target columns are present
    for target in target_cols:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' is missing.")
            
    num_targets = len(target_cols)
    num_features = len(df.columns) - num_targets
    
    if num_features != expected_features:
        raise ValueError(f"Expected {expected_features} features, but found {num_features}. Total columns: {len(df.columns)}")
    
    print(f"Verified dataset structure: {num_features} features, {num_targets} targets.")

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    initial_shape = df.shape
    df_cleaned = df.drop_duplicates(keep='first')
    duplicates_removed = initial_shape[0] - df_cleaned.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows.")
    return df_cleaned

def handle_invalid_numerical(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """
    Check for impossible or invalid values in numerical columns.
    For instance, age, size, and counts cannot be logically negative in this context.
    Clinically abnormal but valid values are kept.
    """
    df_clean = df.copy()
    invalid_detected = {}
    
    # Check for negative values in columns that logically cannot be negative
    for col in numerical_cols:
        if df_clean[col].dtype.kind in 'biufc':  # numeric types
            # Not all numeric columns must be non-negative, but for typical oncology features like 
            # age, bmi, counts, sizes, negative is invalid.
            # Comorbidity_score could be negative? Let's assume standard clinical variables.
            # If the value is < 0 for physical measures, it is invalid.
            neg_mask = df_clean[col] < 0
            count_invalid = neg_mask.sum()
            
            if count_invalid > 0:
                invalid_detected[col] = int(count_invalid)
                # Replace invalid values with NaN so they are handled in the ML pipeline imputation
                df_clean.loc[neg_mask, col] = np.nan
                
    return df_clean, invalid_detected

def clean_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Standardize categorical values (e.g., lowercasing, stripping spaces)
    to handle unexpected variant categories.
    """
    df_clean = df.copy()
    for col in categorical_cols:
        # Strip whitespace and convert to lower case for consistency
        # only if it's string-like
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
            # Restore np.nan if they were converted to string 'nan'
            df_clean.loc[df_clean[col] == 'nan', col] = np.nan
    return df_clean

def handle_missing_values(df: pd.DataFrame, categorical_cols: list, numerical_cols: list):
    """
    Strategy for missing values:
    - Target columns: Drop rows where target columns are missing.
    - Categorical columns: Fill with 'unknown' (prevents data leakage).
    - Numerical columns: Leave as NaN. (Imputation will be fit ONLY on training data in ML pipeline).
    """
    df_clean = df.copy()
    
    # 1. Drop rows with missing targets
    target_cols = ['toxicity_risk', 'therapy_response']
    df_clean = df_clean.dropna(subset=target_cols)
    
    # 2. Categorical: Fill missing with 'unknown'
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('unknown')
        
    # 3. Numerical: Left as NaN to avoid data leakage before train/test split.
    # No action needed here, but documented for transparency.
    
    return df_clean

def run_cleaning_pipeline(raw_path: str, processed_path: str, report_path: str):
    print(f"Loading raw data from {raw_path}...")
    df = load_data(raw_path)
    
    original_shape = df.shape
    print(f"Original shape: {original_shape}")
    
    target_cols = ['toxicity_risk', 'therapy_response']
    
    # Wait, earlier I noticed from my script run that ALL columns were inferred as 'str'
    # dtype='str' in pd.read_csv output. Let's force pandas to infer correctly, or 
    # try converting to numeric where possible.
    # If read_csv was reading them as object, let's fix that.
    
    # Let's infer objects that are actually numbers
    for col in df.columns:
        if col not in target_cols:
            try:
                df_numeric = pd.to_numeric(df[col])
                df[col] = df_numeric
            except ValueError:
                pass
                
    all_features = [c for c in df.columns if c not in target_cols]
    
    numerical_cols = df[all_features].select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df[all_features].select_dtypes(exclude=['number']).columns.tolist()
    
    missing_before = df.isnull().sum().to_dict()
    
    # Pipeline steps
    verify_columns(df, expected_features=34, expected_targets=2)
    df = remove_duplicates(df)
    
    # Check for invalid numeric values
    df, invalid_detected = handle_invalid_numerical(df, numerical_cols)
    
    # Clean categorical
    df = clean_categorical(df, categorical_cols)
    
    # Get duplicates removed count correctly
    df_raw_loaded = load_data(raw_path)
    duplicates_removed = df_raw_loaded.shape[0] - df_raw_loaded.drop_duplicates(keep='first').shape[0]
    
    # Handle missing values
    df = handle_missing_values(df, categorical_cols, numerical_cols)
    
    missing_after = df.isnull().sum().to_dict()
    final_shape = df.shape
    
    print(f"Final shape: {final_shape}")
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    print(f"Saving cleaned dataset to {processed_path}...")
    df.to_csv(processed_path, index=False)
    
    # Prepare and save report
    report = {
        "original_shape": original_shape,
        "final_shape": final_shape,
        "duplicate_rows_removed": duplicates_removed,
        "missing_values_before_cleaning": {k: int(v) for k, v in missing_before.items()},
        "missing_values_after_cleaning": {k: int(v) for k, v in missing_after.items()},
        "detected_invalid_values": invalid_detected,
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "target_columns": target_cols,
        "cleaning_operations_performed": [
            "Verified 34 features and 2 targets.",
            "Removed exact duplicate rows.",
            "Converted columns to proper numeric types where applicable.",
            "Identified negative/invalid values in numerical columns and replaced with NaN.",
            "Standardized categorical strings (lowercase, stripped whitespace).",
            "Dropped rows with missing target variables.",
            "Filled missing categorical values with 'unknown' (safe from leakage).",
            "Left missing numerical values as NaN to be imputed post train-test split."
        ]
    }
    
    print(f"Saving cleaning report to {report_path}...")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    # Final summary for terminal
    print("-" * 30)
    print("CLEANING SUMMARY:")
    print(f"Raw rows: {original_shape[0]}")
    print(f"Cleaned rows: {final_shape[0]}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Total columns: {final_shape[1]}")
    print(f"Remaining missing values (numerical only): {sum(missing_after.values())}")
    
    print("\nTarget Distributions:")
    for target in target_cols:
        print(f"\n{target}:")
        print(df[target].value_counts().to_string())
    print("-" * 30)

if __name__ == "__main__":
    RAW_PATH = os.path.join("data", "stage1_ml", "raw", "oncology_raw_5000_34_features.csv")
    PROCESSED_PATH = os.path.join("data", "stage1_ml", "processed", "oncology_cleaned.csv")
    REPORT_PATH = os.path.join("data", "stage1_ml", "processed", "cleaning_report.json")
    
    # Adjust paths if script is run from inside stage1_ml/data directory
    if not os.path.exists(RAW_PATH):
        # We might be running from inside stage1_ml/data
        # Let's use absolute paths from project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        RAW_PATH = os.path.join(project_root, "data", "stage1_ml", "raw", "oncology_raw_5000_34_features.csv")
        PROCESSED_PATH = os.path.join(project_root, "data", "stage1_ml", "processed", "oncology_cleaned.csv")
        REPORT_PATH = os.path.join(project_root, "data", "stage1_ml", "processed", "cleaning_report.json")
        
    try:
        run_cleaning_pipeline(RAW_PATH, PROCESSED_PATH, REPORT_PATH)
        print("Data cleaning completed successfully.")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        sys.exit(1)
