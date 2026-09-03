"""
clean_oncology_data.py

Cleans the raw oncology dataset (oncology_unclean_dataset.csv) and produces
a tidy, analysis-ready dataset equivalent to oncology_cleaned_dataset.csv.

Pipeline
--------
1.  Drop the `diagnosis_date` column (not present in the cleaned dataset).
2.  Remove exact duplicate rows.
3.  Remove duplicate `patient_id` records, keeping the first occurrence.
4.  Treat placeholder strings ("-" and "Unknown") as missing values.
5.  Drop rows where the target/outcome columns (`survival_status`,
    `risk_level`) are missing -- these can't be reliably imputed.
6.  Impute remaining missing values:
       - `gender`            -> filled with "Other"
       - other categoricals  -> filled with each column's mode
       - numeric columns     -> filled with each column's median
7.  Standardize `cancer_stage` text to uppercase (e.g. "Stage II" -> "STAGE II").
8.  Cast `age`, `hospital_visits_per_year`, and `followup_months` to integers.
9.  Save the result to oncology_cleaned_dataset.csv.

Usage
-----
    python clean_oncology_data.py \
        --input oncology_unclean_dataset.csv \
        --output oncology_cleaned_dataset.csv
"""

import argparse
import numpy as np
import pandas as pd

# Columns that identify the outcome of a patient record; if these are
# missing there isn't a reliable way to impute them, so such rows are dropped.
TARGET_COLUMNS = ["survival_status", "risk_level"]

# Categorical columns imputed using the column's most frequent value (mode).
MODE_IMPUTE_COLUMNS = [
    "cancer_type",
    "cancer_stage",
    "smoking_status",
    "alcohol_use",
    "family_history",
    "treatment_type",
]

# Numeric columns imputed using the column's median value.
MEDIAN_IMPUTE_COLUMNS = [
    "age",
    "tumor_size_cm",
    "bmi",
    "hospital_visits_per_year",
    "followup_months",
]

# Numeric columns that should end up as whole numbers (integers).
INTEGER_COLUMNS = ["age", "hospital_visits_per_year", "followup_months"]

# Placeholder strings in the raw data that really mean "missing".
MISSING_PLACEHOLDERS = ["-", "Unknown"]


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(path)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that aren't part of the cleaned schema."""
    return df.drop(columns=["diagnosis_date"], errors="ignore")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, then duplicate patient records."""
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="patient_id", keep="first")
    return df


def mark_placeholders_as_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Convert placeholder strings like '-' and 'Unknown' into real NaNs."""
    return df.replace(MISSING_PLACEHOLDERS, np.nan)


def drop_rows_missing_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where key outcome columns are missing."""
    return df.dropna(subset=TARGET_COLUMNS)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining missing values using sensible defaults per column."""
    df = df.copy()

    # Gender: fill with an explicit "Other" category rather than the mode,
    # to avoid overrepresenting the majority gender.
    df["gender"] = df["gender"].fillna("Other")

    # Other categoricals: fill with the most common category.
    for col in MODE_IMPUTE_COLUMNS:
        mode_value = df[col].mode(dropna=True).iloc[0]
        df[col] = df[col].fillna(mode_value)

    # Numeric columns: fill with the median, rounded to a sensible precision.
    for col in MEDIAN_IMPUTE_COLUMNS:
        median_value = df[col].median()
        if col in INTEGER_COLUMNS:
            df[col] = df[col].fillna(round(median_value))
        else:
            df[col] = df[col].fillna(round(median_value, 1))

    return df


def standardize_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text formatting, e.g. cancer_stage -> uppercase."""
    df = df.copy()
    df["cancer_stage"] = df["cancer_stage"].str.upper()
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Round and cast numeric columns that should be whole numbers."""
    df = df.copy()
    for col in INTEGER_COLUMNS:
        df[col] = df[col].round().astype(int)
    df["tumor_size_cm"] = df["tumor_size_cm"].round(1)
    df["bmi"] = df["bmi"].round(1)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on the raw dataframe."""
    df = drop_unused_columns(df)
    df = remove_duplicates(df)
    df = mark_placeholders_as_missing(df)
    df = drop_rows_missing_targets(df)
    df = impute_missing_values(df)
    df = standardize_values(df)
    df = fix_dtypes(df)
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Clean the oncology dataset.")
    parser.add_argument(
        "--input",
        default="oncology_unclean_dataset.csv",
        help="Path to the raw/unclean CSV file.",
    )
    parser.add_argument(
        "--output",
        default="oncology_cleaned_dataset.csv",
        help="Path to write the cleaned CSV file.",
    )
    args = parser.parse_args()

    raw_df = load_data(args.input)
    print(f"Loaded raw data: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    cleaned_df = clean_dataset(raw_df)
    print(f"Cleaned data: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
    print(f"Missing values remaining: {cleaned_df.isna().sum().sum()}")

    cleaned_df.to_csv(args.output, index=False)
    print(f"Saved cleaned dataset to: {args.output}")


if __name__ == "__main__":
    main()
