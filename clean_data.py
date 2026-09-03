import pandas as pd
import numpy as np

df = pd.read_csv("/mnt/user-data/outputs/oncology_synthetic_dataset.csv")
report = []
report.append(f"Starting rows: {len(df)}")

# ---- 1. Standardize placeholder junk values to real NaN ----
placeholders = ["", "NA", "N/A", "Unknown", "-", "unknown", "na", "n/a", "None", "none"]
df.replace(placeholders, np.nan, inplace=True)
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
# re-blank any strings that became empty after stripping
df.replace("", np.nan, inplace=True)

# ---- 2. Remove exact duplicate rows ----
before = len(df)
df = df.drop_duplicates()
report.append(f"Exact duplicates removed: {before - len(df)}")

# ---- 3. Resolve near-duplicate patient_ids (same ID, multiple records) ----
before = len(df)
# Keep the record with the fewest missing values for each patient_id
df["_missing_count"] = df.isna().sum(axis=1)
df = df.sort_values("_missing_count").drop_duplicates(subset="patient_id", keep="first")
df = df.drop(columns="_missing_count")
report.append(f"Near-duplicate patient_id records collapsed: {before - len(df)}")

# ---- 4. Standardize categorical text formatting ----
cat_cols = ["gender", "cancer_type", "cancer_stage", "smoking_status", "alcohol_use",
            "family_history", "treatment_type", "survival_status", "risk_level"]
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.title().replace("Nan", np.nan)

# ---- 5. Handle missing values ----
# Numeric columns -> impute with median
num_cols = ["age", "tumor_size_cm", "bmi", "hospital_visits_per_year", "followup_months"]
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    df[col] = df[col].round(1)

# diagnosis_date -> can't sensibly impute a date, mark as "Unknown"
df["diagnosis_date"] = df["diagnosis_date"].fillna("Unknown")

# Categorical columns -> impute with mode (most frequent) except risk_level/survival_status
for col in ["gender", "cancer_type", "cancer_stage", "smoking_status", "alcohol_use",
            "family_history", "treatment_type"]:
    mode_val = df[col].mode(dropna=True)[0]
    df[col] = df[col].fillna(mode_val)

# risk_level & survival_status are outcome-sensitive -> fill with explicit "Not Recorded"
# rather than guessing, to avoid biasing analysis
df["risk_level"] = df["risk_level"].fillna("Not Recorded")
df["survival_status"] = df["survival_status"].fillna("Not Recorded")

# ---- 6. Fix data type consistency ----
df["age"] = df["age"].astype(int)
df["hospital_visits_per_year"] = df["hospital_visits_per_year"].astype(int)
df["followup_months"] = df["followup_months"].astype(int)

# ---- 7. Sanity bounds check (clip impossible values) ----
df["age"] = df["age"].clip(0, 100)
df["bmi"] = df["bmi"].clip(10, 60)
df["tumor_size_cm"] = df["tumor_size_cm"].clip(0, 25)

report.append(f"Final rows: {len(df)}")
report.append(f"Remaining missing values total: {df.isna().sum().sum()}")

df.to_csv("/mnt/user-data/outputs/oncology_cleaned_dataset.csv", index=False)

print("\n".join(report))
print("\nMissing values per column after cleaning:\n", df.isna().sum())
print("\nDtypes:\n", df.dtypes)
