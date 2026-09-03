# ============================================================
# EDA ENGINEER - ONCOLOGY DATASET
# Purpose:
# 1. Understand the dataset
# 2. Explore clinical patterns
# 3. Identify suspicious records
# 4. Check possible data leakage
# 5. Prepare data for ML Engineer
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("data", exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

file_path = r"C:\Users\ADMIN\Documents\ONCOLOGY\oncology_cleaned_dataset_final"

df = pd.read_csv(file_path)

print("=" * 70)
print("EDA ENGINEER - DATASET OVERVIEW")
print("=" * 70)

print("Number of rows and columns:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())


# ------------------------------------------------------------
# 2. CHECK DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ------------------------------------------------------------
# 3. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

print("\nTotal missing values:", df.isnull().sum().sum())


# ------------------------------------------------------------
# 4. CHECK DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# ------------------------------------------------------------
# 5. CHECK DUPLICATE PATIENT IDs
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PATIENT ID CHECK")
print("=" * 70)

duplicate_ids = df["patient_id"].duplicated().sum()

print("Duplicate patient IDs:", duplicate_ids)


# ------------------------------------------------------------
# 6. NUMERICAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

numeric_columns = [
    "age",
    "tumor_size_cm",
    "bmi",
    "hospital_visits_per_year",
    "followup_months"
]

print(df[numeric_columns].describe().round(2))


# ------------------------------------------------------------
# 7. CHECK RISK LEVEL DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RISK LEVEL DISTRIBUTION")
print("=" * 70)

risk_counts = df["risk_level"].value_counts()

print(risk_counts)

print("\nRisk percentages:")
print(
    (df["risk_level"].value_counts(normalize=True) * 100)
    .round(2)
)


# ------------------------------------------------------------
# 8. PLOT RISK LEVEL DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="risk_level"
)

plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("risk_level_distribution.png")
plt.show()


# ------------------------------------------------------------
# 9. NUMERICAL FEATURES VS RISK LEVEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLINICAL FEATURES BY RISK LEVEL")
print("=" * 70)

risk_summary = (
    df.groupby("risk_level")[numeric_columns]
    .mean()
    .round(2)
)

print(risk_summary)


# ------------------------------------------------------------
# 10. AGE VS RISK
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="risk_level",
    y="age"
)

plt.title("Age Distribution by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Age")

plt.tight_layout()
plt.savefig("data/age_by_risk.png")
plt.show()


# ------------------------------------------------------------
# 11. TUMOR SIZE VS RISK
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="risk_level",
    y="tumor_size_cm"
)

plt.title("Tumor Size by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Tumor Size (cm)")

plt.tight_layout()
plt.savefig("data/tumor_size_by_risk.png")
plt.show()


# ------------------------------------------------------------
# 12. BMI VS RISK
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="risk_level",
    y="bmi"
)

plt.title("BMI Distribution by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("BMI")

plt.tight_layout()
plt.savefig("data/bmi_by_risk.png")
plt.show()


# ------------------------------------------------------------
# 13. HOSPITAL VISITS VS RISK
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="risk_level",
    y="hospital_visits_per_year"
)

plt.title("Hospital Visits per Year by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Hospital Visits per Year")

plt.tight_layout()
plt.savefig("data/hospital_visits_by_risk.png")
plt.show()


# ------------------------------------------------------------
# 14. CANCER STAGE VS RISK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CANCER STAGE VS RISK")
print("=" * 70)

stage_risk = pd.crosstab(
    df["cancer_stage"],
    df["risk_level"]
)

print(stage_risk)


plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="cancer_stage",
    hue="risk_level"
)

plt.title("Cancer Stage Distribution by Risk Level")
plt.xlabel("Cancer Stage")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("data/cancer_stage_vs_risk.png")
plt.show()


# ------------------------------------------------------------
# 15. CANCER TYPE VS RISK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CANCER TYPE VS RISK")
print("=" * 70)

cancer_risk = pd.crosstab(
    df["cancer_type"],
    df["risk_level"]
)

print(cancer_risk)


# ------------------------------------------------------------
# 16. SMOKING STATUS VS RISK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SMOKING STATUS VS RISK")
print("=" * 70)

smoking_risk = pd.crosstab(
    df["smoking_status"],
    df["risk_level"]
)

print(smoking_risk)


plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="smoking_status",
    hue="risk_level"
)

plt.title("Smoking Status by Risk Level")
plt.xlabel("Smoking Status")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("data/smoking_vs_risk.png")
plt.show()


# ------------------------------------------------------------
# 17. ALCOHOL USE VS RISK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ALCOHOL USE VS RISK")
print("=" * 70)

alcohol_risk = pd.crosstab(
    df["alcohol_use"],
    df["risk_level"]
)

print(alcohol_risk)


# ------------------------------------------------------------
# 18. FAMILY HISTORY VS RISK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FAMILY HISTORY VS RISK")
print("=" * 70)

family_history_risk = pd.crosstab(
    df["family_history"],
    df["risk_level"]
)

print(family_history_risk)


# ------------------------------------------------------------
# 19. TREATMENT TYPE VS RISK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TREATMENT TYPE VS RISK")
print("=" * 70)

treatment_risk = pd.crosstab(
    df["treatment_type"],
    df["risk_level"]
)

print(treatment_risk)


# ------------------------------------------------------------
# 20. SURVIVAL STATUS VS RISK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SURVIVAL STATUS VS RISK")
print("=" * 70)

survival_risk = pd.crosstab(
    df["survival_status"],
    df["risk_level"]
)

print(survival_risk)


# ------------------------------------------------------------
# 21. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERICAL CORRELATION")
print("=" * 70)

correlation = df[numeric_columns].corr()

print(correlation.round(2))


# ------------------------------------------------------------
# 22. CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Numerical Feature Correlation")
plt.tight_layout()

plt.savefig("data/correlation_heatmap.png")
plt.show()


# ------------------------------------------------------------
# 23. OUTLIER CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OUTLIER CHECK")
print("=" * 70)

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(
        f"{column}: {len(outliers)} potential outliers"
    )


# ------------------------------------------------------------
# 24. FIND SUSPICIOUS CLINICAL RECORDS
# ------------------------------------------------------------
#
# These are FLAGS for investigation.
# They are NOT automatically considered incorrect.
#
# Example:
# Very small tumor + advanced cancer stage
# ------------------------------------------------------------

suspicious_records = df[
    (
        (df["tumor_size_cm"] < 1) &
        (df["cancer_stage"].isin(["Stage III", "Stage IV"]))
    )
].copy()


print("\n" + "=" * 70)
print("SUSPICIOUS CLINICAL RECORDS")
print("=" * 70)

print(
    "Number of potentially suspicious records:",
    len(suspicious_records)
)

print("\nSuspicious records:")
print(suspicious_records.head(20))


# ------------------------------------------------------------
# 25. SAVE SUSPICIOUS RECORDS
# ------------------------------------------------------------

suspicious_records.to_csv(
    "data/suspicious_records.csv",
    index=False
)

print("\nSuspicious records saved to:")
print("data/suspicious_records.csv")


# ------------------------------------------------------------
# 26. POSSIBLE DATA LEAKAGE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("POSSIBLE DATA LEAKAGE CHECK")
print("=" * 70)

possible_leakage_columns = [
    "survival_status",
    "followup_months",
    "hospital_visits_per_year",
    "treatment_type"
]

for column in possible_leakage_columns:

    if column in df.columns:

        print(
            f"WARNING: {column} should be reviewed "
            "before ML training."
        )

print(
    "\nThese variables may contain information "
    "that would not be available at the time "
    "the risk prediction is made."
)


# ------------------------------------------------------------
# 27. CHECK EXTREME VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXTREME VALUE CHECK")
print("=" * 70)

for column in numeric_columns:

    print(f"\n{column}")

    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())


# ------------------------------------------------------------
# 28. SAVE EDA-CHECKED DATASET
# ------------------------------------------------------------

# Get the folder where Eda_engineer.py is located
main_folder = os.path.dirname(os.path.abspath(__file__))

# Save directly in ONCOLOGY main folder
output_path = os.path.join(
    main_folder,
    "eda_checked_dataset.csv"
)

# Save EDA checked dataset
df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nEDA checked dataset saved successfully!")
print("\nFile location:")
print(output_path)

print("\nDataset shape:", df.shape)

