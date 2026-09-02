import pandas as pd
import numpy as np

def clean_oncology_dataset(input_csv="oncology_synthetic_dataset.csv", output_csv="oncology_cleaned_dataset.csv"):
    # 1. Load Dataset
    df = pd.read_csv(input_csv)
    initial_rows = len(df)
    initial_nulls = df.isnull().sum().sum()
    
    print("=" * 50)
    print(f"RAW DATASET SUMMARY: {initial_rows} rows | {initial_nulls} total missing values")
    print("=" * 50)
    
    # 2. Deduplication
    duplicates_count = df.duplicated().sum()
    df_clean = df.drop_duplicates().copy().reset_index(drop=True)
    print(f"[Step 1] Removed {duplicates_count} exact duplicate rows. Remaining rows: {len(df_clean)}")
    
    # 3. Handle Missing Numerical Values
    # Age: Impute with median age (rounded to integer)
    median_age = int(round(df_clean['age'].median()))
    df_clean['age'] = df_clean['age'].fillna(median_age).astype(int)
    
    # Tumor Size: Impute by stage median, fallback to global median
    stage_size_medians = df_clean.groupby('tumor_stage')['tumor_size_cm'].transform('median')
    global_size_median = round(df_clean['tumor_size_cm'].median(), 1)
    df_clean['tumor_size_cm'] = df_clean['tumor_size_cm'].fillna(stage_size_medians).fillna(global_size_median).round(1)
    
    # Survival Months: Impute by risk level / stage median, fallback to global median
    risk_survival_medians = df_clean.groupby('level_of_risk')['survival_months'].transform('median')
    global_survival_median = int(round(df_clean['survival_months'].median()))
    df_clean['survival_months'] = df_clean['survival_months'].fillna(risk_survival_medians).fillna(global_survival_median).astype(int)
    
    # 4. Handle Missing Categorical Values
    # Gender: Mode imputation
    gender_mode = df_clean['gender'].mode()[0] if not df_clean['gender'].mode().empty else 'Unknown'
    df_clean['gender'] = df_clean['gender'].fillna(gender_mode)
    
    # Cancer Type: Mode imputation
    cancer_mode = df_clean['cancer_type'].mode()[0] if not df_clean['cancer_type'].mode().empty else 'Unspecified'
    df_clean['cancer_type'] = df_clean['cancer_type'].fillna(cancer_mode)
    
    # Tumor Stage: Infer from tumor size or fill with mode
    def infer_stage(row):
        if pd.isna(row['tumor_stage']):
            size = row['tumor_size_cm']
            if size <= 2.5:
                return 'Stage I'
            elif size <= 4.5:
                return 'Stage II'
            elif size <= 7.0:
                return 'Stage III'
            else:
                return 'Stage IV'
        return row['tumor_stage']
        
    df_clean['tumor_stage'] = df_clean.apply(infer_stage, axis=1)
    
    # Biomarker Status: Fill missing with 'Not Tested / Marker Unknown'
    df_clean['biomarker_status'] = df_clean['biomarker_status'].fillna('Not Tested')
    
    # Treatment Regimen: Fill with 'Standard Chemotherapy' or mode
    treatment_mode = df_clean['treatment_regimen'].mode()[0] if not df_clean['treatment_regimen'].mode().empty else 'Standard Care'
    df_clean['treatment_regimen'] = df_clean['treatment_regimen'].fillna(treatment_mode)
    
    # Level of Risk: Infer from stage if missing
    def infer_risk(row):
        if pd.isna(row['level_of_risk']) or row['level_of_risk'] == '':
            stage = row['tumor_stage']
            if stage == 'Stage I': return 'Low'
            elif stage == 'Stage II': return 'Moderate'
            elif stage == 'Stage III': return 'High'
            else: return 'Critical'
        return row['level_of_risk']
        
    df_clean['level_of_risk'] = df_clean.apply(infer_risk, axis=1)
    
    # 5. Final Quality Audit
    remaining_nulls = df_clean.isnull().sum().sum()
    final_rows = len(df_clean)
    
    print("\n" + "=" * 50)
    print(f"CLEANED DATASET SUMMARY: {final_rows} rows | {remaining_nulls} total missing values")
    print("=" * 50)
    
    # 6. Save Cleaned Dataset
    df_clean.to_csv(output_csv, index=False)
    print(f"Cleaned dataset saved successfully to: '{output_csv}'")
    return df_clean

if __name__ == "__main__":
    clean_oncology_dataset()
