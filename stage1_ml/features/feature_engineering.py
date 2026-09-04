import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

def create_engineered_features(df):
    """
    Create medically meaningful oncology-related derived features.
    """
    df_feat = df.copy()
    
    # 1. Age groups
    if 'age' in df_feat.columns:
        conditions = [
            (df_feat['age'] < 50),
            (df_feat['age'] >= 50) & (df_feat['age'] <= 65),
            (df_feat['age'] > 65)
        ]
        choices = ['<50', '50-65', '>65']
        df_feat['age_group'] = np.select(conditions, choices, default='unknown')
        
    # 2. BMI categories
    if 'bmi' in df_feat.columns:
        conditions = [
            (df_feat['bmi'] < 18.5),
            (df_feat['bmi'] >= 18.5) & (df_feat['bmi'] < 25),
            (df_feat['bmi'] >= 25) & (df_feat['bmi'] < 30),
            (df_feat['bmi'] >= 30)
        ]
        choices = ['underweight', 'normal', 'overweight', 'obese']
        df_feat['bmi_category'] = np.select(conditions, choices, default='unknown')
        
    # 3. Tumor size categories (assuming cm)
    if 'tumor_size' in df_feat.columns:
        conditions = [
            (df_feat['tumor_size'] <= 2.0),
            (df_feat['tumor_size'] > 2.0) & (df_feat['tumor_size'] <= 5.0),
            (df_feat['tumor_size'] > 5.0)
        ]
        choices = ['T1_small', 'T2_medium', 'T3_large']
        df_feat['tumor_size_category'] = np.select(conditions, choices, default='unknown')
        
    # 4. Treatment intensity features
    if 'treatment_dose' in df_feat.columns and 'treatment_duration' in df_feat.columns:
        # Avoid division by zero
        duration = df_feat['treatment_duration'].replace(0, np.nan)
        df_feat['treatment_intensity'] = df_feat['treatment_dose'] / duration
        
    # 5. Clinical risk indicators (e.g. high comorbidity and poor performance status)
    if 'comorbidity_score' in df_feat.columns and 'performance_status' in df_feat.columns:
        df_feat['high_clinical_risk'] = ((df_feat['comorbidity_score'] > df_feat['comorbidity_score'].median()) & 
                                         (df_feat['performance_status'] > df_feat['performance_status'].median())).astype(int)
                                         
    # 6. Biomarker combination (interaction term)
    if 'biomarker_1' in df_feat.columns and 'biomarker_2' in df_feat.columns:
        df_feat['biomarker_interaction'] = df_feat['biomarker_1'] * df_feat['biomarker_2']
        
    return df_feat

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "stage1_ml", "processed", "oncology_cleaned.csv")
    out_dir = os.path.join(base_dir, "data", "stage1_ml", "features")
    docs_dir = os.path.join(base_dir, "docs")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    print(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    targets = ['overall_patient_risk', 'toxicity_risk', 'therapy_response']
    original_feature_count = df.shape[1] - len(targets)
    
    X = df.drop(columns=targets)
    y = df[targets]
    
    # Generate stratification label using primary target overall_patient_risk
    stratify_col = y['overall_patient_risk'].astype(str)
    
    print("Splitting dataset into Train (70%), Val (15%), Test (15%)...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=stratify_col, random_state=42)
    
    stratify_col_temp = y_temp['overall_patient_risk'].astype(str)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=stratify_col_temp, random_state=42) # 0.15 / 0.85 approx 0.1765
    
    # 1. Feature Engineering (Derived features)
    print("Applying medically meaningful feature engineering...")
    X_train_eng = create_engineered_features(X_train)
    X_val_eng = create_engineered_features(X_val)
    X_test_eng = create_engineered_features(X_test)
    
    # Identify numerical and categorical columns after engineering
    numerical_cols = X_train_eng.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train_eng.select_dtypes(exclude=[np.number]).columns.tolist()
    
    engineered_features_list = [c for c in X_train_eng.columns if c not in X.columns]
    print(f"Engineered {len(engineered_features_list)} new features: {engineered_features_list}")
    
    # 2. Missing Value Handling (Numerical)
    print("Imputing missing numerical values using median (fit on train only)...")
    num_imputer = SimpleImputer(strategy='median')
    X_train_eng[numerical_cols] = num_imputer.fit_transform(X_train_eng[numerical_cols])
    X_val_eng[numerical_cols] = num_imputer.transform(X_val_eng[numerical_cols])
    X_test_eng[numerical_cols] = num_imputer.transform(X_test_eng[numerical_cols])
    
    # 3. Categorical Feature Processing
    print("Encoding categorical features using OneHotEncoder...")
    cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
    X_train_eng[categorical_cols] = cat_imputer.fit_transform(X_train_eng[categorical_cols])
    X_val_eng[categorical_cols] = cat_imputer.transform(X_val_eng[categorical_cols])
    X_test_eng[categorical_cols] = cat_imputer.transform(X_test_eng[categorical_cols])
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat_encoded = encoder.fit_transform(X_train_eng[categorical_cols])
    X_val_cat_encoded = encoder.transform(X_val_eng[categorical_cols])
    X_test_cat_encoded = encoder.transform(X_test_eng[categorical_cols])
    
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)
    
    X_train_cat_df = pd.DataFrame(X_train_cat_encoded, columns=cat_feature_names, index=X_train_eng.index)
    X_val_cat_df = pd.DataFrame(X_val_cat_encoded, columns=cat_feature_names, index=X_val_eng.index)
    X_test_cat_df = pd.DataFrame(X_test_cat_encoded, columns=cat_feature_names, index=X_test_eng.index)
    
    # 4. Numerical Scaling
    print("Scaling numerical features using StandardScaler...")
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_eng[numerical_cols])
    X_val_num_scaled = scaler.transform(X_val_eng[numerical_cols])
    X_test_num_scaled = scaler.transform(X_test_eng[numerical_cols])
    
    X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=numerical_cols, index=X_train_eng.index)
    X_val_num_df = pd.DataFrame(X_val_num_scaled, columns=numerical_cols, index=X_val_eng.index)
    X_test_num_df = pd.DataFrame(X_test_num_scaled, columns=numerical_cols, index=X_test_eng.index)
    
    # Combine
    X_train_processed = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
    X_val_processed = pd.concat([X_val_num_df, X_val_cat_df], axis=1)
    X_test_processed = pd.concat([X_test_num_df, X_test_cat_df], axis=1)
    
    # 5. Feature Selection (Variance Threshold and Correlation)
    print("Performing Variance check...")
    var_thresh = VarianceThreshold(threshold=0.01)
    var_thresh.fit(X_train_processed)
    
    kept_features_var = X_train_processed.columns[var_thresh.get_support()]
    removed_by_var = set(X_train_processed.columns) - set(kept_features_var)
    
    X_train_processed = X_train_processed[kept_features_var]
    X_val_processed = X_val_processed[kept_features_var]
    X_test_processed = X_test_processed[kept_features_var]
    
    print("Performing Correlation check (>0.90)...")
    corr_matrix = X_train_processed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    
    X_train_processed = X_train_processed.drop(columns=to_drop)
    X_val_processed = X_val_processed.drop(columns=to_drop)
    X_test_processed = X_test_processed.drop(columns=to_drop)
    
    final_features = X_train_processed.columns.tolist()
    final_feature_count = len(final_features)
    
    # Attach targets back to save the datasets completely
    train_out = pd.concat([X_train_processed, y_train], axis=1)
    val_out = pd.concat([X_val_processed, y_val], axis=1)
    test_out = pd.concat([X_test_processed, y_test], axis=1)
    
    train_out['dataset_split'] = 'train'
    val_out['dataset_split'] = 'val'
    test_out['dataset_split'] = 'test'
    full_processed = pd.concat([train_out, val_out, test_out], axis=0)
    
    processed_features_path = os.path.join(out_dir, "processed_features.csv")
    print(f"Saving processed datasets to {processed_features_path}...")
    full_processed.to_csv(processed_features_path, index=False)
    
    feature_names_path = os.path.join(out_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump({"features": final_features, "targets": targets}, f, indent=4)
        
    report = {
        "original_feature_count": original_feature_count,
        "final_feature_count": final_feature_count,
        "features_created": engineered_features_list,
        "features_removed": {
            "by_variance": list(removed_by_var),
            "by_correlation": to_drop
        },
        "imputation": "Median for numerical, Constant ('unknown') for categorical",
        "scaling": "StandardScaler",
        "encoding": "OneHotEncoder",
        "dataset_splits": {
            "train_size": len(X_train_processed),
            "val_size": len(X_val_processed),
            "test_size": len(X_test_processed)
        }
    }
    
    report_path = os.path.join(out_dir, "feature_engineering_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    md_content = f"""# Stage 1 ML — Feature Engineering

## 1. Objective
Feature engineering transforms raw data into formats suitable for Machine Learning models while enriching it with domain-specific derived metrics.

## 2. Feature Transformations

### Original Features
The dataset began with {original_feature_count} original predictor features.

### Engineered Features
We created {len(engineered_features_list)} medically meaningful features:
"""
    for ef in engineered_features_list:
        md_content += f"- **{ef}**\n"
        
    md_content += f"""
### Missing Value Handling
- **Numerical**: Imputed using `Median` strategy. Median is robust to extreme clinical outliers.
- **Categorical**: Filled with `'unknown'`.
- **Data Leakage Prevention**: All imputers were fitted **exclusively on the training set**. The validation and test sets were transformed using the learned training distributions.

### Encoding Methods
- Categorical features were encoded using `OneHotEncoder`. This converts string labels into binary vectors.

### Scaling Methods
- Numerical features were standardized using `StandardScaler` (zero mean, unit variance).

### Feature Selection
- **Variance Check**: Removed {len(removed_by_var)} features with near-zero variance.
- **Correlation Check**: Removed {len(to_drop)} highly correlated redundant features (Pearson correlation > 0.90) to prevent multicollinearity.

## 3. Summary
- **Before**: {original_feature_count} features
- **After**: {final_feature_count} processed features (including One-Hot encoded categories)
"""

    md_report_path = os.path.join(docs_dir, "stage1_ml_feature_engineering_report.md")
    with open(md_report_path, 'w') as f:
        f.write(md_content)
        
    print("\n------------------------------")
    print("FEATURE ENGINEERING SUMMARY:")
    print(f"Original feature count: {original_feature_count}")
    print(f"Final feature count: {final_feature_count}")
    print(f"Features created: {len(engineered_features_list)}")
    print(f"Features removed: {len(removed_by_var) + len(to_drop)}")
    print("------------------------------")
    
if __name__ == "__main__":
    main()
