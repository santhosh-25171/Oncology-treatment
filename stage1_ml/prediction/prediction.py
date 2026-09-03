import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

class OncologyPredictionPipeline:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = base_dir
            
        self.models_dir = os.path.join(self.base_dir, "data", "stage1_ml", "models")
        self.tuning_dir = os.path.join(self.models_dir, "tuning")
        
        self.tox_model = None
        self.ther_model = None
        self.tox_encoder = None
        self.ther_encoder = None
        
        self.explainer_tox = None
        self.explainer_ther = None
        
        self.num_imputer = None
        self.cat_imputer = None
        self.encoder = None
        self.scaler = None
        self.var_thresh = None
        self.features_to_drop = []
        self.final_feature_names = []
        self.original_categorical_cols = []
        self.original_numerical_cols = []
        
        self.load_models()
        self._fit_preprocessing()
        self._init_explainers()
        
    def load_models(self):
        """Load tuned models and target label encoders"""
        tox_path = os.path.join(self.tuning_dir, "tuned_toxicity_model.joblib")
        ther_path = os.path.join(self.tuning_dir, "tuned_therapy_response_model.joblib")
        
        self.tox_model = joblib.load(tox_path)
        self.ther_model = joblib.load(ther_path)
        
        self.tox_encoder = joblib.load(os.path.join(self.models_dir, "toxicity_risk_label_encoder.joblib"))
        self.ther_encoder = joblib.load(os.path.join(self.models_dir, "therapy_response_label_encoder.joblib"))
        
    def _fit_preprocessing(self):
        """
        Recreates and fits the preprocessing pipeline on the original training data
        to ensure identical transformations for new patient data.
        """
        data_path = os.path.join(self.base_dir, "data", "stage1_ml", "processed", "oncology_cleaned.csv")
        df = pd.read_csv(data_path)
        
        targets = ['toxicity_risk', 'therapy_response']
        X = df.drop(columns=targets)
        y = df[targets]
        
        stratify_col = y['toxicity_risk'].astype(str) + "_" + y['therapy_response'].astype(str)
        if stratify_col.value_counts().min() < 2:
            stratify_col = y['toxicity_risk']
            
        X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.15, stratify=stratify_col, random_state=42)
        
        stratify_col_temp = y_temp['toxicity_risk'].astype(str) + "_" + y_temp['therapy_response'].astype(str)
        if stratify_col_temp.value_counts().min() < 2:
            stratify_col_temp = y_temp['toxicity_risk']
            
        X_train, _, _, _ = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=stratify_col_temp, random_state=42)
        
        # 1. Engineering
        X_train_eng = self._apply_feature_engineering(X_train)
        
        self.original_numerical_cols = X_train_eng.select_dtypes(include=[np.number]).columns.tolist()
        self.original_categorical_cols = X_train_eng.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # 2. Imputation
        self.num_imputer = SimpleImputer(strategy='median')
        X_train_eng[self.original_numerical_cols] = self.num_imputer.fit_transform(X_train_eng[self.original_numerical_cols])
        
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        X_train_eng[self.original_categorical_cols] = self.cat_imputer.fit_transform(X_train_eng[self.original_categorical_cols])
        
        # 3. Encoding
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat_encoded = self.encoder.fit_transform(X_train_eng[self.original_categorical_cols])
        cat_feature_names = self.encoder.get_feature_names_out(self.original_categorical_cols)
        X_train_cat_df = pd.DataFrame(X_train_cat_encoded, columns=cat_feature_names, index=X_train_eng.index)
        
        # 4. Scaling
        self.scaler = StandardScaler()
        X_train_num_scaled = self.scaler.fit_transform(X_train_eng[self.original_numerical_cols])
        X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=self.original_numerical_cols, index=X_train_eng.index)
        
        X_train_processed = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
        
        # 5. Feature Selection
        self.var_thresh = VarianceThreshold(threshold=0.01)
        self.var_thresh.fit(X_train_processed)
        kept_features_var = X_train_processed.columns[self.var_thresh.get_support()]
        X_train_processed = X_train_processed[kept_features_var]
        
        corr_matrix = X_train_processed.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.features_to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        X_train_processed = X_train_processed.drop(columns=self.features_to_drop)
        
        X_train_processed.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in X_train_processed.columns]
        self.final_feature_names = X_train_processed.columns.tolist()
        
        # Keep background data for SHAP explainer (LogReg requires data for Independent masker)
        self.background_data = X_train_processed.sample(n=min(100, len(X_train_processed)), random_state=42)

    def _init_explainers(self):
        # We know from tuning that XGBoost won both targets.
        # TreeExplainer is best for XGBoost
        self.explainer_tox = shap.TreeExplainer(self.tox_model)
        self.explainer_ther = shap.TreeExplainer(self.ther_model)
        
    def _apply_feature_engineering(self, df):
        df_feat = df.copy()
        if 'age' in df_feat.columns:
            conditions = [(df_feat['age'] < 50), (df_feat['age'] >= 50) & (df_feat['age'] <= 65), (df_feat['age'] > 65)]
            df_feat['age_group'] = np.select(conditions, ['<50', '50-65', '>65'], default='unknown')
        if 'bmi' in df_feat.columns:
            conditions = [(df_feat['bmi'] < 18.5), (df_feat['bmi'] >= 18.5) & (df_feat['bmi'] < 25),
                          (df_feat['bmi'] >= 25) & (df_feat['bmi'] < 30), (df_feat['bmi'] >= 30)]
            df_feat['bmi_category'] = np.select(conditions, ['underweight', 'normal', 'overweight', 'obese'], default='unknown')
        if 'tumor_size' in df_feat.columns:
            conditions = [(df_feat['tumor_size'] <= 2.0), (df_feat['tumor_size'] > 2.0) & (df_feat['tumor_size'] <= 5.0), (df_feat['tumor_size'] > 5.0)]
            df_feat['tumor_size_category'] = np.select(conditions, ['T1_small', 'T2_medium', 'T3_large'], default='unknown')
        if 'treatment_dose' in df_feat.columns and 'treatment_duration' in df_feat.columns:
            duration = df_feat['treatment_duration'].replace(0, np.nan)
            df_feat['treatment_intensity'] = df_feat['treatment_dose'] / duration
        if 'comorbidity_score' in df_feat.columns and 'performance_status' in df_feat.columns:
            df_feat['high_clinical_risk'] = ((df_feat['comorbidity_score'] > 3) & (df_feat['performance_status'] > 1)).astype(int) # approx medians
        if 'biomarker_1' in df_feat.columns and 'biomarker_2' in df_feat.columns:
            df_feat['biomarker_interaction'] = df_feat['biomarker_1'] * df_feat['biomarker_2']
        return df_feat
        
    def _preprocess_input(self, df):
        df_eng = self._apply_feature_engineering(df)
        
        # Ensure all columns exist (fill missing with nan for imputer)
        for c in self.original_numerical_cols + self.original_categorical_cols:
            if c not in df_eng.columns:
                df_eng[c] = np.nan
                
        df_eng[self.original_numerical_cols] = self.num_imputer.transform(df_eng[self.original_numerical_cols])
        df_eng[self.original_categorical_cols] = self.cat_imputer.transform(df_eng[self.original_categorical_cols])
        
        cat_encoded = self.encoder.transform(df_eng[self.original_categorical_cols])
        cat_feature_names = self.encoder.get_feature_names_out(self.original_categorical_cols)
        df_cat = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df_eng.index)
        
        num_scaled = self.scaler.transform(df_eng[self.original_numerical_cols])
        df_num = pd.DataFrame(num_scaled, columns=self.original_numerical_cols, index=df_eng.index)
        
        df_processed = pd.concat([df_num, df_cat], axis=1)
        
        kept_features_var = df_processed.columns[self.var_thresh.get_support()]
        df_processed = df_processed[kept_features_var]
        
        df_processed = df_processed.drop(columns=[c for c in self.features_to_drop if c in df_processed.columns])
        df_processed.columns = [c.replace('[', '').replace(']', '').replace('<', 'lt_').replace('>', 'gt_') for c in df_processed.columns]
        
        # Ensure columns exactly match final_feature_names (add missing as 0)
        for col in self.final_feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.final_feature_names]
        
        return df_processed
        
    def predict(self, patient_dict):
        df = pd.DataFrame([patient_dict])
        
        # Validation checks
        if len(patient_dict) < 10:
            raise ValueError("Input features seem incomplete. Provide a valid patient clinical profile.")
            
        X_processed = self._preprocess_input(df)
        
        # Toxicity Risk
        tox_pred_encoded = self.tox_model.predict(X_processed)[0]
        tox_probs = self.tox_model.predict_proba(X_processed)[0]
        tox_class = self.tox_encoder.inverse_transform([tox_pred_encoded])[0]
        tox_prob_dict = {str(c): float(p) for c, p in zip(self.tox_encoder.classes_, tox_probs)}
        
        # Therapy Response
        ther_pred_encoded = self.ther_model.predict(X_processed)[0]
        ther_probs = self.ther_model.predict_proba(X_processed)[0]
        ther_class = self.ther_encoder.inverse_transform([ther_pred_encoded])[0]
        ther_prob_dict = {str(c): float(p) for c, p in zip(self.ther_encoder.classes_, ther_probs)}
        
        # SHAP Explainability
        tox_shap = self.explainer_tox.shap_values(X_processed)
        if isinstance(tox_shap, list):
            tox_impact = tox_shap[0][0]
        elif len(tox_shap.shape) == 3:
            tox_impact = tox_shap[0, :, 0]
        else:
            tox_impact = tox_shap[0]
            
        tox_top_idx = np.argsort(np.abs(tox_impact))[-3:][::-1]
        tox_factors = [self.final_feature_names[i] for i in tox_top_idx]
        
        ther_shap = self.explainer_ther.shap_values(X_processed)
        if isinstance(ther_shap, list):
            ther_impact = ther_shap[0][0]
        elif len(ther_shap.shape) == 3:
            ther_impact = ther_shap[0, :, 0]
        else:
            ther_impact = ther_shap[0]
            
        ther_top_idx = np.argsort(np.abs(ther_impact))[-3:][::-1]
        ther_factors = [self.final_feature_names[i] for i in ther_top_idx]
        
        result = {
            "toxicity_risk": {
                "prediction": tox_class,
                "confidence": float(np.max(tox_probs)),
                "probabilities": tox_prob_dict,
                "important_factors": tox_factors
            },
            "therapy_response": {
                "prediction": ther_class,
                "confidence": float(np.max(ther_probs)),
                "probabilities": ther_prob_dict,
                "important_factors": ther_factors
            }
        }
        
        return result
