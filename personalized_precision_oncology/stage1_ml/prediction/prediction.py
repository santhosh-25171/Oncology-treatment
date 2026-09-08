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
        
        self.overall_model = None
        self.tox_model = None
        self.ther_model = None
        
        self.overall_encoder = None
        self.tox_encoder = None
        self.ther_encoder = None
        
        self.thresholds = {"overall_patient_risk": 0.48, "toxicity_risk": 0.30, "therapy_response": 0.31}
        
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
        
    def load_models(self):
        """Load calibrated champion models, encoders, and threshold metadata."""
        overall_path = os.path.join(self.tuning_dir, "calibrated_overall_patient_risk_model.joblib")
        tox_path = os.path.join(self.tuning_dir, "calibrated_toxicity_model.joblib")
        ther_path = os.path.join(self.tuning_dir, "calibrated_therapy_response_model.joblib")
        
        # Fallback to tuned models if calibrated models aren't saved separately
        if not os.path.exists(overall_path):
            overall_path = os.path.join(self.tuning_dir, "tuned_overall_patient_risk_model.joblib")
        if not os.path.exists(tox_path):
            tox_path = os.path.join(self.tuning_dir, "tuned_toxicity_model.joblib")
        if not os.path.exists(ther_path):
            ther_path = os.path.join(self.tuning_dir, "tuned_therapy_response_model.joblib")
            
        self.overall_model = joblib.load(overall_path)
        self.tox_model = joblib.load(tox_path)
        self.ther_model = joblib.load(ther_path)
        
        self.overall_encoder = joblib.load(os.path.join(self.models_dir, "overall_patient_risk_label_encoder.joblib"))
        self.tox_encoder = joblib.load(os.path.join(self.models_dir, "toxicity_risk_label_encoder.joblib"))
        self.ther_encoder = joblib.load(os.path.join(self.models_dir, "therapy_response_label_encoder.joblib"))
        
        thresh_path = os.path.join(self.tuning_dir, "best_hyperparameters.json")
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path, "r") as f:
                    params_info = json.load(f)
                    for t in params_info:
                        if "optimized_high_risk_threshold" in params_info[t]:
                            self.thresholds[t] = params_info[t]["optimized_high_risk_threshold"]
            except Exception:
                pass

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
            df_feat['high_clinical_risk'] = ((df_feat['comorbidity_score'] > 2) & (df_feat['performance_status'] > 1)).astype(int)
        if 'biomarker_1' in df_feat.columns and 'biomarker_2' in df_feat.columns:
            df_feat['biomarker_interaction'] = df_feat['biomarker_1'] * df_feat['biomarker_2']
        return df_feat

    def _fit_preprocessing(self):
        data_path = os.path.join(self.base_dir, "data", "stage1_ml", "processed", "oncology_cleaned.csv")
        df = pd.read_csv(data_path)
        
        targets = ['overall_patient_risk', 'toxicity_risk', 'therapy_response']
        X = df.drop(columns=targets)
        y = df[targets]
        
        stratify_col = y['overall_patient_risk'].astype(str)
        X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.15, stratify=stratify_col, random_state=42)
        X_train, _, _, _ = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp['overall_patient_risk'].astype(str), random_state=42)
        
        X_train_eng = self._apply_feature_engineering(X_train)
        
        self.original_numerical_cols = X_train_eng.select_dtypes(include=[np.number]).columns.tolist()
        self.original_categorical_cols = X_train_eng.select_dtypes(exclude=[np.number]).columns.tolist()
        
        self.num_imputer = SimpleImputer(strategy='median')
        X_train_eng[self.original_numerical_cols] = self.num_imputer.fit_transform(X_train_eng[self.original_numerical_cols])
        
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        X_train_eng[self.original_categorical_cols] = self.cat_imputer.fit_transform(X_train_eng[self.original_categorical_cols])
        
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat_encoded = self.encoder.fit_transform(X_train_eng[self.original_categorical_cols])
        cat_feature_names = self.encoder.get_feature_names_out(self.original_categorical_cols)
        X_train_cat_df = pd.DataFrame(X_train_cat_encoded, columns=cat_feature_names, index=X_train_eng.index)
        
        self.scaler = StandardScaler()
        X_train_num_scaled = self.scaler.fit_transform(X_train_eng[self.original_numerical_cols])
        X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=self.original_numerical_cols, index=X_train_eng.index)
        
        X_train_processed = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
        
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

def decide_overall_patient_risk(ov_prob_dict: dict, high_risk_threshold: float = 0.48) -> str:
    """
    Authoritative decision function for Overall Patient Risk classification.
    
    Rules:
    1. If calibrated High-Risk probability >= high_risk_threshold (default 0.48):
       Risk Class is 'High'.
    2. If calibrated High-Risk probability < high_risk_threshold:
       Risk Class MUST NOT be 'High'.
       The decision between non-High classes ('Moderate' vs 'Low') is determined
       by comparing their calibrated probabilities (whichever is higher).
    """
    high_prob = ov_prob_dict.get("High", 0.0)
    
    if high_prob >= high_risk_threshold:
        risk_class = "High"
    else:
        p_mod = ov_prob_dict.get("Moderate", 0.0)
        p_low = ov_prob_dict.get("Low", 0.0)
        risk_class = "Moderate" if p_mod >= p_low else "Low"
        
    # Automated assertion enforcing decision consistency
    if high_prob < high_risk_threshold:
        assert risk_class != "High", f"Inconsistent Classification Bug: high_prob ({high_prob:.4f}) < threshold ({high_risk_threshold:.4f}) but risk_class assigned was 'High'"
    else:
        assert risk_class == "High", f"Inconsistent Classification Bug: high_prob ({high_prob:.4f}) >= threshold ({high_risk_threshold:.4f}) but risk_class assigned was '{risk_class}'"
        
    return risk_class


class OncologyPredictionPipeline:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = base_dir
            
        self.models_dir = os.path.join(self.base_dir, "data", "stage1_ml", "models")
        self.tuning_dir = os.path.join(self.models_dir, "tuning")
        
        self.overall_model = None
        self.tox_model = None
        self.ther_model = None
        
        self.overall_encoder = None
        self.tox_encoder = None
        self.ther_encoder = None
        
        self.thresholds = {"overall_patient_risk": 0.48, "toxicity_risk": 0.30, "therapy_response": 0.31}
        
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
        
    def load_models(self):
        """Load calibrated champion models, encoders, and threshold metadata."""
        overall_path = os.path.join(self.tuning_dir, "calibrated_overall_patient_risk_model.joblib")
        tox_path = os.path.join(self.tuning_dir, "calibrated_toxicity_model.joblib")
        ther_path = os.path.join(self.tuning_dir, "calibrated_therapy_response_model.joblib")
        
        if not os.path.exists(overall_path):
            overall_path = os.path.join(self.tuning_dir, "tuned_overall_patient_risk_model.joblib")
        if not os.path.exists(tox_path):
            tox_path = os.path.join(self.tuning_dir, "tuned_toxicity_model.joblib")
        if not os.path.exists(ther_path):
            ther_path = os.path.join(self.tuning_dir, "tuned_therapy_response_model.joblib")
            
        self.overall_model = joblib.load(overall_path)
        self.tox_model = joblib.load(tox_path)
        self.ther_model = joblib.load(ther_path)
        
        self.overall_encoder = joblib.load(os.path.join(self.models_dir, "overall_patient_risk_label_encoder.joblib"))
        self.tox_encoder = joblib.load(os.path.join(self.models_dir, "toxicity_risk_label_encoder.joblib"))
        self.ther_encoder = joblib.load(os.path.join(self.models_dir, "therapy_response_label_encoder.joblib"))
        
        thresh_path = os.path.join(self.tuning_dir, "best_hyperparameters.json")
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path, "r") as f:
                    params_info = json.load(f)
                    for t in params_info:
                        if "optimized_high_risk_threshold" in params_info[t]:
                            self.thresholds[t] = params_info[t]["optimized_high_risk_threshold"]
            except Exception:
                pass

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
            df_feat['high_clinical_risk'] = ((df_feat['comorbidity_score'] > 2) & (df_feat['performance_status'] > 1)).astype(int)
        if 'biomarker_1' in df_feat.columns and 'biomarker_2' in df_feat.columns:
            df_feat['biomarker_interaction'] = df_feat['biomarker_1'] * df_feat['biomarker_2']
        return df_feat

    def _fit_preprocessing(self):
        data_path = os.path.join(self.base_dir, "data", "stage1_ml", "processed", "oncology_cleaned.csv")
        df = pd.read_csv(data_path)
        
        targets = ['overall_patient_risk', 'toxicity_risk', 'therapy_response']
        X = df.drop(columns=targets)
        y = df[targets]
        
        stratify_col = y['overall_patient_risk'].astype(str)
        X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.15, stratify=stratify_col, random_state=42)
        X_train, _, _, _ = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp['overall_patient_risk'].astype(str), random_state=42)
        
        X_train_eng = self._apply_feature_engineering(X_train)
        
        self.original_numerical_cols = X_train_eng.select_dtypes(include=[np.number]).columns.tolist()
        self.original_categorical_cols = X_train_eng.select_dtypes(exclude=[np.number]).columns.tolist()
        
        self.num_imputer = SimpleImputer(strategy='median')
        X_train_eng[self.original_numerical_cols] = self.num_imputer.fit_transform(X_train_eng[self.original_numerical_cols])
        
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        X_train_eng[self.original_categorical_cols] = self.cat_imputer.fit_transform(X_train_eng[self.original_categorical_cols])
        
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat_encoded = self.encoder.fit_transform(X_train_eng[self.original_categorical_cols])
        cat_feature_names = self.encoder.get_feature_names_out(self.original_categorical_cols)
        X_train_cat_df = pd.DataFrame(X_train_cat_encoded, columns=cat_feature_names, index=X_train_eng.index)
        
        self.scaler = StandardScaler()
        X_train_num_scaled = self.scaler.fit_transform(X_train_eng[self.original_numerical_cols])
        X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=self.original_numerical_cols, index=X_train_eng.index)
        
        X_train_processed = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
        
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

    def _preprocess_input(self, df):
        df_eng = self._apply_feature_engineering(df)
        
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
        
        for col in self.final_feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.final_feature_names]
        return df_processed

    def predict(self, patient_dict):
        df = pd.DataFrame([patient_dict])
        
        if len(patient_dict) < 5:
            raise ValueError("Patient feature input incomplete. Minimum clinical fields required.")
            
        X_processed = self._preprocess_input(df)
        
        # 1. Primary Target: overall_patient_risk
        ov_probs = self.overall_model.predict_proba(X_processed)[0]
        ov_prob_dict = {str(c): float(p) for c, p in zip(self.overall_encoder.classes_, ov_probs)}
        
        # Extract raw uncalibrated base estimator probability for audit/debug if available
        raw_high_prob = float(ov_prob_dict.get("High", 0.0))
        try:
            if hasattr(self.overall_model, "calibrated_classifiers_"):
                raw_probs_list = [cc.estimator.predict_proba(X_processed)[0] for cc in self.overall_model.calibrated_classifiers_]
                raw_mean_probs = np.mean(raw_probs_list, axis=0)
                high_idx = [i for i, c in enumerate(self.overall_encoder.classes_) if str(c).lower() == 'high'][0]
                raw_high_prob = float(raw_mean_probs[high_idx])
        except Exception:
            pass
            
        high_risk_threshold = float(self.thresholds.get("overall_patient_risk", 0.48))
        calibrated_high_prob = float(ov_prob_dict.get("High", 0.0))
        
        # Use authoritative decision logic
        ov_class = decide_overall_patient_risk(ov_prob_dict, high_risk_threshold=high_risk_threshold)
            
        # 2. Secondary Target: toxicity_risk
        tox_probs = self.tox_model.predict_proba(X_processed)[0]
        tox_prob_dict = {str(c): float(p) for c, p in zip(self.tox_encoder.classes_, tox_probs)}
        tox_pred_encoded = self.tox_model.predict(X_processed)[0]
        tox_class = str(self.tox_encoder.inverse_transform([tox_pred_encoded])[0])
        
        # 3. Secondary Target: therapy_response
        ther_probs = self.ther_model.predict_proba(X_processed)[0]
        ther_prob_dict = {str(c): float(p) for c, p in zip(self.ther_encoder.classes_, ther_probs)}
        ther_pred_encoded = self.ther_model.predict(X_processed)[0]
        ther_class = str(self.ther_encoder.inverse_transform([ther_pred_encoded])[0])
        
        # Key contributing factors
        top_factors = [
            {"feature": "comorbidity_score", "direction": "increases_risk" if patient_dict.get("comorbidity_score", 0) > 2 else "baseline"},
            {"feature": "ctDNA_level", "direction": "increases_risk" if patient_dict.get("ctDNA_level", 0) > 3 else "baseline"},
            {"feature": "performance_status", "direction": "increases_risk" if patient_dict.get("performance_status", 0) > 1 else "baseline"}
        ]
        
        result = {
            "overall_patient_risk": {
                "prediction": ov_class,
                "risk_probability": round(calibrated_high_prob, 4),
                "threshold": round(high_risk_threshold, 4),
                "confidence": round(float(np.max(ov_probs)), 4),
                "probabilities": ov_prob_dict,
                "important_factors": top_factors,
                "debug_info": {
                    "model_name": "Calibrated XGBoost (Platt Scaling)",
                    "calibration": "Sigmoidal Logistic Calibration",
                    "raw_high_risk_prob": round(raw_high_prob, 4),
                    "calibrated_high_risk_prob": round(calibrated_high_prob, 4),
                    "decision_threshold": round(high_risk_threshold, 4),
                    "final_risk_class": ov_class,
                    "is_threshold_applied": calibrated_high_prob >= high_risk_threshold
                }
            },
            "toxicity_risk": {
                "prediction": tox_class,
                "confidence": round(float(np.max(tox_probs)), 4),
                "probabilities": tox_prob_dict
            },
            "therapy_response": {
                "prediction": ther_class,
                "confidence": round(float(np.max(ther_probs)), 4),
                "probabilities": ther_prob_dict
            }
        }
        return result

