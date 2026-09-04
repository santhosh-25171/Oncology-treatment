# Stage 1 ML Inference & Decision Threshold Bug Report

## 1. Root Cause Analysis
In the previous implementation of `stage1_ml/prediction/prediction.py`, the risk classification decision logic contained a critical flaw:

```python
# BROKEN LOGIC:
high_risk_prob = float(ov_probs[high_idx])
if high_risk_prob >= 0.48:
    ov_class = "High"
else:
    ov_pred_encoded = self.overall_model.predict(X_processed)[0]
    ov_class = str(self.overall_encoder.inverse_transform([ov_pred_encoded])[0])
```

### Why it failed:
1. `self.overall_model` is a `CalibratedClassifierCV(estimator=XGBClassifier)`.
2. In `sklearn`, `CalibratedClassifierCV.predict(X)` evaluates `argmax(calibrated_probabilities)`.
3. When `calibrated_high_risk_prob = 0.462 < 0.48`:
   - The code entered the `else:` branch.
   - It executed `self.overall_model.predict(X_processed)`, which calculated `argmax([High: 0.462, Low: 0.250, Moderate: 0.288])`.
   - Since `0.462` was the maximum probability in the array, `argmax` returned `0` (`High`).
   - Consequently, `ov_class` was set to `"High"` despite `calibrated_high_risk_prob (0.462) < threshold (0.48)`!

---

## 2. Corrected Behavior & Authoritative Decision Function
We created `decide_overall_patient_risk()` as the single authoritative classification function:

```python
def decide_overall_patient_risk(ov_prob_dict: dict, high_risk_threshold: float = 0.48) -> str:
    """
    Authoritative decision function for Overall Patient Risk classification.
    
    Rules:
    1. If calibrated High-Risk probability >= high_risk_threshold (0.48):
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
```

---

## 3. Before / After Decision Comparison

| Input Scenario | High-Risk Prob | Decision Threshold | Old (Broken) Class | New (Corrected) Class | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Preset LOW_RISK_TEST** | 39.46% | 48.0% | `High` *(Bug)* | **`Moderate`** | ✅ Fixed |
| **Sample Patient** | 46.20% | 48.0% | `High` *(Bug)* | **`Moderate`** | ✅ Fixed |
| **High Risk Patient** | 57.82% | 48.0% | `High` | **`High`** | ✅ Consistent |

---

## 4. Key Files Changed
1. **[`stage1_ml/prediction/prediction.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage1_ml/prediction/prediction.py)**: Added `decide_overall_patient_risk` function with strict assertions and `debug_info` extraction.
2. **[`integration/api/main.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/integration/api/main.py)**: Updated `/predict` response payload to pass `threshold` and `debug_info`.
3. **[`integration/dashboard/app.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/integration/dashboard/app.py)**: Defined coherent deterministic presets (`LOW_RISK_TEST`, `MODERATE_RISK_TEST`, `HIGH_RISK_TEST`), added Developer Diagnostics & Inference Audit section, and updated UI threshold badges.
4. **[`stage1_ml/evaluation/test_risk_decision.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage1_ml/evaluation/test_risk_decision.py)**: Added 8 automated consistency tests passing 100%.

---

## 5. Audit Details
- **Champion Model**: Calibrated XGBoost (Platt Scaling)
- **Calibration Method**: Sigmoidal Logistic Calibration (`CalibratedClassifierCV`)
- **Decision Threshold**: `0.48` loaded directly from `data/stage1_ml/models/tuning/best_hyperparameters.json`
- **Automated Validation Results**: All 8 tests passed (`python stage1_ml/evaluation/test_risk_decision.py`).
