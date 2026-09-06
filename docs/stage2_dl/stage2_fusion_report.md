# Stage 2 Deep Learning: Multimodal Fusion Module Report

## 1. Objective
Combine static vision malignancy probabilities (from the histopathology CNN branch) and longitudinal sequence trajectory trends (from the Transformer forecaster branch) into a unified, risk-calibrated patient signal.

## 2. Architecture & Design Rationale
- **Input Signals**:
  1. `vision_score` $S_{\text{vision}} \in [0, 1]$: Malignancy output probability from `LightweightCNN`.
  2. `trend_score` $S_{\text{trend}} \in [0, 1]$: Calibrated percentage change trajectory slope from `TransformerForecaster`.
- **Fusion Strategy Choice**: A weighted convex combination rule ($S_{\text{combined}} = w_v \cdot S_{\text{vision}} + w_t \cdot S_{\text{trend}}$) is implemented. This transparent rule-based approach is chosen over a black-box neural layer to ensure full clinical auditability of how image features and temporal biomarker shifts drive the final decision.
- **Risk Classification Rules**:
  - `High`: $S_{\text{combined}} \ge 0.66$
  - `Moderate`: $0.33 \le S_{\text{combined}} < 0.66$
  - `Low`: $S_{\text{combined}} < 0.33$

## 3. Output Schema
The module returns a JSON-serializable dictionary:
```json
{
    "vision_score": 0.9996,
    "trend_score": 1.0000,
    "combined_risk": 0.9998,
    "flag": "High"
}
```

## 4. Module Location & Execution
- **Implementation**: [`stage2_dl/integration/fuse.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage2_dl/integration/fuse.py)
- **Execution Command**: `python stage2_dl/integration/fuse.py`

## 5. Prototype Limitations
- Combines 2D vision and 1D temporal sequences; radiological CT/MRI features are not yet connected to the fusion rule.
- Relies on synthetic longitudinal sequence data and benchmark vision images.
