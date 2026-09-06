# Stage 2 Deep Learning: Sequence Trajectory Forecasting Report

## 1. Objective
Develop a sequence trajectory forecasting head to predict next-step biomarker values (such as `ctdna_level` and `tumor_volume`) from longitudinal patient readings over 8 time steps, extending the Stage 2 sequence processing capability beyond classification.

## 2. Dataset
- **Longitudinal Dataset**: Tracks 1,000 patients over 8 chronological time steps across 3 temporal features (`ctdna_level`, `biomarker_2`, `tumor_volume`).
- **Patient Split**: Preserves identical patient-level splits (70% Train, 15% Val, 15% Test) to guarantee zero data leakage between patient cohorts.

## 3. Synthetic Data Limitation
**CRITICAL LIMITATION**: The longitudinal sequence dataset is synthetic. It is designed to evaluate sequence model capabilities and does not represent real-world clinical patient trajectories.

## 4. Architecture & Subclassing
- **Base Architecture**: Reuses the `SequenceTransformer` / `LongitudinalLSTM` backbone (import/subclass without copy-pasting code).
- **Regression Head**: Replaces the final classification layer `Linear(hidden_dim -> num_classes)` with a regression head `Linear(hidden_dim -> output_dim)` (where `output_dim=1` for next-step trajectory forecasting).

## 5. Loss & Optimization
- **Loss Function**: `nn.HuberLoss(delta=1.0)` (Smooth L1 Loss) chosen for robust regression against noisy biomarker readings.
- **Optimizer**: Adam (`lr=1e-3`, `weight_decay=1e-4`).
- **Scheduler**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`.
- **Early Stopping**: 15 epochs patience monitoring Validation Huber Loss.

## 6. Test Evaluation Metrics
Evaluated on the held-out test cohort ($n=150$ patients):
- **MAE (Physical ctDNA)**: $383.87$ ng/mL
- **RMSE (Physical ctDNA)**: $1194.11$ ng/mL
- **$R^2$ Score**: **$0.7025$**
- **Best Validation Huber Loss**: $0.0277$

## 7. Artifact Locations
- **Model Checkpoint**: [`stage2_dl/artifacts/models/best_forecast_model.pth`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage2_dl/artifacts/models/best_forecast_model.pth)
- **Metrics JSON**: `stage2_dl/artifacts/metrics/forecast_test_metrics.json`
- **Trajectory Plot**: `stage2_dl/artifacts/figures/forecast_trajectories.png`

## 8. Reproducibility Commands
```bash
python stage2_dl/sequence/forecast_train.py
python stage2_dl/sequence/forecast_evaluate.py
```
