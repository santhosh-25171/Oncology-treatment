# DL Data Engineering Report

## Status: Completed

The data engineering pipeline and dataset structure for Stage 02 Deep Learning have been standardized, verified, and integrated into the repository.

### Dataset Architecture & Directory Structure
- **Root Directory**: `stage2_dl/sample_data/`
- **Train/Val/Test Split**: Defined at patient level in `stage2_dl/sample_data/train_validation_test_split.csv` to ensure zero cross-set leakage.
  - Split proportion: 70% Train, 15% Validation, 15% Test.
  - Stratified by clinical risk group.
- **Histopathology Images**:
  - Image directory: `stage2_dl/sample_data/images/`
  - Metadata: `stage2_dl/sample_data/spatial/spatial_metadata.csv`
  - Labels: `stage2_dl/sample_data/spatial/labels.csv` (mapping patient ID, patch ID, and tissue class)
  - 6 Tissue Classes: `normal`, `benign`, `malignant`, `tumor_margin`, `necrotic`, `inflammatory`
- **Longitudinal Temporal Biomarkers**:
  - Longitudinal feature sequence: `stage2_dl/sample_data/temporal/biomarker_timeseries.csv`
  - Chronological tracking: Sorted by `patient_id` and `timestep`/`time_days`
  - Binary progression target: `stage2_dl/sample_data/temporal/progression_targets.csv`
  - Clinical interventions: `stage2_dl/sample_data/temporal/treatment_timeline.csv`
- **Multimodal Alignment**:
  - `src/data/fusion_dataset.py` cross-references patient IDs present across spatial pathology and temporal biomarker sequences for joint multimodal fusion training and inference.
  - Verified no patient leakage exists across splits in `test_cnn.py`, `test_temporal.py`, and `test_fusion.py`.

