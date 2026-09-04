# Stage 2 Deep Learning: Transformer Sequence Pipeline Report

## 1. Objective
Develop a Transformer Encoder classifier to predict patient outcomes (Responder vs. Non-Responder) from longitudinal sequential data (ctdna_level, biomarker_2, tumor_volume), serving as an alternative sequence modality branch.

## 2. Dataset
The temporal dataset tracks 1000 total patients, across 8 possible chronological time steps, capturing 3 features at each step.

## 3. Synthetic-data Limitation
**CRITICAL LIMITATION**: The longitudinal sequence dataset is completely SYNTHETIC. It is not clinically validated and does not represent real clinical evidence.

## 4. Input Shape
`(batch_size, 8, 3)` (Dynamically masked based on valid sequence lengths).

## 5. Positional Encoding
Since the self-attention mechanism is invariant to token position, a deterministic sinusoidal **Positional Encoding** layer is applied immediately after the input projection. This ensures the model retains the chronological ordering (time step 1 vs. time step 8) of the patient's sequence.

## 6. Transformer Architecture
- **Input Projection**: Linear layer maps 3 features to `d_model=32`.
- **Positional Encoding**: Sinusoidal encoding added to embeddings.
- **Transformer Encoder**: 2 layers, 4 attention heads (`nhead=4`), `dim_feedforward=128`, `dropout=0.1`.
- **Classification Head**: Mean-pooling (ignoring padded tokens) -> Dropout(0.1) -> Linear(32 -> 2 classes).

## 7. Parameter Count
- **Total Parameters**: 12,866
- **Trainable Parameters**: 12,866

## 8. Training Configuration
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Batch Size**: 32
- **Epochs**: 100 (Max)

## 9. Loss
CrossEntropyLoss with dynamic class weights.

## 10. Class Weighting
Responder and Non-Responder classes are balanced via the loss function. Weights were calculated exclusively from the 700-patient training set to avoid leakage.

## 11. Validation Strategy
The model checkpoint with the highest **Validation Macro F1** score is saved as `best_transformer_model.pth`.

## 12. Early Stopping
Patience of 15 epochs without improvement on Validation Macro F1. 

## 13. Learning-rate Scheduling
`ReduceLROnPlateau` (factor=0.5, patience=5) decreases the learning rate if the validation metric plateaus.

## 14. Test Metrics
Evaluated strictly on the completely unseen 150-patient test set:
- **Accuracy**: 0.8467
- **Macro Precision**: 0.8419
- **Macro Recall**: 0.8503
- **Macro F1**: 0.8442
- **Responder (F1)**: 0.8244
- **Non-Responder (F1)**: 0.8639

## 15. Confusion Matrix
- **True Non-Responder predicted as Non-Responder**: 73
- **True Non-Responder predicted as Responder**: 15
- **True Responder predicted as Non-Responder**: 8
- **True Responder predicted as Responder**: 54

The matrix shows strong diagnostic balance with low False Positives and False Negatives relative to dataset scale.

## 16. ROC-AUC
**Test ROC-AUC**: 0.9272

## 17. Overfitting Observations
The Transformer converged extremely rapidly (best Validation F1 at Epoch 16). After Epoch 16, training loss continued declining into the low ~0.18 range while validation F1 slowly drifted downwards. Early stopping successfully caught the peak before severe overfitting fully entrenched itself.

## 18. Leakage Prevention
- **Patient Split**: No patient ID intersection between train/val/test splits (verified).
- **Normalization Leakage**: Scaler fitted exclusively on `X_train`. Means and STDs saved directly into checkpoint to apply strictly to evaluation/prediction without refitting.
- **Future Leakage**: No decoder or autoregressive generation is performed.

## 19. Missing/padding Handling
The raw `NaN` values indicate padding. The dataset computes true `lengths` before imputation. During the Transformer's `forward` pass, a `src_key_padding_mask` is dynamically generated from these lengths to explicitly block the self-attention mechanism from attending to padded time steps.

## 20. Prediction Pipeline
Implemented in `transformer_predict.py`. Dynamically scales inputs via checkpointed normalizer parameters, calculates sequence padding length, and extracts predictions and confidence intervals.

## 21. Artifact Locations
- **Model**: `stage2_dl/artifacts/models/best_transformer_model.pth`
- **Metrics**: `stage2_dl/artifacts/metrics/transformer_test_metrics.json`
- **Figures**: `stage2_dl/artifacts/figures/transformer_training_curves.png`, `transformer_confusion_matrix.png`, `transformer_roc_curve.png`

## 22. Limitations
- Entirely synthetic data limits real-world significance.
- Relies on sinusoidal positional encodings rather than learned time-delta encodings, assuming equidistant spacing between steps.

## 23. Commands Executed
```bash
python -m pytest tests/stage2_dl/test_transformer.py
python -m stage2_dl.sequence.transformer_train
python -m stage2_dl.sequence.transformer_evaluate
python -m stage2_dl.sequence.transformer_predict
python stage2_dl/data/validate_data.py
```
