# Stage 2 Deep Learning: LSTM Sequence Pipeline Report

## 1. Objective
Develop a modest LSTM-based classifier to predict patient outcomes (Responder vs. Non-Responder) using longitudinal sequential data (ctdna_level, biomarker_2, tumor_volume).

## 2. Dataset
The temporal dataset contains 1000 total patients, tracked across 8 time steps with 3 features each.

## 3. Synthetic-data Limitation
**CRITICAL LIMITATION**: The longitudinal sequence dataset is completely SYNTHETIC. It is not clinically validated and does not represent real clinical evidence. 

## 4. Input Representation
The input features are numerical continuous variables. Missing time steps (padding) are represented as `NaN` values before preprocessing. These are imputed to `0.0` after feature scaling. The sequence lengths are calculated dynamically by checking for `NaN` presence in the raw data, allowing `pack_padded_sequence` to ignore padded time steps completely during LSTM forward propagation.

## 5. Sequence Shape
`(batch_size, 8, 3)`

## 6. LSTM Architecture
- **LSTM Layer**: input_size=3, hidden_size=64, num_layers=1, batch_first=True
- **Dropout**: p=0.5 applied to the final valid hidden state
- **Linear Head**: 64 -> 2 classes

## 7. Parameter Count
- **Total Parameters**: 17,538
- **Trainable**: 17,538

## 8. Training Configuration
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Batch Size**: 32
- **Epochs**: 100 (Max)

## 9. Loss Function
CrossEntropyLoss with class weights calculated exclusively from the training set.

## 10. Class Weighting
Responder and Non-Responder classes are balanced via the loss function to prevent majority class collapse. Weights were derived solely from the 700-patient training split.

## 11. Validation Strategy
The model checkpoint with the highest **Validation Macro F1** score is saved as `best_lstm_model.pth`.

## 12. Early Stopping
Early stopping is configured with a patience of 15 epochs without improvement on Validation Macro F1.

## 13. Learning-rate Scheduling
`ReduceLROnPlateau` (factor=0.5, patience=5) monitors the Validation Macro F1 and reduces the learning rate when plateaus occur.

## 14. Test Results
- **Accuracy**: 0.8200
- **Macro Precision**: 0.8143
- **Macro Recall**: 0.8203
- **Macro F1**: 0.8164
- **Responder (F1)**: 0.7907
- **Non-Responder (F1)**: 0.8421

## 15. Confusion Matrix Interpretation
The model demonstrates well-balanced recall across both classes (Responder: 82%, Non-Responder: 81%), avoiding the severe false-negative bias observed in the image branch.

## 16. ROC-AUC
**Test ROC-AUC**: 0.9250

## 17. Overfitting Observations
The LSTM trained extremely smoothly with validation F1 climbing alongside training F1 until epoch ~34, where it peaked. Early stopping eventually halted the training at epoch 49, avoiding substantial overfitting.

## 18. Leakage Prevention
- **Patient Split**: Ensured by `sequence_loader.py` (intersection of train/val/test patient IDs is 0).
- **Normalization**: Scalers fit exclusively on `X_train`. The means and standard deviations are saved into the model checkpoint to apply to unseen data deterministically.

## 19. Missing/padding Handling
Using `pack_padded_sequence`, the LSTM processes only up to the true chronological length of each patient's sequence before padding. 

## 20. Prediction Pipeline
Implemented in `predict.py`. It accepts a raw numpy sequence array, scales it utilizing the checkpoint's saved parameters, and returns probabilities without re-fitting normalizers.

## 21. Artifact Locations
- **Model**: `stage2_dl/artifacts/models/best_lstm_model.pth`
- **Metrics**: `stage2_dl/artifacts/metrics/lstm_test_metrics.json`
- **Figures**: `stage2_dl/artifacts/figures/lstm_training_curves.png`, `lstm_confusion_matrix.png`, `lstm_roc_curve.png`

## 22. Limitations
- Does not combine imaging modalities.
- Assumes linear trajectory pacing between time steps (no explicit time-delta features provided).
- Entirely synthetic data.

## 23. Commands Executed
```bash
python -m pytest tests/stage2_dl/test_sequence.py
python -m stage2_dl.sequence.train
python -m stage2_dl.sequence.evaluate
python -m stage2_dl.sequence.predict
python stage2_dl/data/validate_data.py
```
