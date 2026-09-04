# Stage 2 Deep Learning: CNN Vision Pipeline Report

## 1. Objective
Develop a lightweight Convolutional Neural Network (CNN) to perform binary classification (malignant vs. normal/benign) on the BreastMNIST dataset. This prototype aims to explore deep learning applications in oncology image analysis.

## 2. Dataset Description
The dataset used is **BreastMNIST (MedMNIST v2)**, a benchmark dataset for medical image classification.
- **Original Resolution**: 28x28 grayscale images.
- **Classes**: 0 = Malignant, 1 = Normal/Benign.

## 3. Dataset Split
The official split is strictly preserved to prevent data leakage:
- **Total Images**: 780
- **Train**: 546 (Malignant: 147, Normal/Benign: 399)
- **Validation**: 78 (Malignant: 21, Normal/Benign: 57)
- **Test**: 156 (Malignant: 42, Normal/Benign: 114)

## 4. Preprocessing
To satisfy CNN structural requirements without creating artificial medical information:
- Images are resized from 28x28 to **128x128** using bicubic interpolation.
- The values are normalized to a range of [0, 1].

## 5. CNN Architecture
A lightweight PyTorch-based CNN (`LightweightCNN`) with 3 convolutional blocks:
- **Block 1**: Conv2D(1 -> 32) -> BatchNorm -> ReLU -> MaxPool2D(2)
- **Block 2**: Conv2D(32 -> 64) -> BatchNorm -> ReLU -> MaxPool2D(2)
- **Block 3**: Conv2D(64 -> 128) -> BatchNorm -> ReLU -> MaxPool2D(2)
- **Head**: Global Average Pooling (GAP) -> Dropout(0.5) -> Linear(128 -> 2)

## 6. Parameter Count
- **Total Parameters**: 114,370
- **Trainable Parameters**: 114,370

## 7. Augmentation
Conservative data augmentation is applied **only to the training set**:
- Random Rotation (±10 degrees)
- Random Affine Translation (±5%)
- No horizontal flipping (disabled by default for stability).
- **Validation/Test**: Deterministic preprocessing only.

## 8. Class Imbalance Handling
The training set exhibits an approximate imbalance ratio of 2.71:1 (Normal to Malignant). 
This is handled using a class-weighted `CrossEntropyLoss`. Weights are calculated **strictly from the training labels** to avoid validation/test set leakage.

## 9. Training Configuration
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Loss**: Weighted CrossEntropyLoss
- **Batch Size**: 32
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping Patience**: 10 epochs
- **Random Seed**: 42 (Reproducible behavior)

## 10. Model Selection Strategy
The best model checkpoint is selected purely based on **Validation Macro F1** to ensure a balanced evaluation across both majority and minority classes. The test set remains entirely unseen during the selection phase.

## 11. Final Test Metrics
Evaluated on the unseen test set (156 images):
- **Accuracy**: 0.8205
- **Macro Precision**: 0.8087
- **Macro Recall**: 0.7042
- **Macro F1**: 0.7310

## 12. Malignant-Class Metrics (Important Focus)
- **Malignant Precision**: 0.7917
- **Malignant Recall**: 0.4524
- **Malignant F1**: 0.5758

## 13. Confusion Matrix Interpretation
The model achieves high precision (0.79) for identifying malignancies but struggles with recall (0.45), indicating a tendency toward false negatives (missing malignant cases). While the model accurately identifies normal/benign samples (recall=0.95), the low malignant recall highlights the difficulty of extracting complex structural features from low-resolution (28x28) interpolated sources.

## 14. ROC-AUC
- **Test ROC-AUC**: 0.8095

## 15. Training Curves
The loss and Macro F1 curves for both training and validation sets are saved under `stage2_dl/artifacts/figures/training_curves.png`.

## 16. Overfitting Analysis
Moderate overfitting is observed. Training F1 consistently reached >0.73, while validation Macro F1 plateaued around 0.72. The use of Dropout (0.5), early stopping, and conservative augmentation successfully stabilized the divergence, but inherent dataset size (546 training images) continues to limit generalization.

## 17. Grad-CAM Explainability
A lightweight PyTorch Grad-CAM module (`stage2_dl/vision/explainability.py`) extracts feature map gradients from the final convolutional block. These activation maps are saved in `stage2_dl/artifacts/figures/gradcam_test_img_*.png`. 
*Note: Grad-CAM regions should not be interpreted as medically diagnostic indicators.*

## 18. Limitations
- **BreastMNIST is a benchmark dataset.**
- The source images are only **28x28**. Resizing to 128x128 is an interpolation method and does not recover real biological textures.
- Patient-level leakage cannot be independently verified due to the lack of patient IDs in BreastMNIST metadata.
- **This model is a student/research prototype.**
- **The model is NOT clinically validated.** Results must NOT be interpreted as evidence of real-world diagnostic or treatment effectiveness.

## 19. Reproducibility Instructions
To ensure reproducible deterministic behavior, the seed `42` is explicitly set via NumPy and PyTorch before training execution.

## 20. Files Created/Modified
- `stage2_dl/vision/cnn_model.py`
- `stage2_dl/vision/augmentation.py`
- `stage2_dl/vision/train.py`
- `stage2_dl/vision/evaluate.py`
- `stage2_dl/vision/predict.py`
- `stage2_dl/vision/explainability.py`
- `tests/stage2_dl/test_vision.py`
- `docs/stage2_dl/stage2_cnn_report.md`

## 21. Commands to Run
**1. Run Data Validation**
```bash
python stage2_dl/data/validate_data.py
```

**2. Train CNN**
```bash
python -m stage2_dl.vision.train
```

**3. Evaluate CNN**
```bash
python -m stage2_dl.vision.evaluate
```

**4. Generate Explainability Visualizations**
```bash
python -m stage2_dl.vision.explainability
```

**5. Run Automated Tests**
```bash
python -m pytest tests/stage2_dl/test_vision.py
```
