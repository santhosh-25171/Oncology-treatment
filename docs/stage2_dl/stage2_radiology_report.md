# Stage 2 Deep Learning: Radiological CT/MRI Branch Report

## 1. Objective
Develop a radiological image processing branch using a 2D Convolutional Neural Network (`RadiologyCNN`) to analyze CT/MRI-style scan slices (e.g. NoduleMNIST/OrganMNIST benchmark slices) for nodule/lesion identification.

## 2. Dataset Description
- **Dataset**: MedMNIST-family CT/MRI scan slice benchmark prototype dataset.
- **Classes**: `nodule_lesion` (Class 0) vs. `normal_tissue` (Class 1).
- **Split**: 500 Train (70%), 100 Validation (15%), 150 Test (15%).

## 3. Prototype Disclaimer & Limitations
**PROTOTYPE NOTICE**: This pipeline utilizes a public MedMNIST-family benchmark dataset structure for CT/MRI-style scan slices. It serves as a functional deep learning prototype for radiologic lesion detection and is **NOT** clinically validated diagnostic evidence.

## 4. Architecture
`RadiologyCNN` consists of 3 convolutional blocks:
- **Block 1**: Conv2D(1 -> 32) -> BatchNorm -> ReLU -> MaxPool2D(2)
- **Block 2**: Conv2D(32 -> 64) -> BatchNorm -> ReLU -> MaxPool2D(2)
- **Block 3**: Conv2D(64 -> 128) -> BatchNorm -> ReLU -> MaxPool2D(2)
- **Head**: Global Average Pooling (GAP) -> Dropout(0.5) -> Linear(128 -> 2)

## 5. Training Configuration
- **Loss Function**: Class-weighted `CrossEntropyLoss` to handle class imbalance.
- **Optimizer**: Adam (`lr=1e-3`, `weight_decay=1e-4`).
- **Scheduler**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=5)`.
- **Early Stopping**: 15 epochs monitoring Validation Macro F1.

## 6. Evaluation Metrics
Evaluated on the unseen test set ($n=150$ slices):
- **Accuracy**: $70.00\%$
- **Macro F1**: $0.4118$
- **Macro Recall**: $0.5000$

## 7. Grad-CAM Explainability
A PyTorch Grad-CAM module (`stage2_dl/radiology/explainability.py`) extracts gradient activation heatmaps from the final convolutional block to highlight radiologic scan regions driving predictions.

## 8. Artifact Locations
- **Model Checkpoint**: [`stage2_dl/artifacts/models/best_radiology_model.pth`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/ML%20project/stage2_dl/artifacts/models/best_radiology_model.pth)
- **Metrics JSON**: `stage2_dl/artifacts/metrics/radiology_test_metrics.json`
- **Confusion Matrix**: `stage2_dl/artifacts/figures/radiology_confusion_matrix.png`
- **Grad-CAM Visualization**: `stage2_dl/artifacts/figures/radiology_gradcam.png`

## 9. Reproducibility Commands
```bash
python stage2_dl/radiology/train.py
python stage2_dl/radiology/evaluate.py
python stage2_dl/radiology/explainability.py
```
