import os
import sys
import time
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Configure Paths
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "sample_data"
artifacts_dir = base_dir / "artifacts"
models_dir = artifacts_dir / "models"
results_dir = artifacts_dir / "results" / "cnn"
docs_dir = base_dir / "docs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# 1. Inspect Data & Prevent Leakage
logger.info("Loading metadata...")
spatial_meta = pd.read_csv(data_dir / "spatial" / "spatial_metadata.csv")
splits = pd.read_csv(data_dir / "train_validation_test_split.csv")

# Ensure split exists in spatial_meta or merge
if 'split' not in spatial_meta.columns:
    spatial_meta = spatial_meta.merge(splits, on='patient_id', how='left')

# Leakage check
train_pats = set(spatial_meta[spatial_meta['split'] == 'train']['patient_id'])
val_pats = set(spatial_meta[spatial_meta['split'] == 'validation']['patient_id'])
test_pats = set(spatial_meta[spatial_meta['split'] == 'test']['patient_id'])

if train_pats.intersection(val_pats) or train_pats.intersection(test_pats) or val_pats.intersection(test_pats):
    logger.error("DATA LEAKAGE DETECTED! Patients cross splits.")
    sys.exit(1)
logger.info("Leakage check passed: No patient overlap across splits.")

# Class mapping
classes = ['normal', 'benign', 'malignant', 'tumor_margin', 'necrotic', 'inflammatory']
class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

# 2. PyTorch Dataset
class OncologyImageDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_path = Path(row['image_path'])
        if raw_path.exists():
            img_path = raw_path
        elif (self.data_dir / row['image_path']).exists():
            img_path = self.data_dir / row['image_path']
        else:
            img_path = self.data_dir.parent.parent / raw_path
            
        label = class_to_idx[row['tissue_class']]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image: {img_path}")
            raise e
            
        if self.transform:
            img = self.transform(img)
            
        return img, label, row['patient_id']

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_df = spatial_meta[spatial_meta['split'] == 'train']
val_df = spatial_meta[spatial_meta['split'] == 'validation']
test_df = spatial_meta[spatial_meta['split'] == 'test']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device.type.upper()}")
if device.type == 'cuda':
    logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CPU execution detected. Downsampling heavily to finish in reasonable time.")
    train_df = train_df.groupby('tissue_class').head(50)
    val_df = val_df.groupby('tissue_class').head(20)
    test_df = test_df.groupby('tissue_class').head(20)

train_ds = OncologyImageDataset(train_df, data_dir, train_transform)
val_ds = OncologyImageDataset(val_df, data_dir, val_test_transform)
test_ds = OncologyImageDataset(test_df, data_dir, val_test_transform)

# 3. Handle Class Imbalance
class_counts = train_df['tissue_class'].value_counts()
weights = [1.0 / class_counts.get(cls, 1) for cls in classes]
sum_weights = sum(weights)
class_weights = torch.FloatTensor([w / sum_weights for w in weights])
logger.info(f"Class weights: {class_weights.numpy()}")

# 4. Baseline CNN Architecture
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 5. DataLoaders with OOM Fallback
def get_dataloaders(batch_size):
    # Windows limits workers, keep at 0 for safety
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_ld, val_ld, test_ld

batch_sizes = [32, 16, 8]
train_loader, val_loader, test_loader = None, None, None
model = BaselineCNN(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Smoke Test & Batch Size selection
for bz in batch_sizes:
    logger.info(f"Trying batch size {bz} for smoke test...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(bz)
        model.train()
        imgs, labels, _ = next(iter(train_loader))
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        logger.info(f"Smoke test passed with batch size {bz}")
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            logger.warning(f"OOM with batch size {bz}. Scaling down...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if bz == 8:
                logger.error("Failed even with batch size 8. Aborting.")
                sys.exit(1)
        else:
            raise e

# Print Configuration
logger.info("--- Training Configuration ---")
logger.info(f"Train samples: {len(train_ds)}")
logger.info(f"Val samples: {len(val_ds)}")
logger.info(f"Test samples: {len(test_ds)}")
logger.info(f"Classes: 6")
logger.info(f"Batch Size: {train_loader.batch_size}")
logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

# 6. Training Loop
EPOCHS = 10
patience = 3
best_val_f1 = 0.0
epochs_no_improve = 0
best_epoch = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # Train
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_train_loss = running_loss / len(train_ds)
    epoch_train_acc = accuracy_score(all_labels, all_preds)
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_ds)
    epoch_val_acc = accuracy_score(val_labels, val_preds)
    epoch_val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accs.append(epoch_train_acc)
    val_accs.append(epoch_val_acc)
    
    epoch_time = time.time() - epoch_start
    
    logger.info(f"Epoch {epoch}/{EPOCHS} | "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | "
                f"Val Macro-F1: {epoch_val_f1:.4f} | Time: {epoch_time:.1f}s")
    
    if epoch == 1:
        est_total = epoch_time * EPOCHS
        logger.info(f"First epoch duration: {epoch_time:.1f}s. Estimated total time: {est_total/60:.1f} minutes.")
        if epoch_time > 600:
            logger.warning("Training is extremely slow. Capping at 3 epochs for smoke check.")
            EPOCHS = 3
    
    # Early Stopping & Checkpoint
    if epoch_val_f1 > best_val_f1:
        best_val_f1 = epoch_val_f1
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_macro_f1': best_val_f1,
            'classes': classes
        }, models_dir / "cnn_best.pt")
        logger.info("Saved new best model.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

total_time = time.time() - start_time

# 7. Evaluation
logger.info("Loading best model for final evaluation...")
if os.path.exists(models_dir / "cnn_best.pt"):
    checkpoint = torch.load(models_dir / "cnn_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def evaluate(loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

test_labels, test_preds = evaluate(test_loader)
val_labels, val_preds = evaluate(val_loader)

# Metrics calculation
test_acc = accuracy_score(test_labels, test_preds)
test_macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
test_weighted_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
val_macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

report_dict = classification_report(test_labels, test_preds, target_names=classes, output_dict=True, zero_division=0)

# Save metrics
metrics_summary = {
    "dataset": {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "classes": classes
    },
    "training": {
        "device": device.type,
        "batch_size": train_loader.batch_size,
        "best_epoch": best_epoch,
        "total_time_seconds": total_time
    },
    "validation_metrics": {
        "macro_f1": val_macro_f1
    },
    "test_metrics": {
        "accuracy": test_acc,
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1
    },
    "per_class_test": {cls: report_dict[cls] for cls in classes}
}

with open(results_dir / "cnn_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)

# Create CSV for quick view
pd.DataFrame([metrics_summary['test_metrics']]).to_csv(results_dir / "cnn_results.csv", index=False)

# 8. Visualizations
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(results_dir / "loss_curve.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(results_dir / "acc_curve.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Test Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(results_dir / "confusion_matrix.png")
plt.close()

# Per-class F1
f1_scores = [report_dict[cls]['f1-score'] for cls in classes]
plt.figure(figsize=(10, 5))
sns.barplot(x=classes, y=f1_scores, hue=classes, palette='viridis', legend=False)
plt.title('Test F1-Score per Class')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(results_dir / "per_class_f1.png")
plt.close()

# 9. Markdown Report Generation
md_report = f"""# Stage 02 Deep Learning — Baseline CNN Report

## 1. Objective
To establish a baseline spatial modeling pipeline using a simple Convolutional Neural Network (CNN) to classify 224x224 histopathology images into six categories.

## 2. Dataset Used
- **Source**: Stage 02 Synthetic Oncology Dataset
- **Train**: {len(train_ds)} images
- **Validation**: {len(val_ds)} images
- **Test**: {len(test_ds)} images

## 3. Why CNN is Suitable
CNNs exploit spatial hierarchies and local pixel correlations, making them natively ideal for extracting cellular and morphological patterns from pathology images.

## 4. Image Preprocessing & 5. Data Augmentation
- **Train**: Resize(224), RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(10°), ColorJitter, Normalize.
- **Val/Test**: Resize(224), Normalize ONLY. 
- *Deterministic preprocessing prevents evaluation variability.*

## 6. Data Leakage Prevention
Patient ID overlap checks were rigidly enforced. No single patient spans multiple splits.

## 7. Class Imbalance Handling
CrossEntropyLoss was configured with explicit class weights inversely proportional to class frequencies to prevent the network from ignoring minority classes (e.g., necrotic).

## 8. Dataset/DataLoader Design
A custom `OncologyImageDataset` handles lazy loading from disk. Missing/corrupted images trigger explicit exceptions rather than silent replacement. Batch size dynamically scaled down upon OOM.

## 9. CNN Architecture
- **Input**: 224x224x3
- **Blocks**: 3 blocks of (Conv2D -> BatchNorm -> ReLU -> MaxPool)
- **Filters**: 32 -> 64 -> 128
- **Global Average Pooling**: Replaces massive FC layers, heavily reducing parameters and overfitting.
- **Dropout**: 0.5 before the final classifier.
- **Output**: 6 classes

## 10-16. Hyperparameters
- **Activation**: ReLU (combats vanishing gradients)
- **Batch Normalization**: Stabilizes training
- **Pooling**: MaxPool2d(2) and AdaptiveAvgPool2d(1)
- **Loss**: Weighted CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)

## 17-18. Training Process
- **Epochs**: max {EPOCHS} (Early stopping patience = 3)
- **Best Epoch**: {best_epoch}
- **Device**: {device.type}

## 19-21. Results
- **Test Accuracy**: {test_acc:.4f}
- **Test Macro-F1**: {test_macro_f1:.4f}
- **Test Weighted-F1**: {test_weighted_f1:.4f}

![Confusion Matrix](../artifacts/results/cnn/confusion_matrix.png)
![Loss Curve](../artifacts/results/cnn/loss_curve.png)

## 22-24. Limitations & Synthetic Risks
**IMPORTANT: This CNN is a research/educational prototype trained on synthetic histopathology-style images and is not clinically validated.**
Synthetic Shortcut Risk: The CNN may overfit to procedural artifact generation patterns rather than learning genuine cellular morphology.

## 25. Next Step
Temporal modeling with LSTM/Transformer to capture longitudinal biomarker trajectories.
"""

with open(docs_dir / "cnn_report.md", "w") as f:
    f.write(md_report)

logger.info("CNN Pipeline successfully completed!")
