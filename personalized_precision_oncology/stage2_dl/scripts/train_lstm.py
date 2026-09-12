import os
import sys
import time
import json
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Add src to path to import our modules
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.models.lstm import LSTMProgressionModel
from src.data.temporal_dataset import TemporalSequenceDataset, temporal_collate_fn

# Configure Paths
data_dir = base_dir / "sample_data"
artifacts_dir = base_dir / "artifacts"
prep_dir = artifacts_dir / "models" / "temporal_preprocessing"
models_dir = artifacts_dir / "models"
results_dir = artifacts_dir / "results" / "lstm"
docs_dir = base_dir / "docs"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# 1. Load Preprocessed Artifacts & Data
logger.info("Loading preprocessing artifacts and data...")
with open(prep_dir / "temporal_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(prep_dir / "imputation_dict.pkl", "rb") as f:
    imputation_dict = pickle.load(f)
with open(prep_dir / "feature_config.json", "r") as f:
    feature_config = json.load(f)

INPUT_FEATURES = feature_config['input_features']
NUMERIC_FEATURES = feature_config['numeric']

temporal_df = pd.read_csv(data_dir / "temporal" / "biomarker_timeseries.csv")
targets_df = pd.read_csv(data_dir / "temporal" / "progression_targets.csv")
splits_df = pd.read_csv(data_dir / "train_validation_test_split.csv")

if 'split' not in temporal_df.columns:
    temporal_df = temporal_df.merge(splits_df, on='patient_id', how='left')

# Integrity Check
train_pats = set(splits_df[splits_df['split'] == 'train']['patient_id'])
val_pats = set(splits_df[splits_df['split'] == 'validation']['patient_id'])
test_pats = set(splits_df[splits_df['split'] == 'test']['patient_id'])

if train_pats.intersection(val_pats) or train_pats.intersection(test_pats) or val_pats.intersection(test_pats):
    logger.error("DATA LEAKAGE DETECTED! Patients cross splits.")
    sys.exit(1)

# Apply preprocessing
logger.info("Applying saved preprocessing steps...")
for col, val in imputation_dict.items():
    if col in temporal_df.columns:
        temporal_df[col] = temporal_df[col].fillna(val)

categorical_present = [col for col in ['treatment_status', 'treatment_cycle', 'treatment_type', 'response_status'] if col in temporal_df.columns]
if categorical_present:
    temporal_df = pd.get_dummies(temporal_df, columns=categorical_present, dummy_na=False)

# Ensure all INPUT_FEATURES are present (fill with 0 if missing due to categorical OHE variations)
for col in INPUT_FEATURES:
    if col not in temporal_df.columns:
        temporal_df[col] = 0

numeric_present = [col for col in NUMERIC_FEATURES if col in temporal_df.columns]
temporal_df[numeric_present] = scaler.transform(temporal_df[numeric_present])

# Sort chronologically
temporal_df = temporal_df.sort_values(by=['patient_id', 'study_day'])

# Validate no targets in inputs
TARGET_COLS = ['progression_90d', 'future_tumor_volume_cm3', 'future_tumor_growth_rate', 'response_category_90d']
if any(t in INPUT_FEATURES for t in TARGET_COLS):
    logger.error("TARGET LEAKAGE DETECTED! Input features contain target variables.")
    sys.exit(1)

# Datasets
train_df = temporal_df[temporal_df['split'] == 'train']
val_df = temporal_df[temporal_df['split'] == 'validation']
test_df = temporal_df[temporal_df['split'] == 'test']

train_ds = TemporalSequenceDataset(train_df, targets_df, INPUT_FEATURES)
val_ds = TemporalSequenceDataset(val_df, targets_df, INPUT_FEATURES)
test_ds = TemporalSequenceDataset(test_df, targets_df, INPUT_FEATURES)

# Determine class weights based on train set only
train_targets = targets_df[targets_df['patient_id'].isin(train_pats)]['progression_90d']
class_counts = train_targets.value_counts().sort_index()
weights = [1.0 / count for count in class_counts]
weights = torch.FloatTensor([w / sum(weights) for w in weights])
logger.info(f"Class weights (0, 1): {weights.numpy()}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# Model
model = LSTMProgressionModel(input_size=len(INPUT_FEATURES), hidden_size=64, num_layers=2, num_classes=2, dropout=0.2).to(device)

def get_dataloaders(batch_size):
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=temporal_collate_fn)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn)
    return train_ld, val_ld, test_ld

# Smoke test and Batch Size fallback
batch_sizes = [32, 16, 8]
train_loader, val_loader, test_loader = None, None, None
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

logger.info("Running Smoke Test...")
for bz in batch_sizes:
    try:
        train_loader, val_loader, test_loader = get_dataloaders(bz)
        model.train()
        batch = next(iter(train_loader))
        x = batch['padded_sequences'].to(device)
        lengths = batch['lengths']
        y = batch['targets_progression'].to(device)
        
        logger.info(f"First batch input shape: {x.shape}")
        
        optimizer.zero_grad()
        logits = model(x, lengths)
        logger.info(f"First batch output shape: {logits.shape}")
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("NaN or Inf detected in output.")
            
        logger.info(f"Smoke test passed with batch size {bz}")
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            logger.warning(f"OOM with batch size {bz}, reducing...")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        else:
            raise e

# Training
EPOCHS = 10
patience = 3
best_val_f1 = 0.0
epochs_no_improve = 0
best_epoch = 0

train_losses, val_losses = [], []
train_f1s, val_f1s = [], []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # Train
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch in train_loader:
        x = batch['padded_sequences'].to(device)
        lengths = batch['lengths']
        y = batch['targets_progression'].to(device)
        
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    epoch_train_loss = running_loss / len(train_ds)
    epoch_train_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Val
    model.eval()
    val_loss = 0.0
    all_val_preds, all_val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['padded_sequences'].to(device)
            lengths = batch['lengths']
            y = batch['targets_progression'].to(device)
            
            logits = model(x, lengths)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            
            _, preds = torch.max(logits, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(y.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_ds)
    epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_f1s.append(epoch_train_f1)
    val_f1s.append(epoch_val_f1)
    
    epoch_time = time.time() - epoch_start
    logger.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Macro-F1: {epoch_val_f1:.4f} | Time: {epoch_time:.1f}s")
    
    if epoch_val_f1 > best_val_f1:
        best_val_f1 = epoch_val_f1
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': len(INPUT_FEATURES),
            'hidden_size': 64,
            'num_layers': 2,
            'num_classes': 2,
            'best_val_f1': best_val_f1,
            'training_config': {'epochs': EPOCHS, 'batch_size': bz, 'optimizer': 'adam', 'lr': 0.001}
        }, models_dir / "lstm_best.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered.")
            break

# Evaluation
logger.info("Evaluating on Test Set...")
checkpoint = torch.load(models_dir / "lstm_best.pt", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_preds, test_probs, test_labels = [], [], []
with torch.no_grad():
    for batch in test_loader:
        x = batch['padded_sequences'].to(device)
        lengths = batch['lengths']
        y = batch['targets_progression'].to(device)
        
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)[:, 1]
        _, preds = torch.max(logits, 1)
        
        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(y.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
test_prec = precision_score(test_labels, test_preds, zero_division=0)
test_rec = recall_score(test_labels, test_preds, zero_division=0)
test_f1 = f1_score(test_labels, test_preds, average='macro')
try:
    test_auc = roc_auc_score(test_labels, test_probs)
except ValueError:
    test_auc = 0.0

metrics = {
    "accuracy": float(test_acc),
    "precision": float(test_prec),
    "recall": float(test_rec),
    "macro_f1": float(test_f1),
    "roc_auc": float(test_auc),
    "best_epoch": best_epoch
}

with open(results_dir / "lstm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Plots
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('LSTM Training and Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(results_dir / "training_curves_loss.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(train_f1s, label='Train Macro-F1')
plt.plot(val_f1s, label='Val Macro-F1')
plt.title('LSTM Training and Validation Macro-F1')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(results_dir / "training_curves_f1.png")
plt.close()

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Progression', 'Progression'], yticklabels=['No Progression', 'Progression'])
plt.title('Test Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig(results_dir / "confusion_matrix.png")
plt.close()

# Report
report = f"""# Stage 02 Deep Learning — LSTM Temporal Modeling Report

## Purpose
Following the baseline CNN spatial analysis, the LSTM model handles the sequential, chronological dependencies of patient biomarkers and tumor volume measurements. It maps dynamic longitudinal data to predict a future state.

## Input
- **Features**: {len(INPUT_FEATURES)} normalized features including tumor volumes, multi-gene markers (ctDNA, CEA, LDH), and treatment status encodings.
- **Sequence construction**: Patient observations were ordered chronologically.

## Variable sequence lengths
Sequences vary dynamically (e.g., 25-34 observations). We deployed zero-padding alongside `pack_padded_sequence` so the LSTM strictly processes meaningful timesteps and ignores artifacts introduced by padding.

## Architecture
- Input features ({len(INPUT_FEATURES)}) → LSTM (2 layers, 64 hidden, batch_first, dropout=0.2)
- Final hidden representation extracted effectively corresponding to the exact sequence length (un-padded).
- Dropout (0.3) → Linear (64 -> 2)
- Predicting `progression_90d` (0 = Stable/Responsive, 1 = Progression).

## Why LSTM
LSTMs intrinsically persist hidden states across time, capturing critical rates of change (e.g., aggressive tumor growth spurts or rebounding ctDNA levels) that are otherwise lost in non-sequential algorithms.

## Training Configuration
- Optimizer: Adam (lr=0.001)
- Loss: Weighted CrossEntropyLoss to address class imbalance.
- Batch size: {bz}
- Best Epoch: {best_epoch}

## Evaluation Results (Test Set)
- **Accuracy**: {test_acc:.4f}
- **Precision**: {test_prec:.4f}
- **Recall**: {test_rec:.4f}
- **Macro-F1**: {test_f1:.4f}
- **ROC-AUC**: {test_auc:.4f}

![Confusion Matrix](../artifacts/results/lstm/confusion_matrix.png)

## Limitations
**This LSTM is a research/educational prototype trained on synthetic clinical data.** It is categorically NOT clinically validated and cannot safely diagnose or predict actual cancer progression. The simulated data may possess artifacts or simplistic patterns that an LSTM exploits, which do not translate to real human biology.
"""
with open(docs_dir / "lstm_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\nLSTM TEMPORAL MODEL COMPLETE")
print(f"Dataset:\nPatients: {len(train_ds) + len(val_ds) + len(test_ds)}")
print(f"Input features: {len(INPUT_FEATURES)}")
print(f"Sequence length: {temporal_df.groupby('patient_id').size().min()}–{temporal_df.groupby('patient_id').size().max()}")
print("\nTask:\n90-day progression prediction")
print(f"\nBest epoch: {best_epoch}")
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall: {test_rec:.4f}")
print(f"Test Macro-F1: {test_f1:.4f}")
print(f"Test ROC-AUC: {test_auc:.4f}")
print("\nCheckpoint:\nstage2_dl/artifacts/models/lstm_best.pt")
print("\nResults:\nstage2_dl/artifacts/results/lstm/")
print("\nReport:\nstage2_dl/docs/lstm_report.md")
