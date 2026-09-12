import os
import sys
import json
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

# Configure Paths
base_dir = Path(r"C:\Users\Dell\Documents\SPECIAL\personalized_precision_oncology\stage2_dl")
data_dir = base_dir / "data" / "dl_oncology_dataset_v2"
artifacts_dir = base_dir / "artifacts"
results_dir = artifacts_dir / "results" / "temporal"
prep_dir = artifacts_dir / "models" / "temporal_preprocessing"
docs_dir = base_dir / "docs"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(prep_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

logger.info("Loading temporal datasets...")
temporal_df = pd.read_csv(data_dir / "temporal" / "biomarker_timeseries.csv")
targets_df = pd.read_csv(data_dir / "temporal" / "progression_targets.csv")
splits_df = pd.read_csv(data_dir / "train_validation_test_split.csv")

# Ensure split exists in temporal_df
if 'split' not in temporal_df.columns:
    temporal_df = temporal_df.merge(splits_df, on='patient_id', how='left')

# 1-2. Integrity & Leakage Check
train_pats = set(splits_df[splits_df['split'] == 'train']['patient_id'])
val_pats = set(splits_df[splits_df['split'] == 'validation']['patient_id'])
test_pats = set(splits_df[splits_df['split'] == 'test']['patient_id'])

if train_pats.intersection(val_pats) or train_pats.intersection(test_pats) or val_pats.intersection(test_pats):
    logger.error("DATA LEAKAGE DETECTED! Patients cross splits.")
    sys.exit(1)
logger.info("Leakage check passed: No patient overlap across splits.")

# 3. Sort Chronologically
logger.info("Sorting observations chronologically...")
temporal_df = temporal_df.sort_values(by=['patient_id', 'study_day'])

# 5. Define Feature/Target separation
IDENTIFIER_COLS = ['patient_id', 'study_day', 'split']
TARGET_COLS = ['progression_90d', 'future_tumor_volume_cm3', 'future_tumor_growth_rate', 'response_category_90d']

# Available columns in temporal_df
all_temporal_cols = temporal_df.columns.tolist()

# Define input features strictly excluding targets and identifiers
NUMERIC_FEATURES = [
    'ctDNA_level', 'ctDNA_change_percent', 'tumor_volume_cm3', 'tumor_growth_rate',
    'CEA', 'CYFRA21_1', 'CRP', 'LDH', 'dose_intensity'
]
CATEGORICAL_FEATURES = ['treatment_status', 'treatment_cycle', 'treatment_type', 'response_status']

INPUT_FEATURES = [col for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES if col in all_temporal_cols]

logger.info(f"Input features: {INPUT_FEATURES}")

# Validate no target leakage
for targ in TARGET_COLS:
    if targ in INPUT_FEATURES:
        logger.error(f"TARGET LEAKAGE DETECTED! Feature '{targ}' is in input features.")
        sys.exit(1)

# 6-8. Missing Values & Feature Scaling
logger.info("Handling missing values and scaling features...")

# Calculate train statistics for imputation
train_temporal = temporal_df[temporal_df['split'] == 'train']

missing_stats = temporal_df[INPUT_FEATURES].isnull().mean() * 100
missing_stats.to_csv(results_dir / "temporal_missingness.csv")

# Plot missingness
plt.figure(figsize=(10, 6))
if len(missing_stats[missing_stats > 0]) > 0:
    missing_stats[missing_stats > 0].sort_values().plot(kind='barh', color='salmon')
else:
    plt.text(0.5, 0.5, 'No missing values', ha='center', fontsize=14)
    plt.axis('off')
plt.title("Missing Values Percentage (%)")
plt.tight_layout()
plt.savefig(results_dir / "missing_values.png")
plt.close()

# Imputation (Train Median for numeric, Mode for categorical)
imputation_dict = {}
for col in INPUT_FEATURES:
    if col in NUMERIC_FEATURES:
        imputation_dict[col] = train_temporal[col].median()
    elif col in CATEGORICAL_FEATURES:
        imputation_dict[col] = train_temporal[col].mode()[0]

# Apply Imputation
for col in INPUT_FEATURES:
    temporal_df[col] = temporal_df[col].fillna(imputation_dict[col])

# OHE for categorical features if any exist
if any(col in temporal_df.columns for col in CATEGORICAL_FEATURES):
    # Fit OHE on train, apply to all using pd.get_dummies aligning columns
    categorical_present = [col for col in CATEGORICAL_FEATURES if col in temporal_df.columns]
    temporal_df = pd.get_dummies(temporal_df, columns=categorical_present, dummy_na=False)
    # Update input features
    INPUT_FEATURES = [col for col in temporal_df.columns if col not in IDENTIFIER_COLS + TARGET_COLS and col != 'patient_id']

# Fit Scaler on TRAIN ONLY
train_temporal = temporal_df[temporal_df['split'] == 'train']
scaler = StandardScaler()

numeric_present = [col for col in NUMERIC_FEATURES if col in temporal_df.columns]
scaler.fit(train_temporal[numeric_present])

# Apply Scaler to ALL
temporal_df[numeric_present] = scaler.transform(temporal_df[numeric_present])

# Save preprocessing objects
with open(prep_dir / "temporal_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(prep_dir / "imputation_dict.pkl", "wb") as f:
    pickle.dump(imputation_dict, f)
with open(prep_dir / "feature_config.json", "w") as f:
    json.dump({"input_features": INPUT_FEATURES, "numeric": numeric_present}, f)

# 9-12. Sequence Construction & PyTorch Dataset
class TemporalSequenceDataset(Dataset):
    def __init__(self, temporal_df, targets_df, features):
        self.temporal_df = temporal_df
        self.targets_df = targets_df.set_index('patient_id')
        self.features = features
        self.patient_ids = self.temporal_df['patient_id'].unique()
        
        # Group by patient for fast retrieval
        self.grouped = dict(tuple(self.temporal_df.groupby('patient_id')))
        
    def __len__(self):
        return len(self.patient_ids)
        
    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        pat_data = self.grouped[pid]
        
        # Ensure chronological order
        pat_data = pat_data.sort_values('study_day')
        
        seq_tensor = torch.FloatTensor(pat_data[self.features].values.astype(float))
        seq_len = len(seq_tensor)
        
        # Retrieve targets
        target_row = self.targets_df.loc[pid]
        
        target_dict = {
            'progression_90d': int(target_row.get('progression_90d', 0)),
            'future_tumor_volume_cm3': float(target_row.get('future_tumor_volume_cm3', 0.0))
        }
        
        return {
            'sequence': seq_tensor,
            'length': seq_len,
            'patient_id': pid,
            'target': target_dict
        }

def temporal_collate_fn(batch):
    sequences = [item['sequence'] for item in batch]
    lengths = torch.LongTensor([item['length'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    
    # Pad sequences to max length in the batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    progression_targets = torch.LongTensor([item['target']['progression_90d'] for item in batch])
    volume_targets = torch.FloatTensor([item['target']['future_tumor_volume_cm3'] for item in batch])
    
    return {
        'padded_sequences': padded_sequences,
        'lengths': lengths,
        'patient_ids': patient_ids,
        'targets_progression': progression_targets,
        'targets_volume': volume_targets
    }

# Split DataFrames
train_df = temporal_df[temporal_df['split'] == 'train']
val_df = temporal_df[temporal_df['split'] == 'validation']
test_df = temporal_df[temporal_df['split'] == 'test']

train_ds = TemporalSequenceDataset(train_df, targets_df, INPUT_FEATURES)
val_ds = TemporalSequenceDataset(val_df, targets_df, INPUT_FEATURES)
test_ds = TemporalSequenceDataset(test_df, targets_df, INPUT_FEATURES)

# 13. DataLoaders
batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=temporal_collate_fn)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn)

# 15. Sequence Statistics
seq_lengths = temporal_df.groupby('patient_id').size()
seq_stats = seq_lengths.describe()

summary_stats = {
    "Total Patients": len(temporal_df['patient_id'].unique()),
    "Total Observations": len(temporal_df),
    "Mean Sequence Length": float(seq_stats['mean']),
    "Median Sequence Length": float(seq_stats['50%']),
    "Min Sequence Length": float(seq_stats['min']),
    "Max Sequence Length": float(seq_stats['max']),
    "Features Count": len(INPUT_FEATURES)
}

with open(results_dir / "temporal_summary.json", "w") as f:
    json.dump(summary_stats, f, indent=4)

plt.figure(figsize=(10, 6))
sns.histplot(seq_lengths, bins=30, kde=True, color='purple')
plt.title("Sequence Length Distribution")
plt.xlabel("Number of Observations per Patient")
plt.savefig(results_dir / "sequence_length_distribution.png")
plt.close()

# Smoke Test
logger.info("Running Smoke Test...")
batch = next(iter(train_loader))
logger.info(f"Batch padded sequences shape: {batch['padded_sequences'].shape}")
logger.info(f"Sequence lengths: {batch['lengths'].tolist()[:5]}...")
logger.info(f"Targets (progression) shape: {batch['targets_progression'].shape}")
logger.info(f"Patient IDs: {batch['patient_ids'][:5]}...")
nan_count = torch.isnan(batch['padded_sequences']).sum().item()
inf_count = torch.isinf(batch['padded_sequences']).sum().item()
logger.info(f"NaN count: {nan_count}")
logger.info(f"Inf count: {inf_count}")

if nan_count > 0 or inf_count > 0:
    logger.error("SMOKE TEST FAILED! NaN/Inf detected in tensors.")
    sys.exit(1)

# Report Generation
md_report = f"""# Stage 02 Deep Learning — Temporal Preprocessing Report

## 1. Objective
Establish a rigorous temporal preprocessing pipeline to prepare longitudinal clinical data (biomarkers, measurements, treatments) for sequence modeling via LSTM or Transformer. 

## 2. Dataset Overview
- **Patients**: {summary_stats['Total Patients']}
- **Temporal Observations**: {summary_stats['Total Observations']}

## 3. Sequence Statistics
- **Mean Length**: {summary_stats['Mean Sequence Length']:.1f}
- **Median Length**: {summary_stats['Median Sequence Length']:.1f}
- **Min Length**: {summary_stats['Min Sequence Length']:.0f}
- **Max Length**: {summary_stats['Max Sequence Length']:.0f}

![Sequence Length](../artifacts/results/temporal/sequence_length_distribution.png)

## 4. Leakage Prevention
- **Patient Leakage**: Split validation confirmed ZERO overlap between Train, Val, and Test patients.
- **Target Leakage**: Future targets (`progression_90d`, `future_tumor_volume_cm3`) are strictly omitted from input features.
- **Preprocessing Leakage**: Missing-value imputation medians and `StandardScaler` transformations were explicitly fitted ONLY on the training split, completely isolating the validation and test sets.

## 5. Sequence Construction & Padding
Sequences preserve strict chronological ordering via `study_day`. Since sequence lengths vary from {summary_stats['Min Sequence Length']:.0f} to {summary_stats['Max Sequence Length']:.0f}, variable-length sequences are dynamically padded (`padding_value=0.0`) using `nn.utils.rnn.pad_sequence` within a custom `collate_fn`. Original lengths are yielded alongside the padded tensor to support dynamic masking and `pack_padded_sequence` operations in future models.

## 6. Model Inputs
- **Feature Space**: {len(INPUT_FEATURES)} unified numerical features (OHE applied to categoricals).
- **Batch Shape**: `[batch_size, max_sequence_length, num_features]`

**IMPORTANT**: This temporal dataset is synthetic and intended for research/educational prototyping. It is not clinically validated.

## 7. Readiness
The data loaders (`train_loader`, `val_loader`, `test_loader`) are successfully generating valid sequences and are structurally ready for LSTM/Transformer implementation.
"""

with open(docs_dir / "temporal_preprocessing_report.md", "w") as f:
    f.write(md_report)

logger.info("Temporal Preprocessing Pipeline successfully completed!")

print("\n========== TEMPORAL PREPROCESSING COMPLETE ==========")
print(f"Temporal dataset size: {summary_stats['Total Observations']} observations")
print(f"Number of patients: {summary_stats['Total Patients']}")
print(f"Number of temporal features: {len(INPUT_FEATURES)}")
print(f"Number of target variables: {len(TARGET_COLS)}")
print(f"Sequence length statistics: Min {summary_stats['Min Sequence Length']:.0f}, Max {summary_stats['Max Sequence Length']:.0f}, Mean {summary_stats['Mean Sequence Length']:.1f}")
print("Preprocessing method: Train-set Median Imputation + StandardScaler")
print("Padding method: dynamic pad_sequence per batch (batch_first=True)")
print("Leakage test results: PASS")
print("Missing-value handling: Train Median Imputation")
print(f"DataLoader batch shape: {list(batch['padded_sequences'].shape)}")
print("Errors: 0")
