import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from stage2_dl.data.sequence_loader import OncologySequenceLoader
from stage2_dl.data.preprocessing import SequencePreprocessor
from stage2_dl.sequence.forecast_model import TransformerForecaster, LSTMForecaster

class ForecastDataset(Dataset):
    """
    Dataset for longitudinal sequence trajectory forecasting.
    Given sequence observations up to time step t-1, predicts the next-step value (ctDNA level / tumor volume).
    """
    def __init__(self, X_raw, preprocessor, is_train=False, target_feature_idx=0, output_dim=1):
        """
        X_raw: (N, T, F) raw sequence array
        preprocessor: SequencePreprocessor instance
        target_feature_idx: 0 for ctdna_level, 2 for tumor_volume, or None for multi-feature
        """
        self.output_dim = output_dim
        self.target_feature_idx = target_feature_idx
        
        if is_train:
            preprocessor.fit(X_raw)
            
        self.X_scaled = preprocessor.transform(X_raw)
        
        self.inputs = []
        self.targets = []
        self.lengths = []
        
        for idx in range(len(X_raw)):
            seq = X_raw[idx]
            seq_scaled = self.X_scaled[idx]
            
            # Count valid (non-NaN) steps
            valid_mask = ~np.isnan(seq).any(axis=1)
            valid_len = np.sum(valid_mask)
            
            if valid_len >= 2:
                # Input sequence up to step valid_len-2 (length valid_len-1)
                in_len = valid_len - 1
                in_seq = np.zeros_like(seq_scaled)
                in_seq[:in_len] = seq_scaled[:in_len]
                
                # Next-step target at step valid_len-1
                if output_dim == 1:
                    target_val = seq_scaled[valid_len - 1, target_feature_idx]
                    self.targets.append(np.array([target_val], dtype=np.float32))
                else:
                    target_val = seq_scaled[valid_len - 1]
                    self.targets.append(target_val.astype(np.float32))
                    
                self.inputs.append(in_seq.astype(np.float32))
                self.lengths.append(in_len)
                
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)
        self.lengths = np.array(self.lengths)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        return x, y, length

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_forecast_model():
    set_seed(42)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    seq_path = os.path.join(base_dir, 'data', 'stage2_dl', 'sequences', 'raw', 'synthetic_longitudinal_oncology.csv')
    artifacts_dir = os.path.join(base_dir, 'stage2_dl', 'artifacts')
    os.makedirs(os.path.join(artifacts_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, 'figures'), exist_ok=True)

    # 1. Load Data with identical splits
    loader = OncologySequenceLoader(seq_path)
    (X_tr, _, _), (X_va, _, _), _ = loader.load_and_split()
    
    preprocessor = SequencePreprocessor()
    
    train_dataset = ForecastDataset(X_tr, preprocessor, is_train=True, target_feature_idx=0, output_dim=1)
    val_dataset = ForecastDataset(X_va, preprocessor, is_train=False, target_feature_idx=0, output_dim=1)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Model & Loss Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use TransformerForecaster as primary sequence architecture
    model = TransformerForecaster(input_size=3, d_model=32, nhead=4, num_layers=2, output_dim=1).to(device)
    
    # Loss: HuberLoss (Smooth L1 Loss) for robustness to noisy biomarker trajectories
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    num_epochs = 100
    early_stopping_patience = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        
        for inputs, targets, lengths in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            preds = model(inputs, lengths=lengths)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, lengths in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs, lengths=lengths)
                loss = criterion(preds, targets)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        scheduler.step(epoch_val_loss)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1:03d} | Train Huber Loss: {epoch_train_loss:.4f} | Val Huber Loss: {epoch_val_loss:.4f}", flush=True)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            
            checkpoint_path = os.path.join(artifacts_dir, 'models', 'best_forecast_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'preprocessor_means': preprocessor.feature_means,
                'preprocessor_stds': preprocessor.feature_stds,
                'model_config': {'input_size': 3, 'd_model': 32, 'nhead': 4, 'num_layers': 2, 'output_dim': 1},
                'target_feature': 'ctdna_level'
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
                break
                
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Huber Loss')
    plt.plot(val_losses, label='Val Huber Loss')
    plt.title('Trajectory Forecasting Loss vs Epochs (Transformer Forecaster)')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'figures', 'forecast_training_curves.png'))
    plt.close()
    
    print(f"[SUCCESS] Trajectory Forecaster training complete. Best Val Huber Loss: {best_val_loss:.4f}", flush=True)

if __name__ == "__main__":
    train_forecast_model()
