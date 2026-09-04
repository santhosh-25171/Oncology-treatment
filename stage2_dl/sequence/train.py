import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stage2_dl.data.sequence_loader import OncologySequenceLoader
from stage2_dl.data.preprocessing import SequencePreprocessor
from stage2_dl.sequence.lstm_model import LongitudinalLSTM

class SequenceDataset(Dataset):
    def __init__(self, X_raw, y, preprocessor, is_train=False):
        """
        X_raw: Original sequences (with NaNs for padding)
        y: Labels
        preprocessor: The fitted (or to-be-fitted) SequencePreprocessor
        """
        self.lengths = (~np.isnan(X_raw)).any(axis=2).sum(axis=1)
        
        if is_train:
            preprocessor.fit(X_raw)
            
        self.X_scaled = preprocessor.transform(X_raw)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_scaled[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        return x, y, length

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_class_weights(y_train):
    class_counts = np.bincount(y_train)
    total = len(y_train)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def train_lstm():
    set_seed(42)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    seq_path = os.path.join(base_dir, 'data', 'stage2_dl', 'sequences', 'raw', 'synthetic_longitudinal_oncology.csv')
    artifacts_dir = os.path.join(base_dir, 'stage2_dl', 'artifacts')
    os.makedirs(os.path.join(artifacts_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, 'figures'), exist_ok=True)

    # 1. Load Data
    loader = OncologySequenceLoader(seq_path)
    (X_tr, y_tr, _), (X_va, y_va, _), _ = loader.load_and_split()
    
    preprocessor = SequencePreprocessor()
    
    train_dataset = SequenceDataset(X_tr, y_tr, preprocessor, is_train=True)
    val_dataset = SequenceDataset(X_va, y_va, preprocessor, is_train=False)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Model & Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongitudinalLSTM(input_size=3, hidden_size=64, num_layers=1, num_classes=2, dropout_rate=0.5).to(device)
    
    class_weights = calculate_class_weights(y_tr).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    num_epochs = 100
    early_stopping_patience = 15
    best_val_macro_f1 = 0.0
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_preds, all_train_labels = [], []
        
        for inputs, labels, lengths in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs, lengths=lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, lengths=lengths)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        
        scheduler.step(epoch_val_f1)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_f1s.append(epoch_train_f1)
        val_f1s.append(epoch_val_f1)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {epoch_train_loss:.4f} | Val F1: {epoch_val_f1:.4f}")
        
        if epoch_val_f1 > best_val_macro_f1:
            best_val_macro_f1 = epoch_val_f1
            epochs_no_improve = 0
            
            checkpoint_path = os.path.join(artifacts_dir, 'models', 'best_lstm_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_macro_f1': best_val_macro_f1,
                'class_mapping': {0: 'Non-Responder', 1: 'Responder'},
                'preprocessor_means': preprocessor.feature_means,
                'preprocessor_stds': preprocessor.feature_stds,
                'model_config': {'input_size': 3, 'hidden_size': 64, 'num_layers': 1, 'dropout_rate': 0.5}
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss vs Epochs (LSTM)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_f1s, label='Train Macro F1')
    ax2.plot(val_f1s, label='Val Macro F1')
    ax2.set_title('Macro F1 vs Epochs (LSTM)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'figures', 'lstm_training_curves.png'))
    plt.close()

if __name__ == "__main__":
    train_lstm()
