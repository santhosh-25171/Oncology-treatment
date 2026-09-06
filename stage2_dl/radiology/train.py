import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage2_dl.radiology.radiology_dataset import RadiologyDatasetLoader, RadiologyDataset
from stage2_dl.radiology.cnn_model import RadiologyCNN

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_class_weights(labels):
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def train_radiology_model():
    set_seed(42)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifacts_dir = os.path.join(base_dir, 'stage2_dl', 'artifacts')
    os.makedirs(os.path.join(artifacts_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, 'figures'), exist_ok=True)

    # 1. Load Dataset Slices
    loader = RadiologyDatasetLoader()
    tr_paths, tr_labels = loader.get_image_paths_and_labels('train')
    va_paths, va_labels = loader.get_image_paths_and_labels('val')
    
    train_ds = RadiologyDataset(tr_paths, tr_labels)
    val_ds = RadiologyDataset(va_paths, va_labels)
    
    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 2. Model & Loss Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadiologyCNN(num_classes=len(loader.classes), dropout_rate=0.5).to(device)
    
    weights = calculate_class_weights(tr_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    num_epochs = 100
    early_stopping_patience = 15
    best_val_f1 = 0.0
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        tr_preds, tr_targets = [], []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            tr_preds.extend(preds.cpu().numpy())
            tr_targets.extend(targets.cpu().numpy())
            
        epoch_tr_loss = running_train_loss / len(train_ds)
        epoch_tr_f1 = f1_score(tr_targets, tr_preds, average='macro')
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        va_preds, va_targets = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                va_preds.extend(preds.cpu().numpy())
                va_targets.extend(targets.cpu().numpy())
                
        epoch_va_loss = running_val_loss / len(val_ds)
        epoch_va_f1 = f1_score(va_targets, va_preds, average='macro')
        
        scheduler.step(epoch_va_f1)
        
        train_losses.append(epoch_tr_loss)
        val_losses.append(epoch_va_loss)
        train_f1s.append(epoch_tr_f1)
        val_f1s.append(epoch_va_f1)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {epoch_tr_loss:.4f} | Val F1: {epoch_va_f1:.4f}", flush=True)
        
        if epoch_va_f1 > best_val_f1:
            best_val_f1 = epoch_va_f1
            epochs_no_improve = 0
            
            checkpoint_path = os.path.join(artifacts_dir, 'models', 'best_radiology_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_macro_f1': best_val_f1,
                'class_mapping': loader.classes,
                'input_shape': (1, 128, 128)
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
                break
                
    # Plotting Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Radiology CNN Loss vs Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_f1s, label='Train Macro F1')
    ax2.plot(val_f1s, label='Val Macro F1')
    ax2.set_title('Radiology CNN Macro F1 vs Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'figures', 'radiology_training_curves.png'))
    plt.close()
    
    print(f"[SUCCESS] Radiology CNN training complete. Best Val Macro F1: {best_val_f1:.4f}", flush=True)

if __name__ == "__main__":
    train_radiology_model()
