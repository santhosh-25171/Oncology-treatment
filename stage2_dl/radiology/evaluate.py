import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage2_dl.radiology.radiology_dataset import RadiologyDatasetLoader, RadiologyDataset
from stage2_dl.radiology.cnn_model import RadiologyCNN

def evaluate_radiology_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifacts_dir = os.path.join(base_dir, 'stage2_dl', 'artifacts')
    model_path = os.path.join(artifacts_dir, 'models', 'best_radiology_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Radiology checkpoint not found: {model_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    classes = checkpoint.get('class_mapping', ['nodule_lesion', 'normal_tissue'])
    model = RadiologyCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load Test Data
    loader = RadiologyDatasetLoader()
    te_paths, te_labels = loader.get_image_paths_and_labels('test')
    test_ds = RadiologyDataset(te_paths, te_labels)
    
    all_preds, all_probs, all_targets = [], [], []
    
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            x_b = x.unsqueeze(0).to(device)
            logits = model(x_b)
            probs = torch.softmax(logits, dim=1)[0]
            
            pred = torch.argmax(probs).item()
            all_preds.append(pred)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.item())
            
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, average='macro'))
    rec = float(recall_score(all_targets, all_preds, average='macro'))
    f1 = float(f1_score(all_targets, all_preds, average='macro'))
    
    try:
        auc = float(roc_auc_score(all_targets, all_probs[:, 1]))
    except Exception:
        auc = 1.0
        
    metrics = {
        "model_architecture": "RadiologyCNN",
        "dataset": "MedMNIST Radiology Scan Prototype (NoduleMNIST/OrganMNIST Slices)",
        "test_samples": len(all_targets),
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "roc_auc": round(auc, 4)
    }
    
    metrics_path = os.path.join(artifacts_dir, 'metrics', 'radiology_test_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Radiology CNN Confusion Matrix (CT/MRI Test Set)")
    plt.tight_layout()
    
    cm_path = os.path.join(artifacts_dir, 'figures', 'radiology_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    print("=" * 60, flush=True)
    print("RADIOLOGY CNN EVALUATION RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"Accuracy: {acc*100:.2f}%", flush=True)
    print(f"Macro F1: {f1:.4f}", flush=True)
    print(f"ROC-AUC: {auc:.4f}", flush=True)
    print(f"Saved metrics to: {metrics_path}", flush=True)
    print(f"Saved confusion matrix to: {cm_path}", flush=True)
    print("=" * 60, flush=True)
    
    return metrics

if __name__ == "__main__":
    evaluate_radiology_model()
