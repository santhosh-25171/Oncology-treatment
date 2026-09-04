import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stage2_dl.data.image_loader import OncologyImageLoader
from stage2_dl.data.preprocessing import ImagePreprocessor
from stage2_dl.vision.cnn_model import LightweightCNN
from stage2_dl.vision.train import BreastMNISTDataset

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_dir = os.path.join(base_dir, 'data', 'stage2_dl', 'images')
    artifacts_dir = os.path.join(base_dir, 'stage2_dl', 'artifacts')
    os.makedirs(os.path.join(artifacts_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, 'figures'), exist_ok=True)

    # 1. Load Test Data
    loader = OncologyImageLoader(img_dir)
    test_data, _ = loader.load_dataset('test')
    preprocessor = ImagePreprocessor(target_size=(128, 128))
    test_dataset = BreastMNISTDataset(test_data, preprocessor, augment=None)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightCNN(num_classes=2).to(device)
    checkpoint_path = os.path.join(artifacts_dir, 'models', 'best_cnn_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Inference
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 4. Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    prec_class, rec_class, f1_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    # Assuming class 0 is malignant and class 1 is normal_benign
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

    metrics_dict = {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "roc_auc": roc_auc,
        "classes": {
            "malignant": {
                "precision": prec_class[0],
                "recall": rec_class[0],
                "f1": f1_class[0]
            },
            "normal_benign": {
                "precision": prec_class[1],
                "recall": rec_class[1],
                "f1": f1_class[1]
            }
        }
    }

    with open(os.path.join(artifacts_dir, 'metrics', 'cnn_test_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    print(json.dumps(metrics_dict, indent=4))

    # 5. Plotting
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Malignant (0)', 'Normal/Benign (1)'])
    ax.set_yticklabels(['Malignant (0)', 'Normal/Benign (1)'])
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center')
            
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on Unseen Test Set')
    plt.savefig(os.path.join(artifacts_dir, 'figures', 'confusion_matrix.png'))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(artifacts_dir, 'figures', 'roc_curve.png'))
    plt.close()

if __name__ == "__main__":
    evaluate()
