import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stage2_dl.data.sequence_loader import OncologySequenceLoader
from stage2_dl.data.preprocessing import SequencePreprocessor
from stage2_dl.sequence.transformer_model import SequenceTransformer
from stage2_dl.sequence.train import SequenceDataset

def evaluate_transformer():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    seq_path = os.path.join(base_dir, 'data', 'stage2_dl', 'sequences', 'raw', 'synthetic_longitudinal_oncology.csv')
    artifacts_dir = os.path.join(base_dir, 'stage2_dl', 'artifacts')
    os.makedirs(os.path.join(artifacts_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, 'figures'), exist_ok=True)

    checkpoint_path = os.path.join(artifacts_dir, 'models', 'best_transformer_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    model = SequenceTransformer(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loader = OncologySequenceLoader(seq_path)
    (X_tr, _, _), _, (X_te, y_te, _) = loader.load_and_split()
    
    preprocessor = SequencePreprocessor()
    preprocessor.fit(X_tr) 
    
    if 'preprocessor_means' in checkpoint:
        preprocessor.feature_means = checkpoint['preprocessor_means']
        preprocessor.feature_stds = checkpoint['preprocessor_stds']

    test_dataset = SequenceDataset(X_te, y_te, preprocessor, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, lengths in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs, lengths=lengths)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    prec_class, rec_class, f1_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

    metrics_dict = {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "roc_auc": float(roc_auc),
        "classes": {
            "Responder": {
                "precision": float(prec_class[1]),
                "recall": float(rec_class[1]),
                "f1": float(f1_class[1])
            },
            "Non-Responder": {
                "precision": float(prec_class[0]),
                "recall": float(rec_class[0]),
                "f1": float(f1_class[0])
            }
        }
    }

    with open(os.path.join(artifacts_dir, 'metrics', 'transformer_test_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Responder (0)', 'Responder (1)'])
    ax.set_yticklabels(['Non-Responder (0)', 'Responder (1)'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Transformer Confusion Matrix')
    plt.savefig(os.path.join(artifacts_dir, 'figures', 'transformer_confusion_matrix.png'))
    plt.close()

    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC area = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Transformer ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(artifacts_dir, 'figures', 'transformer_roc_curve.png'))
    plt.close()

    print(json.dumps(metrics_dict, indent=4))

if __name__ == "__main__":
    evaluate_transformer()
