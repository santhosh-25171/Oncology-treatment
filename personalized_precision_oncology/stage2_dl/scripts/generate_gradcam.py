import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

# Add project root to sys.path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.models.cnn import BaselineCNN
from src.explainability.gradcam import GradCAM

def main():
    print("========================================")
    print("GRAD-CAM VISUAL SALIENCY MAP GENERATION")
    print("========================================")
    
    models_dir = base_dir / "artifacts" / "models"
    results_dir = base_dir / "artifacts" / "results" / "gradcam"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    cnn_path = models_dir / "cnn_best.pt"
    if not cnn_path.exists():
        print(f"[ERROR] Trained CNN checkpoint not found at {cnn_path}")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    classes = ['normal', 'benign', 'malignant', 'tumor_margin', 'necrotic', 'inflammatory']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    model = BaselineCNN(num_classes=6).to(device)
    checkpoint = torch.load(cnn_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[SUCCESS] Loaded trained CNN model.")
    
    # Attach GradCAM to block3 conv layer
    target_layer = model.block3[0]
    gradcam = GradCAM(model, target_layer)
    
    # Load test metadata
    sample_dir = base_dir / "sample_data"
    labels_csv = sample_dir / "image_labels_sample_1000.csv"
    df = pd.read_csv(labels_csv)
    test_df = df[df['split'] == 'test']
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gradcam_records = []
    
    # Select 1 test sample per tissue class
    fig, axes = plt.subplots(len(classes), 3, figsize=(12, 4 * len(classes)))
    
    for row_idx, cls in enumerate(classes):
        cls_samples = test_df[test_df['tissue_class'] == cls]
        if len(cls_samples) == 0:
            print(f"[WARN] No test samples for class {cls}")
            continue
            
        sample_row = cls_samples.iloc[0]
        raw_path = Path(sample_row['image_path'])
        if raw_path.exists():
            img_path = raw_path
        elif (sample_dir / sample_row['image_path']).exists():
            img_path = sample_dir / sample_row['image_path']
        else:
            img_path = base_dir.parent / raw_path
            
        orig_img = Image.open(img_path).convert("RGB").resize((224, 224))
        orig_np = np.array(orig_img)
        
        tensor = transform(orig_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            pred_class = idx_to_class[pred_idx]
            conf = float(probs[pred_idx])
            
        heatmap, _ = gradcam.generate_heatmap(tensor, class_idx=pred_idx)
        overlay = GradCAM.overlay_heatmap(orig_np, heatmap, alpha=0.5, colormap='jet')
        
        # Save individual plot
        single_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.imshow(orig_np)
        ax1.set_title(f"Original Image\nTrue: {cls}", fontsize=11)
        ax1.axis('off')
        
        im2 = ax2.imshow(heatmap, cmap='jet')
        ax2.set_title(f"Grad-CAM Heatmap\nLayer: Block 3 Conv", fontsize=11)
        ax2.axis('off')
        single_fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax3.imshow(overlay)
        ax3.set_title(f"Overlay\nPred: {pred_class} ({conf:.1%})", fontsize=11)
        ax3.axis('off')
        
        single_fig.tight_layout()
        single_plot_path = results_dir / f"gradcam_{cls}.png"
        single_fig.savefig(single_plot_path, dpi=150)
        plt.close(single_fig)
        
        # Add to summary figure
        axes[row_idx, 0].imshow(orig_np)
        axes[row_idx, 0].set_title(f"True: {cls}", fontsize=10)
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(heatmap, cmap='jet')
        axes[row_idx, 1].set_title(f"Grad-CAM Heatmap", fontsize=10)
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(f"Pred: {pred_class} ({conf:.1%})", fontsize=10)
        axes[row_idx, 2].axis('off')
        
        gradcam_records.append({
            "patient_id": sample_row['patient_id'],
            "true_class": cls,
            "predicted_class": pred_class,
            "confidence": conf,
            "image_path": str(img_path),
            "output_plot": str(single_plot_path),
            "peak_activation": float(np.max(heatmap))
        })
        print(f"Generated Grad-CAM for {cls}: Pred={pred_class} (Conf={conf:.2%})")
        
    fig.tight_layout()
    summary_path = results_dir / "gradcam_all_classes_summary.png"
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"[SUCCESS] Saved multi-class Grad-CAM summary to {summary_path}")
    
    report_json = results_dir / "gradcam_results.json"
    with open(report_json, "w") as f:
        json.dump({
            "target_layer": "block3.0",
            "model_architecture": "BaselineCNN",
            "num_evaluated_classes": len(gradcam_records),
            "results": gradcam_records
        }, f, indent=4)
    print(f"[SUCCESS] Saved Grad-CAM report metadata to {report_json}")
    
    gradcam.remove_hooks()
    print("========================================")
    print("GRAD-CAM COMPLETE")
    print("========================================")

if __name__ == "__main__":
    main()
