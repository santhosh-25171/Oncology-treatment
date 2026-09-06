import os
import sys
import json
import torch
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage2_dl.integration.fuse import OncologyMultimodalFusion

def main():
    print("=" * 70)
    print("      PRECISION ONCOLOGY -- STAGE 2 DEEP LEARNING PIPELINE")
    print("=" * 70)
    
    # 1. Initialize Multimodal Fusion Engine
    print("\n[Step 1/4] Loading Stage 2 Deep Learning Checkpoints & Preprocessors...")
    fusion_engine = OncologyMultimodalFusion()
    print("  [OK] Pathology/Radiology LightweightCNN Loaded (PyTorch)")
    print("  [OK] Transformer Sequential Trajectory Forecaster Loaded (PyTorch)")
    print("  [OK] Calibrated Multimodal Rule-Based Fusion Engine Ready")

    # 2. Prepare Sample Clinical Multimodal Inputs
    print("\n[Step 2/4] Ingesting Multimodal Patient Data Stream...")
    sample_img_path = os.path.join(PROJECT_ROOT, "stage2_dl", "artifacts", "figures", "gradcam_test_img_0.png")
    
    # Sample longitudinal ctDNA + biomarker trajectory (seq_len=4, features=[ctDNA, TumorVol, Marker2])
    sample_seq = np.array([
        [15.0, 100.0, 1.2],
        [25.0, 110.0, 1.4],
        [38.0, 128.0, 1.8],
        [62.0, 155.0, 2.4]
    ])
    
    print(f"  * Image Branch Input: {sample_img_path}")
    print(f"  * Sequence Branch Input: {sample_seq.shape[0]}-step longitudinal ctDNA trajectory")
    print(f"    Latest ctDNA Reading: {sample_seq[-1, 0]} ng/mL (Rising Trend)")

    # 3. Execute Deep Learning Pipeline & Multimodal Risk Fusion
    print("\n[Step 3/4] Running Multimodal Deep Learning Inference...")
    fusion_result = fusion_engine.fuse(sample_img_path, sample_seq)
    
    vision_score = fusion_result["vision_score"]
    trend_score = fusion_result["trend_score"]
    combined_risk = fusion_result["combined_risk"]
    risk_flag = fusion_result["flag"]

    # 4. Display Formatted Clinical Output
    print("\n[Step 4/4] Pipeline Execution Results:")
    print("-" * 50)
    print(f"  - Vision Malignancy Confidence:    {vision_score * 100:.1f}%")
    print(f"  - Sequence Trajectory Velocity:    {trend_score * 100:.1f}%")
    print(f"  - Combined Multimodal Risk Score:  {combined_risk * 100:.1f}%")
    print(f"  - Clinical Triage Risk Flag:       [{risk_flag.upper()} RISK]")
    print("-" * 50)
    
    print("\n[SUCCESS] Stage 2 Deep Learning Pipeline Flow Completed Successfully!")
    return fusion_result

if __name__ == "__main__":
    main()
