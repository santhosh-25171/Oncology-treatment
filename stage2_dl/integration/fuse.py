import os
import sys
import json
import torch
import numpy as np
from typing import Dict, Any, Union, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage2_dl.data.preprocessing import ImagePreprocessor, SequencePreprocessor
from stage2_dl.vision.cnn_model import LightweightCNN
from stage2_dl.sequence.forecast_model import TransformerForecaster

class OncologyMultimodalFusion:
    """
    Multimodal Fusion Module:
    Combines Vision Malignancy Probability (from trained LightweightCNN) and
    Sequence Trajectory Forecast Trend (from trained TransformerForecaster)
    into a single unified risk score.
    
    Fusion Strategy Choice Rationale:
    We use a weighted convex combination rule (w_v * S_vision + w_t * S_trend)
    calibrated to [0, 1]. A transparent rule-based fusion is selected over a black-box
    learned layer because clinical decision support requires clear auditability of how
    image malignancy scores and temporal biomarker trajectory trends contribute to the
    final patient risk flag.
    """
    def __init__(self, cnn_checkpoint_path: Optional[str] = None, forecast_checkpoint_path: Optional[str] = None,
                 vision_weight: float = 0.5, trend_weight: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_weight = vision_weight
        self.trend_weight = trend_weight
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if cnn_checkpoint_path is None:
            cnn_checkpoint_path = os.path.join(base_dir, 'stage2_dl', 'artifacts', 'models', 'best_cnn_model.pth')
            
        if forecast_checkpoint_path is None:
            forecast_checkpoint_path = os.path.join(base_dir, 'stage2_dl', 'artifacts', 'models', 'best_forecast_model.pth')
            
        # 1. Load Vision Model (Read-Only)
        self.image_preprocessor = ImagePreprocessor(target_size=(128, 128))
        self.cnn_model = LightweightCNN(num_classes=2).to(self.device)
        
        if os.path.exists(cnn_checkpoint_path):
            cnn_ckpt = torch.load(cnn_checkpoint_path, map_location=self.device, weights_only=False)
            self.cnn_model.load_state_dict(cnn_ckpt['model_state_dict'])
        self.cnn_model.eval()
        
        # 2. Load Sequence Forecast Model (Read-Only)
        self.seq_preprocessor = SequencePreprocessor()
        if os.path.exists(forecast_checkpoint_path):
            fc_ckpt = torch.load(forecast_checkpoint_path, map_location=self.device, weights_only=False)
            self.seq_preprocessor.feature_means = fc_ckpt['preprocessor_means']
            self.seq_preprocessor.feature_stds = fc_ckpt['preprocessor_stds']
            
            cfg = fc_ckpt.get('model_config', {'input_size': 3, 'd_model': 32, 'nhead': 4, 'num_layers': 2, 'output_dim': 1})
            self.forecast_model = TransformerForecaster(
                input_size=cfg['input_size'],
                d_model=cfg['d_model'],
                nhead=cfg['nhead'],
                num_layers=cfg['num_layers'],
                output_dim=cfg['output_dim']
            ).to(self.device)
            self.forecast_model.load_state_dict(fc_ckpt['model_state_dict'])
            self.forecast_model.eval()
        else:
            self.forecast_model = None

    def get_vision_score(self, img_input: Union[str, np.ndarray]) -> float:
        """
        Computes vision score (probability of malignancy) from image input.
        img_input: File path to image or preprocessed numpy array (1, 128, 128)
        """
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                # Fallback default score if image file path is not found
                return 0.5
            img_np = self.image_preprocessor.preprocess(img_input)
        else:
            img_np = img_input
            
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.cnn_model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            # Class 0: Malignant probability
            malignant_prob = float(probs[0].cpu().item())
            
        return malignant_prob

    def get_trend_score(self, seq_raw: np.ndarray) -> float:
        """
        Computes trend score from raw longitudinal sequence array (seq_len, 3).
        Calculates predicted % change in ctDNA level over next reading, mapped to [0, 1].
        """
        if self.forecast_model is None or self.seq_preprocessor.feature_means is None:
            return 0.5
            
        # Count non-NaN steps
        valid_mask = ~np.isnan(seq_raw).any(axis=1)
        valid_len = np.sum(valid_mask)
        
        if valid_len == 0:
            return 0.5
            
        # Extract latest non-NaN ctDNA reading
        latest_ctdna = float(seq_raw[valid_len - 1, 0])
        
        # Transform sequence using preprocessor
        seq_scaled = self.seq_preprocessor.transform(seq_raw.reshape(1, *seq_raw.shape))[0]
        
        seq_tensor = torch.from_numpy(seq_scaled).float().unsqueeze(0).to(self.device)
        len_tensor = torch.tensor([valid_len], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.forecast_model(seq_tensor, lengths=len_tensor)
            pred_scaled_val = float(pred_scaled.detach().cpu().numpy().reshape(-1)[0])
            
        # Convert predicted next-step ctDNA back to physical units
        mean_ctdna = self.seq_preprocessor.feature_means[0]
        std_ctdna = self.seq_preprocessor.feature_stds[0]
        predicted_next_ctdna = pred_scaled_val * std_ctdna + mean_ctdna
        
        # Calculate trajectory slope delta in standardized z-score space
        latest_ctdna_scaled = (latest_ctdna - mean_ctdna) / std_ctdna
        z_delta = pred_scaled_val - latest_ctdna_scaled
        
        # Map trajectory z_delta to a smooth calibrated trend score in [0, 1] via Sigmoid
        trend_score = 1.0 / (1.0 + np.exp(-2.0 * z_delta))
        return float(trend_score)

    def fuse(self, img_input: Union[str, np.ndarray], seq_raw: np.ndarray) -> Dict[str, Any]:
        """
        Executes multimodal fusion combining vision score and sequence trajectory trend score.
        Returns JSON-serializable output dictionary:
        { "vision_score": float, "trend_score": float, "combined_risk": float, "flag": "Low|Moderate|High" }
        """
        vision_score = round(self.get_vision_score(img_input), 4)
        trend_score = round(self.get_trend_score(seq_raw), 4)
        
        combined_risk = round(self.vision_weight * vision_score + self.trend_weight * trend_score, 4)
        
        if combined_risk >= 0.66:
            flag = "High"
        elif combined_risk >= 0.33:
            flag = "Moderate"
        else:
            flag = "Low"
            
        return {
            "vision_score": vision_score,
            "trend_score": trend_score,
            "combined_risk": combined_risk,
            "flag": flag
        }

def run_fusion_demo():
    fusion = OncologyMultimodalFusion()
    
    # Define 5 patient profiles spanning low, moderate, and high risk spectrums
    patient_profiles = [
        {
            "name": "Patient 1 (Low Risk - Remission)",
            "img": np.random.normal(loc=50, scale=10, size=(1, 128, 128)).clip(0, 255).astype(np.float32) / 255.0,
            "seq": np.array([
                [35.0, 50.0, 30.0],
                [28.0, 45.0, 25.0],
                [20.0, 40.0, 20.0],
                [12.0, 35.0, 15.0],
                [5.0,  30.0, 10.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan]
            ], dtype=np.float32)
        },
        {
            "name": "Patient 2 (Low-Moderate Risk - Gradual Response)",
            "img": np.random.normal(loc=90, scale=15, size=(1, 128, 128)).clip(0, 255).astype(np.float32) / 255.0,
            "seq": np.array([
                [20.0, 40.0, 18.0],
                [18.5, 39.0, 17.5],
                [17.0, 38.0, 17.0],
                [15.5, 37.0, 16.5],
                [14.0, 36.0, 16.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan]
            ], dtype=np.float32)
        },
        {
            "name": "Patient 3 (Moderate Risk - Stable Trajectory)",
            "img": np.random.normal(loc=128, scale=20, size=(1, 128, 128)).clip(0, 255).astype(np.float32) / 255.0,
            "seq": np.array([
                [15.0, 42.0, 20.0],
                [15.2, 42.1, 20.1],
                [15.1, 42.0, 20.0],
                [15.3, 42.2, 20.2],
                [15.2, 42.1, 20.1],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan]
            ], dtype=np.float32)
        },
        {
            "name": "Patient 4 (High-Moderate Risk - Early Progression)",
            "img": np.random.normal(loc=180, scale=25, size=(1, 128, 128)).clip(0, 255).astype(np.float32) / 255.0,
            "seq": np.array([
                [10.0, 35.0, 15.0],
                [12.0, 38.0, 17.0],
                [14.5, 42.0, 19.5],
                [17.0, 46.0, 22.0],
                [20.5, 51.0, 25.5],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan]
            ], dtype=np.float32)
        },
        {
            "name": "Patient 5 (High Risk - Aggressive Progression)",
            "img": np.random.normal(loc=220, scale=30, size=(1, 128, 128)).clip(0, 255).astype(np.float32) / 255.0,
            "seq": np.array([
                [15.0, 40.0, 20.0],
                [22.0, 52.0, 28.0],
                [32.0, 68.0, 39.0],
                [48.0, 88.0, 54.0],
                [70.0, 115.0, 75.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan]
            ], dtype=np.float32)
        }
    ]
    
    print("=" * 80, flush=True)
    print("CHECK 3: MULTIMODAL FUSION DIAGNOSTICS ACROSS 5 RISK SPECTRUM PATIENTS", flush=True)
    print("=" * 80, flush=True)
    
    results = []
    for p in patient_profiles:
        res = fusion.fuse(p["img"], p["seq"])
        res_entry = {
            "patient_profile": p["name"],
            "vision_score": res["vision_score"],
            "trend_score": res["trend_score"],
            "combined_risk": res["combined_risk"],
            "flag": res["flag"]
        }
        results.append(res_entry)
        print(f"[{res['flag']:8s}] {p['name']:48s} | Vision={res['vision_score']:.4f} | Trend={res['trend_score']:.4f} | Combined={res['combined_risk']:.4f}", flush=True)
        
    print("-" * 80, flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_fusion_demo()
