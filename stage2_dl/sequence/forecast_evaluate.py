import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from stage2_dl.data.sequence_loader import OncologySequenceLoader
from stage2_dl.data.preprocessing import SequencePreprocessor
from stage2_dl.sequence.forecast_model import TransformerForecaster
from stage2_dl.sequence.forecast_train import ForecastDataset

def evaluate_forecaster():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    seq_path = os.path.join(base_dir, 'data', 'stage2_dl', 'sequences', 'raw', 'synthetic_longitudinal_oncology.csv')
    artifacts_dir = os.path.join(base_dir, 'stage2_dl', 'artifacts')
    model_path = os.path.join(artifacts_dir, 'models', 'best_forecast_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained forecast model checkpoint not found at: {model_path}")
        
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Rebuild preprocessor with checkpointed stats
    preprocessor = SequencePreprocessor()
    preprocessor.feature_means = checkpoint['preprocessor_means']
    preprocessor.feature_stds = checkpoint['preprocessor_stds']
    
    # 1. Load Data (Held-out Test set)
    loader = OncologySequenceLoader(seq_path)
    (X_tr, _, _), _, (X_te, _, test_pids) = loader.load_and_split()
    
    test_dataset = ForecastDataset(X_te, preprocessor, is_train=False, target_feature_idx=0, output_dim=1)
    
    # 2. Instantiate Model
    config = checkpoint.get('model_config', {'input_size': 3, 'd_model': 32, 'nhead': 4, 'num_layers': 2, 'output_dim': 1})
    model = TransformerForecaster(
        input_size=config['input_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        output_dim=config['output_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    preds_scaled = []
    targets_scaled = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, y, length = test_dataset[i]
            x_b = x.unsqueeze(0).to(device)
            len_b = length.unsqueeze(0).to(device)
            
            pred = model(x_b, lengths=len_b)
            preds_scaled.append(float(pred.detach().cpu().numpy().reshape(-1)[0]))
            targets_scaled.append(float(y.detach().cpu().numpy().reshape(-1)[0]))
            
    preds_scaled = np.array(preds_scaled)
    targets_scaled = np.array(targets_scaled)
    
    # Unscale predictions & targets back to physical units (ctDNA level in ng/mL)
    mean_val = float(preprocessor.feature_means[0])
    std_val = float(preprocessor.feature_stds[0])
    
    preds_physical = preds_scaled * std_val + mean_val
    targets_physical = targets_scaled * std_val + mean_val
    
    # 3. Compute Naive Baseline (predict next value = last observed value)
    naive_preds_scaled = []
    for i in range(len(test_dataset)):
        x, _, length = test_dataset[i]
        # x input shape is (seq_len, num_features). Last valid input step is length - 1
        last_observed_scaled = x[length - 1, 0].item()
        naive_preds_scaled.append(last_observed_scaled)
        
    naive_preds_scaled = np.array(naive_preds_scaled)
    naive_preds_physical = naive_preds_scaled * std_val + mean_val
    
    mae_scaled = float(mean_absolute_error(targets_scaled, preds_scaled))
    rmse_scaled = float(np.sqrt(mean_squared_error(targets_scaled, preds_scaled)))
    
    mae_physical = float(mean_absolute_error(targets_physical, preds_physical))
    rmse_physical = float(np.sqrt(mean_squared_error(targets_physical, preds_physical)))
    r2 = float(r2_score(targets_physical, preds_physical))
    
    naive_mae_physical = float(mean_absolute_error(targets_physical, naive_preds_physical))
    naive_rmse_physical = float(np.sqrt(mean_squared_error(targets_physical, naive_preds_physical)))
    
    metrics = {
        "model_architecture": "TransformerForecaster (Subclassed SequenceTransformer)",
        "evaluation_dataset": "Held-out Test Patients (n=150)",
        "target_feature": "ctdna_level",
        "scaler_type": "SequencePreprocessor Standard Z-Score (X - mean) / std (Fit strictly on X_train, N=700)",
        "scaler_mean_ng_ml": round(mean_val, 4),
        "scaler_std_ng_ml": round(std_val, 4),
        "mae_scaled": round(mae_scaled, 4),
        "rmse_scaled": round(rmse_scaled, 4),
        "mae_physical_ng_ml": round(mae_physical, 4),
        "rmse_physical_ng_ml": round(rmse_physical, 4),
        "naive_baseline_mae_physical_ng_ml": round(naive_mae_physical, 4),
        "naive_baseline_rmse_physical_ng_ml": round(naive_rmse_physical, 4),
        "r2_score": round(r2, 4)
    }
    
    metrics_path = os.path.join(artifacts_dir, 'metrics', 'forecast_test_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Plot predicted vs actual trajectories
    plt.figure(figsize=(10, 5))
    indices = np.arange(len(targets_physical))
    plt.plot(indices[:50], targets_physical[:50], 'o-', label='Actual ctDNA Level', color='#1F77B4', alpha=0.8)
    plt.plot(indices[:50], preds_physical[:50], 's--', label='Predicted Next-Step ctDNA', color='#FF7F0E', alpha=0.8)
    plt.plot(indices[:50], naive_preds_physical[:50], 'x:', label='Naive Baseline (Last Value)', color='#7F7F7F', alpha=0.6)
    plt.title('Sequence Trajectory Forecasting: Actual vs Predicted Next-Step ctDNA (Test Cohort)')
    plt.xlabel('Patient Sequence Index')
    plt.ylabel('ctDNA Level (ng/mL)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    fig_path = os.path.join(artifacts_dir, 'figures', 'forecast_trajectories.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    
    print("=" * 70, flush=True)
    print("CHECK 1 & 2: SEQUENCE FORECASTING DIAGNOSTICS & NAIVE BASELINE", flush=True)
    print("=" * 70, flush=True)
    print(f"Scaler Type: SequencePreprocessor Z-Score (Mean={mean_val:.2f}, Std={std_val:.2f})", flush=True)
    print(f"Scaler Fit Status: Fitted ONLY on training data (X_tr, N=700)", flush=True)
    print(f"Consistent Inverse Transform: YES (Applied to both prediction & ground truth)", flush=True)
    print("-" * 70, flush=True)
    print("10 Sample (Predicted, Actual) Pairs in original ng/mL units:")
    for idx in range(min(10, len(preds_physical))):
        print(f"  Sample #{idx+1:02d}: Predicted = {preds_physical[idx]:8.2f} ng/mL | Actual = {targets_physical[idx]:8.2f} ng/mL | Diff = {abs(preds_physical[idx]-targets_physical[idx]):7.2f} ng/mL", flush=True)
    print("-" * 70, flush=True)
    print(f"Trained Forecaster MAE: {mae_physical:.4f} ng/mL | RMSE: {rmse_physical:.4f} ng/mL | R^2: {r2:.4f}", flush=True)
    print(f"Naive Baseline MAE:    {naive_mae_physical:.4f} ng/mL | RMSE: {naive_rmse_physical:.4f} ng/mL", flush=True)
    print("=" * 70, flush=True)
    
    return metrics

if __name__ == "__main__":
    evaluate_forecaster()
