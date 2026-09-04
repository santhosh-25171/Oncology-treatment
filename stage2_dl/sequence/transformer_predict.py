import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stage2_dl.data.preprocessing import SequencePreprocessor
from stage2_dl.sequence.transformer_model import SequenceTransformer

class TransformerPredictor:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if checkpoint_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            checkpoint_path = os.path.join(base_dir, 'stage2_dl', 'artifacts', 'models', 'best_transformer_model.pth')
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint['model_config']
        
        self.model = SequenceTransformer(**config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.preprocessor = SequencePreprocessor()
        if 'preprocessor_means' in checkpoint:
            self.preprocessor.feature_means = checkpoint['preprocessor_means']
            self.preprocessor.feature_stds = checkpoint['preprocessor_stds']
            
        self.classes = {0: 'Non-Responder', 1: 'Responder'}

    def predict_sequence(self, sequence_array):
        x_raw = np.expand_dims(sequence_array, axis=0)
        lengths = (~np.isnan(x_raw)).any(axis=2).sum(axis=1)
        x_scaled = self.preprocessor.transform(x_raw)
        
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.model(x_tensor, lengths=lengths_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
        pred_idx = torch.argmax(probs).item()
        
        return {
            "predicted_class": self.classes[pred_idx],
            "probabilities": {
                "Non-Responder": float(probs[0]),
                "Responder": float(probs[1])
            },
            "confidence": float(probs[pred_idx])
        }

if __name__ == "__main__":
    pass
