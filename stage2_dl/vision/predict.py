import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stage2_dl.data.preprocessing import ImagePreprocessor
from stage2_dl.vision.cnn_model import LightweightCNN

class CNNPredictor:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = ImagePreprocessor(target_size=(128, 128))
        self.model = LightweightCNN(num_classes=2).to(self.device)
        self.classes = ['malignant', 'normal_benign']
        
        if checkpoint_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            checkpoint_path = os.path.join(base_dir, 'stage2_dl', 'artifacts', 'models', 'best_cnn_model.pth')
            
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'class_mapping' in checkpoint:
                self.classes = checkpoint['class_mapping']
        self.model.eval()

    def predict_image(self, img_path):
        """
        Predicts the class for a single image.
        Returns a dictionary with predicted class, probabilities, and confidence.
        """
        img_np = self.preprocessor.preprocess(img_path)
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        return {
            "predicted_class": self.classes[pred_idx],
            "probabilities": {
                self.classes[0]: float(probs[0]),
                self.classes[1]: float(probs[1])
            },
            "confidence": float(confidence)
        }
