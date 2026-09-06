import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage2_dl.data.preprocessing import ImagePreprocessor
from stage2_dl.radiology.cnn_model import RadiologyCNN
from stage2_dl.radiology.radiology_dataset import RadiologyDatasetLoader

class RadiologyGradCAM:
    """
    Grad-CAM Activation Explainer for Radiology CNN feature visual attribution.
    Extracts gradient activations from the final convolutional block.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook into final conv layer (conv3)
        target_layer = self.model.conv3[0]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=0):
        self.model.eval()
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        return cam, output

def generate_radiology_gradcam_demo():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifacts_dir = os.path.join(base_dir, 'stage2_dl', 'artifacts')
    model_path = os.path.join(artifacts_dir, 'models', 'best_radiology_model.pth')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadiologyCNN(num_classes=2).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    loader = RadiologyDatasetLoader()
    te_paths, te_labels = loader.get_image_paths_and_labels('test')
    
    if len(te_paths) == 0:
        return
        
    preprocessor = ImagePreprocessor(target_size=(128, 128))
    img_np = preprocessor.preprocess(te_paths[0])
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).to(device)
    img_tensor.requires_grad = True
    
    gradcam = RadiologyGradCAM(model)
    cam, _ = gradcam.generate_heatmap(img_tensor, target_class=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img_np[0], cmap='gray')
    ax1.set_title("CT Slice Input")
    ax1.axis('off')
    
    ax2.imshow(img_np[0], cmap='gray')
    ax2.imshow(cam, cmap='jet', alpha=0.5)
    ax2.set_title("Grad-CAM Activation Map")
    ax2.axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(artifacts_dir, 'figures', 'radiology_gradcam.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    
    print(f"[SUCCESS] Radiology Grad-CAM figure saved to: {fig_path}", flush=True)

if __name__ == "__main__":
    generate_radiology_gradcam_demo()
