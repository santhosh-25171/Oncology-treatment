import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stage2_dl.vision.cnn_model import LightweightCNN
from stage2_dl.data.preprocessing import ImagePreprocessor
from stage2_dl.data.image_loader import OncologyImageLoader

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        target = output[0][target_class]
        target.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        
        return heatmap

def run_explainability():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_dir = os.path.join(base_dir, 'data', 'stage2_dl', 'images')
    artifacts_dir = os.path.join(base_dir, 'stage2_dl', 'artifacts')
    os.makedirs(os.path.join(artifacts_dir, 'figures'), exist_ok=True)
    
    checkpoint_path = os.path.join(artifacts_dir, 'models', 'best_cnn_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Skipping explainability: Checkpoint not found at {checkpoint_path}")
        return
        
    model = LightweightCNN(num_classes=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # We will target the last convolutional block's ReLU/MaxPool for GradCAM
    target_layer = model.block3[-1]
    grad_cam = GradCAM(model, target_layer)
    
    loader = OncologyImageLoader(img_dir)
    test_data, _ = loader.load_dataset('test')
    preprocessor = ImagePreprocessor(target_size=(128, 128))
    
    if not test_data:
        print("No test data found.")
        return
        
    # Test on the first 3 images
    for idx in range(min(3, len(test_data))):
        img_path = test_data[idx]['path']
        label = test_data[idx]['class_name']
        
        img_np = preprocessor.preprocess(img_path)
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).to(device)
        
        heatmap = grad_cam.generate(img_tensor)
        
        # Plotting
        original_img = Image.open(img_path).convert('L').resize((128, 128))
        heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((128, 128), Image.BILINEAR))
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title(f'Original\nTrue: {label}')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(original_img, cmap='gray')
        axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        save_path = os.path.join(artifacts_dir, 'figures', f'gradcam_test_img_{idx}.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Grad-CAM visualization to {save_path}")

if __name__ == "__main__":
    run_explainability()
