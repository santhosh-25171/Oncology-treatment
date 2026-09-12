import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []
        
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate normalized Grad-CAM heatmap (H, W) in range [0, 1].
        input_tensor: Tensor of shape (1, 3, H, W)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Ensure tensor has batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")
            
        # Global Average Pooling of gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True) # (1, C, 1, 1)
        
        # Weighted linear combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, h, w)
        cam = F.relu(cam)
        
        # Resize to input dimensions (H, W)
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, class_idx

    @staticmethod
    def overlay_heatmap(original_image_np, heatmap, alpha=0.5, colormap='jet'):
        """
        Overlay heatmap onto RGB image.
        original_image_np: (H, W, 3) numpy array uint8 [0, 255] or float [0, 1]
        heatmap: (H, W) float array [0, 1]
        """
        if original_image_np.dtype == np.uint8:
            img = original_image_np.astype(np.float32) / 255.0
        else:
            img = original_image_np.copy()
            
        cmap = plt.get_cmap(colormap)
        colored_cam = cmap(heatmap)[:, :, :3] # Keep RGB, drop alpha
        
        overlay = (1.0 - alpha) * img + alpha * colored_cam
        overlay = np.clip(overlay, 0.0, 1.0)
        return (overlay * 255).astype(np.uint8)
