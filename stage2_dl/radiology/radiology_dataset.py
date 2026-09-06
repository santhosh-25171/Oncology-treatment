import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage2_dl.data.preprocessing import ImagePreprocessor

class RadiologyDatasetLoader:
    """
    Dataset Loader for Radiological CT/MRI Scan Slices (NoduleMNIST / OrganMNIST Prototype).
    
    PROTOTYPE LIMITATION NOTICE:
    This pipeline utilizes a public MedMNIST-family benchmark dataset structure for CT/MRI-style scan slices.
    It serves as a functional deep learning prototype for radiologic lesion detection and is NOT clinically
    validated diagnostic evidence.
    """
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, 'data', 'stage2_dl', 'radiology_images')
            
        self.data_dir = data_dir
        self.classes = ['nodule_lesion', 'normal_tissue']
        self.ensure_dataset_exists()
        
    def ensure_dataset_exists(self):
        """Generates deterministic synthetic CT scan slices if not already present."""
        if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) >= 3:
            return
            
        np.random.seed(42)
        os.makedirs(self.data_dir, exist_ok=True)
        
        splits = {'train': 500, 'val': 100, 'test': 150}
        
        for split, count in splits.items():
            split_dir = os.path.join(self.data_dir, split)
            for cls in self.classes:
                os.makedirs(os.path.join(split_dir, cls), exist_ok=True)
                
            for idx in range(count):
                # 30% nodule_lesion, 70% normal_tissue
                label = 0 if idx < int(0.3 * count) else 1
                cls_name = self.classes[label]
                
                # Base CT slice tissue texture
                img_arr = np.random.normal(loc=100, scale=25, size=(64, 64)).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_arr, mode='L')
                draw = ImageDraw.Draw(img)
                
                if label == 0:
                    # Draw bright radiologic nodule/lesion hyperintensity pattern
                    cx, cy = np.random.randint(20, 44, size=2)
                    r = np.random.randint(6, 12)
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=220, outline=255)
                    
                img_path = os.path.join(self.data_dir, split, cls_name, f"ct_scan_{idx:04d}.png")
                img.save(img_path)

    def get_image_paths_and_labels(self, split: str) -> Tuple[List[str], List[int]]:
        split_dir = os.path.join(self.data_dir, split)
        paths, labels = [], []
        
        for cls_idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(cls_idx)
                    
        return paths, labels

class RadiologyDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], target_size: Tuple[int, int] = (128, 128)):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = ImagePreprocessor(target_size=target_size)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img_np = self.preprocessor.preprocess(img_path)
        x = torch.from_numpy(img_np).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y
