import os
import glob
from PIL import Image
import numpy as np

class OncologyImageLoader:
    def __init__(self, data_dir):
        """
        data_dir: Path to data/stage2_dl/images
        """
        self.data_dir = data_dir
        self.classes = ['malignant', 'normal_benign']
        
    def load_dataset(self, split):
        """
        Loads the dataset for a specific split ('train', 'val', 'test').
        Returns a list of dictionaries with image paths and labels.
        """
        split_dir = os.path.join(self.data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory not found: {split_dir}")
            
        dataset = []
        corrupted_count = 0
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            img_paths = glob.glob(os.path.join(class_dir, '*.png'))
            for img_path in img_paths:
                # Validate image
                try:
                    with Image.open(img_path) as img:
                        img.verify() # check for corruption
                    dataset.append({
                        'path': img_path,
                        'label': class_idx,
                        'class_name': class_name
                    })
                except Exception:
                    corrupted_count += 1
                    
        return dataset, corrupted_count

    def get_split_stats(self):
        stats = {}
        for split in ['train', 'val', 'test']:
            dataset, corrupted = self.load_dataset(split)
            class_counts = {c: 0 for c in self.classes}
            for item in dataset:
                class_counts[item['class_name']] += 1
                
            stats[split] = {
                'total_images': len(dataset),
                'corrupted_images': corrupted,
                'class_distribution': class_counts
            }
        return stats
