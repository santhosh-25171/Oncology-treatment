import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from stage2_dl.data.image_loader import OncologyImageLoader
from stage2_dl.data.preprocessing import ImagePreprocessor

def run_image_eda():
    img_dir = os.path.join(base_dir, 'data', 'stage2_dl', 'images')
    fig_dir = os.path.join(base_dir, 'data', 'stage2_dl', 'eda', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    loader = OncologyImageLoader(img_dir)
    stats = loader.get_split_stats()
    
    total_imgs = sum(s['total_images'] for s in stats.values())
    total_corrupted = sum(s['corrupted_images'] for s in stats.values())
    
    global_class_counts = {'malignant': 0, 'normal_benign': 0}
    for s in stats.values():
        global_class_counts['malignant'] += s['class_distribution']['malignant']
        global_class_counts['normal_benign'] += s['class_distribution']['normal_benign']
    
    # Calculate percentages
    pct_malignant = global_class_counts['malignant'] / total_imgs * 100
    pct_benign = global_class_counts['normal_benign'] / total_imgs * 100
    imbalance_ratio = global_class_counts['normal_benign'] / global_class_counts['malignant']
    
    # Check duplicate arrays
    all_arrays = []
    
    all_pixels = []
    
    # Load all paths for visualization and pixel analysis
    train_data, _ = loader.load_dataset('train')
    
    for split in ['train', 'val', 'test']:
        s_data, _ = loader.load_dataset(split)
        for item in s_data:
            img_arr = np.array(Image.open(item['path']).convert('L'))
            all_arrays.append(img_arr.tobytes())
            
            if np.random.rand() < 0.1:
                all_pixels.extend(img_arr.flatten())
                
    all_pixels = np.array(all_pixels)
    
    duplicate_count = len(all_arrays) - len(set(all_arrays))
    
    # Representative Image Grid
    malignant_imgs = [d for d in train_data if d['class_name'] == 'malignant'][:5]
    benign_imgs = [d for d in train_data if d['class_name'] == 'normal_benign'][:5]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axes[0, i].imshow(Image.open(malignant_imgs[i]['path']).convert('L'), cmap='gray')
        axes[0, i].set_title("Malignant")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(Image.open(benign_imgs[i]['path']).convert('L'), cmap='gray')
        axes[1, i].set_title("Normal/Benign")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'image_grid.png'))
    plt.close()
    
    # Preprocessing comparison (28x28 vs 128x128)
    sample_path = benign_imgs[0]['path']
    orig_img = Image.open(sample_path).convert('L')
    
    preprocessor = ImagePreprocessor(target_size=(128, 128), normalize=False)
    processed_array = preprocessor.preprocess(sample_path) # shape: (1, 128, 128)
    processed_img = processed_array[0] # drop channel for display
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(np.array(orig_img), cmap='gray')
    ax1.set_title(f"Original RAW (28x28)")
    ax1.axis('off')
    
    ax2.imshow(processed_img, cmap='gray')
    ax2.set_title(f"Preprocessed CNN Input (128x128)\n*Interpolated*")
    ax2.axis('off')
    
    plt.suptitle("128x128 is interpolated from the original 28x28 image and does not add medical information.")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'preprocessing_comparison.png'))
    plt.close()
    
    # Pixel Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(all_pixels, bins=50, color='gray', alpha=0.7)
    plt.title("Pixel Intensity Distribution (Original 0-255 scale)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(fig_dir, 'pixel_distribution.png'))
    plt.close()
    
    return {
        "dataset_size": total_imgs,
        "split_counts": {k: v['total_images'] for k, v in stats.items()},
        "class_distribution": {
            "counts": global_class_counts,
            "percentages": {
                "malignant": float(pct_malignant),
                "normal_benign": float(pct_benign)
            },
            "imbalance_ratio": float(imbalance_ratio)
        },
        "properties": {
            "original_resolution": "28x28",
            "channels": 1,
            "pixel_min": float(all_pixels.min()),
            "pixel_max": float(all_pixels.max()),
            "pixel_mean": float(all_pixels.mean()),
            "pixel_std": float(all_pixels.std())
        },
        "quality": {
            "corrupted_images": total_corrupted,
            "missing_images": 0,
            "duplicate_images_detected": duplicate_count
        },
        "leakage_assessment": {
            "patient_ids_available": False,
            "note": "Patient-level leakage cannot be independently verified from available BreastMNIST metadata as patient IDs are not provided."
        }
    }

if __name__ == "__main__":
    res = run_image_eda()
    print(json.dumps(res, indent=2))
