import os
import numpy as np
from PIL import Image
import pandas as pd

def setup_image_dataset(base_dir: str) -> None:
    print("Setting up BreastMNIST Image Dataset...")
    npz_path = os.path.join(base_dir, 'breastmnist.npz')
    if not os.path.exists(npz_path):
        import urllib.request
        print("Downloading breastmnist.npz...")
        urllib.request.urlretrieve('https://zenodo.org/record/6496656/files/breastmnist.npz', npz_path)
    
    data = np.load(npz_path)
    images_dir = os.path.join(base_dir, 'data', 'stage2_dl', 'images')
    splits = ['train', 'val', 'test']
    classes = {0: 'malignant', 1: 'normal_benign'}
    
    for split in splits:
        split_dir = os.path.join(images_dir, split)
        for class_name in classes.values():
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
            
        images = data[f'{split}_images']
        labels = data[f'{split}_labels']
        
        for i in range(len(images)):
            img_array = images[i]
            label_idx = labels[i][0]
            class_name = classes[label_idx]
            
            # The images are saved as original 28x28 PNGs without artificial resizing
            img = Image.fromarray(img_array)
            img_path = os.path.join(split_dir, class_name, f'img_{i}.png')
            if not os.path.exists(img_path):
                img.save(img_path)
            
    print("BreastMNIST 28x28 source images saved to data/stage2_dl/images/")

def setup_sequence_dataset(base_dir: str) -> None:
    """
    Generates a realistic SYNTHETIC temporal dataset.
    This data is NOT clinically validated and is purely for modeling prototype purposes.
    """
    print("Setting up Synthetic Longitudinal Sequence Dataset...")
    # 1. SYNTHETIC LABEL GENERATION
    # Use a fixed random seed for reproducibility.
    np.random.seed(42)
    
    n_patients = 1000
    records = []
    
    for patient_id in range(n_patients):
        # We determine outcome upfront (approx 40% response rate)
        # The label outcome determines the general distribution of trajectory rates, 
        # but there is massive overlap preventing trivial predictability.
        is_responder = bool(np.random.rand() < 0.4)
        outcome = 'Responder' if is_responder else 'Non-Responder'
        
        # Patient-level baseline characteristics (high variance)
        base_tumor = np.random.lognormal(mean=2.0, sigma=0.5) 
        base_ctdna = np.random.lognormal(mean=4.0, sigma=1.0)
        base_marker2 = np.random.uniform(0.5, 10.0)
        
        treatment = np.random.choice(['Type_A', 'Type_B'])
        
        # Determine the underlying trajectory rate per patient
        # Introduce overlap between responders and non-responders
        if is_responder:
            # Responders tend to decrease, but can grow slightly
            tumor_rate = np.random.normal(loc=-0.05, scale=0.03) 
            ctdna_rate = np.random.normal(loc=-0.10, scale=0.05)
        else:
            # Non-responders tend to grow, but can decrease slightly
            tumor_rate = np.random.normal(loc=0.02, scale=0.03)
            ctdna_rate = np.random.normal(loc=0.05, scale=0.05)
            
        n_obs = np.random.randint(4, 9)
        
        # Simulate a random walk for non-monotonic fluctuations
        random_walk_tumor = 0
        random_walk_ctdna = 0
        
        for t in range(n_obs):
            # Time is generally weekly, but with some noise (days)
            day = max(0, int(t * 7 + np.random.normal(0, 1.5)))
            
            # Non-monotonic variations
            random_walk_tumor += np.random.normal(0, 0.02)
            random_walk_ctdna += np.random.normal(0, 0.05)
            
            # Trajectory calculation: Base * exp(rate * time) * random_walk_noise + measurement_noise
            tumor_val = base_tumor * np.exp(tumor_rate * day) * (1 + random_walk_tumor) + np.random.normal(0, 0.1)
            ctdna_val = base_ctdna * np.exp(ctdna_rate * day) * (1 + random_walk_ctdna) + np.random.normal(0, 2.0)
            
            # Marker 2 is largely unrelated noise/baseline
            marker2_val = base_marker2 + np.random.normal(0, 1.0)
            
            # Constrain to plausible physical values
            tumor_val = max(0.1, tumor_val)
            ctdna_val = max(0.0, ctdna_val)
            marker2_val = max(0.0, marker2_val)
            
            # Missing values injection (~10% missingness)
            if np.random.rand() < 0.10:
                ctdna_val = np.nan
            if np.random.rand() < 0.10:
                marker2_val = np.nan
            if np.random.rand() < 0.05:
                tumor_val = np.nan
                
            records.append({
                'patient_id': f'P{patient_id:04d}',
                'day': day,
                'ctdna_level': ctdna_val,
                'biomarker_2': marker2_val,
                'tumor_volume': tumor_val,
                'treatment': treatment,
                'outcome': outcome
            })
            
    df = pd.DataFrame(records)
    seq_dir = os.path.join(base_dir, 'data', 'stage2_dl', 'sequences', 'raw')
    os.makedirs(seq_dir, exist_ok=True)
    df.to_csv(os.path.join(seq_dir, 'synthetic_longitudinal_oncology.csv'), index=False)
    print("Synthetic sequences saved to data/stage2_dl/sequences/raw/")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    setup_image_dataset(base_dir)
    setup_sequence_dataset(base_dir)
