import os
import json
import numpy as np
from image_loader import OncologyImageLoader
from sequence_loader import OncologySequenceLoader
from preprocessing import ImagePreprocessor, SequencePreprocessor

def run_validation():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_dir = os.path.join(base_dir, 'data', 'stage2_dl', 'images')
    seq_path = os.path.join(base_dir, 'data', 'stage2_dl', 'sequences', 'raw', 'synthetic_longitudinal_oncology.csv')

    print("\n" + "="*50)
    print("STAGE 2: DATA ENGINEERING VALIDATION")
    print("="*50)

    # 1. IMAGE VALIDATION
    print("\n--- IMAGE DATASET VALIDATION ---")
    img_loader = OncologyImageLoader(img_dir)
    img_stats = img_loader.get_split_stats()

    total_images = sum([s['total_images'] for s in img_stats.values()])
    corrupted_images = sum([s['corrupted_images'] for s in img_stats.values()])
    
    # Calculate global class distribution
    global_class_counts = {'malignant': 0, 'normal_benign': 0}
    for stat in img_stats.values():
        global_class_counts['malignant'] += stat['class_distribution']['malignant']
        global_class_counts['normal_benign'] += stat['class_distribution']['normal_benign']

    # Sample preprocess check (28x28 -> 128x128)
    sample_img_path = None
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith('.png'):
                sample_img_path = os.path.join(root, file)
                break
        if sample_img_path:
            break
            
    preprocessor = ImagePreprocessor(target_size=(128, 128))
    sample_tensor = preprocessor.preprocess(sample_img_path)
    cnn_shape = list(sample_tensor.shape)

    print(f"Total images: {total_images}")
    for split, stat in img_stats.items():
        print(f"[{split.upper()}] Images: {stat['total_images']} | Classes: {stat['class_distribution']}")
    print(f"Corrupted images: {corrupted_images}")
    print(f"Original Resolution: 28x28 (Inferred from source MedMNIST)")
    print(f"Preprocessed CNN Input Shape: {cnn_shape} (Min: {sample_tensor.min():.2f}, Max: {sample_tensor.max():.2f})")

    # 2. SEQUENCE VALIDATION
    print("\n--- SEQUENCE DATASET VALIDATION ---")
    seq_loader = OncologySequenceLoader(seq_path)
    (X_tr, y_tr, p_tr), (X_va, y_va, p_va), (X_te, y_te, p_te) = seq_loader.load_and_split()
    
    # Test Preprocessing Fitting Leakage Prevention
    seq_prep = SequencePreprocessor()
    seq_prep.fit(X_tr) # Fitted strictly on Train
    X_tr_scaled = seq_prep.transform(X_tr)
    X_va_scaled = seq_prep.transform(X_va)
    X_te_scaled = seq_prep.transform(X_te)
    
    leakage_free = seq_loader.validate_leakage(p_tr, p_va, p_te)
    
    train_dist = { "Responder": int(np.sum(y_tr==1)), "Non-Responder": int(np.sum(y_tr==0)) }
    val_dist = { "Responder": int(np.sum(y_va==1)), "Non-Responder": int(np.sum(y_va==0)) }
    test_dist = { "Responder": int(np.sum(y_te==1)), "Non-Responder": int(np.sum(y_te==0)) }
    
    total_seq_patients = len(p_tr) + len(p_va) + len(p_te)

    # Missingness calculation (prior to prep)
    total_cells = X_tr.size + X_va.size + X_te.size
    total_nans = np.isnan(X_tr).sum() + np.isnan(X_va).sum() + np.isnan(X_te).sum()
    missingness_ratio = float(total_nans) / total_cells

    print(f"Total Unique Patients: {total_seq_patients}")
    print(f"[TRAIN] Patients: {len(p_tr)} | Shape: {X_tr.shape} | Targets: {train_dist}")
    print(f"[VAL]   Patients: {len(p_va)} | Shape: {X_va.shape} | Targets: {val_dist}")
    print(f"[TEST]  Patients: {len(p_te)} | Shape: {X_te.shape} | Targets: {test_dist}")
    print(f"Patient Leakage Free: {leakage_free}")
    print(f"Overall Sequence Missingness: {missingness_ratio:.2%}")

    # 3. REPORT GENERATION
    report = {
        "image_dataset": {
            "name": "BreastMNIST (MedMNIST v2)",
            "source": "https://zenodo.org/record/6496656/files/breastmnist.npz",
            "original_resolution": "28x28",
            "cnn_resolution": "128x128",
            "channels": 1,
            "classes": ["malignant", "normal_benign"],
            "split_counts": {
                "train": img_stats['train']['total_images'],
                "val": img_stats['val']['total_images'],
                "test": img_stats['test']['total_images']
            },
            "class_distribution": global_class_counts,
            "corrupted_images": corrupted_images
        },
        "sequence_dataset": {
            "synthetic": True,
            "patients": total_seq_patients,
            "features": ["ctdna_level", "biomarker_2", "tumor_volume"],
            "time_steps": 8,
            "target_distribution": {
                "train": train_dist,
                "val": val_dist,
                "test": test_dist
            },
            "missingness": {
                "total_nans": int(total_nans),
                "missingness_ratio": round(missingness_ratio, 4)
            },
            "patient_leakage": not leakage_free
        },
        "limitations": [
            "BreastMNIST 28x28 resolution loses detailed textural pathology information. Resizing to 128x128 via bicubic interpolation strictly enables structural CNN compatibility and does NOT recover missing medical detail.",
            "The longitudinal sequence dataset is entirely SYNTHETIC. Although improved with varied patient baselines, noise, and trajectory overlaps, it is NOT CLINICALLY VALIDATED."
        ]
    }

    report_path = os.path.join(base_dir, 'data', 'stage2_dl', 'dataset_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    print(f"\n[SUCCESS] Saved comprehensive dataset report to: {report_path}")

if __name__ == "__main__":
    run_validation()
