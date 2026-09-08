import os
import yaml
import pandas as pd

def test_dataset_paths():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_root = config['dataset']['root']
    assert os.path.exists(dataset_root), "Dataset root path does not exist"
    
    assert os.path.exists(config['dataset']['spatial_metadata']), "Spatial metadata missing"
    assert os.path.exists(config['dataset']['spatial_images']), "Images directory missing"
    assert os.path.exists(config['dataset']['temporal_data']), "Temporal data missing"

def test_expected_classes():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    meta_path = config['dataset']['spatial_metadata']
    df = pd.read_csv(meta_path)
    
    expected_classes = {'normal', 'benign', 'malignant', 'tumor_margin', 'necrotic', 'inflammatory'}
    actual_classes = set(df['tissue_class'].unique())
    assert expected_classes.issubset(actual_classes), "Missing expected tissue classes"
