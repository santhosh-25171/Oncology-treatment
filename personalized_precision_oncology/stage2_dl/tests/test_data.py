import os
import yaml
import pandas as pd
from pathlib import Path

def _get_config():
    config_file = Path(__file__).resolve().parent.parent / 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def _resolve_path(path_str):
    p = Path(path_str)
    if p.exists():
        return p
    repo_root = Path(__file__).resolve().parent.parent.parent
    if (repo_root / path_str).exists():
        return repo_root / path_str
    stage2_dir = Path(__file__).resolve().parent.parent
    if (stage2_dir / path_str).exists():
        return stage2_dir / path_str
    return p

def test_dataset_paths():
    config = _get_config()
    dataset_root = _resolve_path(config['dataset']['root'])
    assert dataset_root.exists(), f"Dataset root path does not exist: {dataset_root}"
    
    spatial_meta = _resolve_path(config['dataset']['spatial_metadata'])
    assert spatial_meta.exists(), f"Spatial metadata missing: {spatial_meta}"
    
    spatial_images = _resolve_path(config['dataset']['spatial_images'])
    assert spatial_images.exists(), f"Images directory missing: {spatial_images}"
    
    temporal_data = _resolve_path(config['dataset']['temporal_data'])
    assert temporal_data.exists(), f"Temporal data missing: {temporal_data}"

def test_expected_classes():
    config = _get_config()
    meta_path = _resolve_path(config['dataset']['spatial_metadata'])
    df = pd.read_csv(meta_path)
    
    expected_classes = {'normal', 'benign', 'malignant', 'tumor_margin', 'necrotic', 'inflammatory'}
    actual_classes = set(df['tissue_class'].unique())
    assert expected_classes.issubset(actual_classes), "Missing expected tissue classes"
