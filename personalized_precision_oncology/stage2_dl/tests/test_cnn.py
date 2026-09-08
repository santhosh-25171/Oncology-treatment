import os
import pytest
import torch
from pathlib import Path
import pandas as pd
from PIL import Image

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "sample_data"

# Redefine model locally for testing structural integrity
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def test_cnn_forward_pass():
    model = BaselineCNN(num_classes=6)
    # Dummy batch of 4 images, 3 channels, 224x224
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (4, 6), f"Expected shape (4, 6) but got {output.shape}"

def test_dataset_leakage():
    splits = pd.read_csv(data_dir / "train_validation_test_split.csv")
    train_pats = set(splits[splits['split'] == 'train']['patient_id'])
    val_pats = set(splits[splits['split'] == 'validation']['patient_id'])
    test_pats = set(splits[splits['split'] == 'test']['patient_id'])
    
    assert len(train_pats.intersection(val_pats)) == 0
    assert len(train_pats.intersection(test_pats)) == 0
    assert len(val_pats.intersection(test_pats)) == 0

def test_number_of_classes():
    meta = pd.read_csv(data_dir / "spatial" / "spatial_metadata.csv")
    unique_classes = meta['tissue_class'].nunique()
    assert unique_classes == 6
