import os
import pytest
import torch
import torch.nn as nn
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(base_dir))

from src.models.fusion import BaselineCNN, MultimodalFusionModel
from src.models.transformer import TransformerProgressionModel

@pytest.fixture
def models():
    cnn = BaselineCNN(num_classes=6)
    tx = TransformerProgressionModel(input_size=30, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=2)
    fusion = MultimodalFusionModel(cnn, tx, cnn_embed_dim=128, temporal_embed_dim=64, num_classes=2)
    return cnn, tx, fusion

def test_frozen_parameters(models):
    cnn, tx, fusion = models
    # CNN params should be frozen
    assert all(not p.requires_grad for p in fusion.cnn.parameters())
    # Transformer params should be frozen
    assert all(not p.requires_grad for p in fusion.transformer.parameters())
    # Fusion head should be trainable
    assert any(p.requires_grad for p in fusion.fusion_head.parameters())

def test_cnn_embedding_shape(models):
    cnn, tx, fusion = models
    images = torch.randn(5, 3, 224, 224) # 5 tiles
    embed = fusion.extract_image_features(images)
    assert embed.shape == (128,), f"Expected (128,), got {embed.shape}"

def test_transformer_embedding_shape(models):
    cnn, tx, fusion = models
    temp_x = torch.randn(2, 10, 30) # batch_size=2
    lengths = torch.LongTensor([10, 5])
    embed = fusion.extract_temporal_features(temp_x, lengths)
    assert embed.shape == (2, 64), f"Expected (2, 64), got {embed.shape}"

def test_fusion_output_shape_and_nan(models):
    cnn, tx, fusion = models
    images_list = [torch.randn(3, 3, 224, 224), torch.randn(1, 3, 224, 224)] # 2 patients
    temporal_x = torch.randn(2, 10, 30)
    temporal_lengths = torch.LongTensor([10, 5])
    
    logits = fusion(images_list, temporal_x, temporal_lengths)
    
    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

def test_fusion_backward_pass(models):
    cnn, tx, fusion = models
    images_list = [torch.randn(3, 3, 224, 224)]
    temporal_x = torch.randn(1, 10, 30)
    temporal_lengths = torch.LongTensor([10])
    targets = torch.LongTensor([1])
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fusion.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    logits = fusion(images_list, temporal_x, temporal_lengths)
    loss = criterion(logits, targets)
    loss.backward()
    
    # Check that fusion head has gradients
    has_grad = any(p.grad is not None for p in fusion.fusion_head.parameters())
    assert has_grad

def test_no_patient_leakage():
    import pandas as pd
    data_dir = base_dir / "sample_data"
    splits_df = pd.read_csv(data_dir / "train_validation_test_split.csv")
    train_pats = set(splits_df[splits_df['split'] == 'train']['patient_id'])
    val_pats = set(splits_df[splits_df['split'] == 'validation']['patient_id'])
    test_pats = set(splits_df[splits_df['split'] == 'test']['patient_id'])
    
    assert len(train_pats.intersection(val_pats)) == 0
    assert len(train_pats.intersection(test_pats)) == 0
    assert len(val_pats.intersection(test_pats)) == 0
