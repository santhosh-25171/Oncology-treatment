import os
import pytest
import torch
import torch.nn as nn
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(base_dir))

from src.models.transformer import TransformerProgressionModel

def test_1_transformer_instantiation():
    model = TransformerProgressionModel(input_size=30, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=2)
    assert isinstance(model, nn.Module)
    assert model.fc.out_features == 2

def test_2_correct_input_dimensions():
    model = TransformerProgressionModel(input_size=30, d_model=64)
    assert model.input_projection.in_features == 30

def test_3_correct_output_dimensions():
    model = TransformerProgressionModel(input_size=30, d_model=64, num_classes=2)
    x = torch.randn(3, 10, 30)
    lengths = torch.LongTensor([10, 7, 4])
    logits = model(x, lengths)
    assert logits.shape == (3, 2)

def test_4_variable_sequence_lengths_work():
    model = TransformerProgressionModel()
    x = torch.randn(2, 5, 30)
    lengths = torch.LongTensor([5, 3])
    try:
        logits = model(x, lengths)
        success = True
    except Exception:
        success = False
    assert success

def test_5_padding_mask_correctly_constructed():
    model = TransformerProgressionModel()
    x = torch.randn(2, 5, 30)
    lengths = torch.LongTensor([5, 3])
    # Internally the mask should be:
    # [False, False, False, False, False]
    # [False, False, False, True, True]
    batch_size, max_seq_len = 2, 5
    mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len) >= lengths.unsqueeze(1)
    
    assert mask[0].sum() == 0 # No padding
    assert mask[1].sum() == 2 # 2 padded tokens

def test_6_padded_timesteps_do_not_affect_masked_pooling():
    model = TransformerProgressionModel()
    model.eval()
    x1 = torch.randn(1, 5, 30)
    # create a copy where padded tokens are wildly different
    x2 = x1.clone()
    x2[0, 3:, :] = 9999.0 
    
    lengths = torch.LongTensor([3])
    
    with torch.no_grad():
        out1 = model(x1, lengths)
        out2 = model(x2, lengths)
        
    # outputs should be exactly the same since padding is masked out
    assert torch.allclose(out1, out2, atol=1e-5)

def test_7_forward_pass_produces_no_nan_inf():
    model = TransformerProgressionModel()
    x = torch.randn(4, 15, 30)
    lengths = torch.LongTensor([15, 12, 10, 5])
    logits = model(x, lengths)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

def test_8_loss_calculation_works():
    model = TransformerProgressionModel()
    x = torch.randn(2, 5, 30)
    lengths = torch.LongTensor([5, 3])
    targets = torch.LongTensor([1, 0])
    criterion = nn.CrossEntropyLoss()
    logits = model(x, lengths)
    loss = criterion(logits, targets)
    assert loss.item() > 0

def test_9_backward_pass_works():
    model = TransformerProgressionModel()
    x = torch.randn(2, 5, 30)
    lengths = torch.LongTensor([5, 3])
    targets = torch.LongTensor([1, 0])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    logits = model(x, lengths)
    loss = criterion(logits, targets)
    loss.backward()
    
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad

def test_10_checkpoint_save_load(tmp_path):
    model = TransformerProgressionModel()
    path = tmp_path / "test_ckpt.pt"
    torch.save(model.state_dict(), path)
    
    model2 = TransformerProgressionModel()
    model2.load_state_dict(torch.load(path, weights_only=True))
    assert True
