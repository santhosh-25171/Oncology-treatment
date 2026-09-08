import os
import pytest
import torch
import torch.nn as nn
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(base_dir))

from src.models.lstm import LSTMProgressionModel

def test_lstm_instantiation():
    model = LSTMProgressionModel(input_size=30, hidden_size=64, num_layers=2, num_classes=2)
    assert isinstance(model, nn.Module)
    assert model.lstm.input_size == 30
    assert model.fc.out_features == 2

def test_lstm_forward_pass_variable_lengths():
    model = LSTMProgressionModel(input_size=30, hidden_size=64, num_layers=2, num_classes=2)
    
    # Batch size 3, max seq_len 10, features 30
    x = torch.randn(3, 10, 30)
    lengths = torch.LongTensor([10, 7, 4])
    
    # Mock packing requires no NaN/Inf
    logits = model(x, lengths)
    
    assert logits.shape == (3, 2), "Output should be [batch_size, num_classes]"
    assert not torch.isnan(logits).any(), "NaN found in outputs"
    assert not torch.isinf(logits).any(), "Inf found in outputs"

def test_lstm_backward_pass():
    model = LSTMProgressionModel(input_size=30, hidden_size=64, num_layers=1, num_classes=2)
    x = torch.randn(2, 5, 30)
    lengths = torch.LongTensor([5, 3])
    targets = torch.LongTensor([1, 0])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    logits = model(x, lengths)
    loss = criterion(logits, targets)
    loss.backward()
    
    # Check if gradients are computed for LSTM weights
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients were computed during backward pass."
