import os
import sys
import torch
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stage2_dl.sequence.transformer_model import SequenceTransformer, PositionalEncoding
from stage2_dl.data.preprocessing import SequencePreprocessor

def test_transformer_initialization():
    model = SequenceTransformer(input_size=3, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, num_classes=2)
    assert isinstance(model, torch.nn.Module)

def test_transformer_shapes():
    model = SequenceTransformer(input_size=3, d_model=32, nhead=4, num_layers=2)
    dummy_input = torch.randn(2, 8, 3) 
    lengths = torch.tensor([8, 5], dtype=torch.long)
    output = model(dummy_input, lengths=lengths)
    
    assert output.shape == (2, 2)

def test_transformer_forward_pass_no_lengths():
    model = SequenceTransformer(input_size=3, d_model=32, nhead=4, num_layers=2)
    dummy_input = torch.randn(2, 8, 3) 
    output = model(dummy_input)
    assert output.shape == (2, 2)
    
def test_positional_encoding():
    pe_layer = PositionalEncoding(d_model=32, max_len=16)
    x = torch.zeros(1, 8, 32)
    out = pe_layer(x)
    # Ensure it's not all zeros anymore (PE was added)
    assert torch.sum(torch.abs(out)) > 0
    # Shape unchanged
    assert out.shape == (1, 8, 32)

def test_transformer_checkpoint_save_and_load(tmp_path):
    model = SequenceTransformer(input_size=3, d_model=32, nhead=4, num_layers=2)
    chkpt_file = tmp_path / "test_transformer_checkpoint.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'input_size': 3, 'd_model': 32, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1, 'num_classes': 2}
    }, chkpt_file)
    
    loaded_checkpoint = torch.load(chkpt_file, weights_only=False)
    new_model = SequenceTransformer(**loaded_checkpoint['model_config'])
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)

def test_transformer_preprocessing_leakage():
    preprocessor = SequencePreprocessor()
    X_train = np.random.rand(2, 3, 3)
    X_train[1, 2, :] = np.nan 
    
    preprocessor.fit(X_train)
    X_scaled = preprocessor.transform(X_train)
    assert np.all(X_scaled[1, 2, :] == 0.0)
