import os
import sys
import torch
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from stage2_dl.sequence.lstm_model import LongitudinalLSTM
from stage2_dl.sequence.train import SequenceDataset
from stage2_dl.data.preprocessing import SequencePreprocessor

def test_lstm_initialization():
    model = LongitudinalLSTM(input_size=3, hidden_size=64, num_layers=1, num_classes=2)
    assert isinstance(model, torch.nn.Module)

def test_lstm_shapes():
    model = LongitudinalLSTM(input_size=3, hidden_size=64, num_layers=1, num_classes=2)
    dummy_input = torch.randn(2, 8, 3) # Batch 2, SeqLen 8, Features 3
    lengths = torch.tensor([8, 5], dtype=torch.long)
    output = model(dummy_input, lengths=lengths)
    
    assert output.shape == (2, 2)

def test_lstm_forward_pass_no_lengths():
    model = LongitudinalLSTM(input_size=3, hidden_size=64, num_layers=1, num_classes=2)
    dummy_input = torch.randn(2, 8, 3) 
    output = model(dummy_input) # length optional
    assert output.shape == (2, 2)

def test_checkpoint_save_and_load(tmp_path):
    model = LongitudinalLSTM(input_size=3, hidden_size=64, num_layers=1, num_classes=2)
    chkpt_file = tmp_path / "test_lstm_checkpoint.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'input_size': 3, 'hidden_size': 64, 'num_layers': 1, 'dropout_rate': 0.5}
    }, chkpt_file)
    
    loaded_checkpoint = torch.load(chkpt_file, weights_only=False)
    new_model = LongitudinalLSTM(**loaded_checkpoint['model_config'])
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)

def test_sequence_preprocessing_compatibility():
    preprocessor = SequencePreprocessor()
    
    # Mock some data (2 samples, 3 timesteps, 3 features)
    X_train = np.random.rand(2, 3, 3)
    # Simulate missing data (padding)
    X_train[1, 2, :] = np.nan 
    
    preprocessor.fit(X_train)
    assert preprocessor.feature_means is not None
    assert preprocessor.feature_stds is not None
    
    X_scaled = preprocessor.transform(X_train)
    
    # Ensure missing data (NaN) became 0.0 after transform
    assert np.all(X_scaled[1, 2, :] == 0.0)

def test_dataset_wrapper_extracts_lengths():
    preprocessor = SequencePreprocessor()
    X_train = np.random.rand(2, 4, 3)
    X_train[1, 3, :] = np.nan 
    X_train[1, 2, :] = np.nan 
    
    y = np.array([0, 1])
    dataset = SequenceDataset(X_train, y, preprocessor, is_train=True)
    
    # Batch item 0 has length 4
    assert dataset[0][2].item() == 4
    # Batch item 1 has length 2
    assert dataset[1][2].item() == 2
