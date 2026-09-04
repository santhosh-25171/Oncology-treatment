import os
import sys
import torch
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 1. Imports
from stage2_dl.vision.cnn_model import LightweightCNN
from stage2_dl.vision.augmentation import get_train_transforms, get_val_test_transforms
from stage2_dl.vision.predict import CNNPredictor

def test_cnn_initialization():
    # 2. Initialization
    model = LightweightCNN(num_classes=2)
    assert isinstance(model, torch.nn.Module)

def test_cnn_forward_pass_shapes():
    # 3, 4, 5. Forward pass and IO shapes
    model = LightweightCNN(num_classes=2)
    dummy_input = torch.randn(2, 1, 128, 128)
    output = model(dummy_input)
    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"

def test_loss_calculation_and_training_batch():
    # 6, 7. Loss calculation and Training Batch
    model = LightweightCNN(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dummy_input = torch.randn(2, 1, 128, 128)
    dummy_labels = torch.tensor([0, 1], dtype=torch.long)
    
    # Record initial weights
    initial_weight = model.fc.weight.clone().detach()
    
    model.train()
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    assert not torch.isnan(loss), "Loss is NaN"
    
    loss.backward()
    optimizer.step()
    
    # Ensure weights were updated
    assert not torch.equal(model.fc.weight, initial_weight), "Weights were not updated"

def test_checkpoint_save_and_load(tmp_path):
    # 8, 9. Checkpoint Save/Load
    model = LightweightCNN(num_classes=2)
    chkpt_file = tmp_path / "test_checkpoint.pth"
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_mapping': ['malignant', 'normal_benign']
    }, chkpt_file)
    
    # Load
    loaded_checkpoint = torch.load(chkpt_file)
    new_model = LightweightCNN(num_classes=2)
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    
    # Verify state matches
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)

def test_prediction_output_shapes(monkeypatch):
    # 10, 11. Prediction probability shape & correct mapping
    class DummyPredictor(CNNPredictor):
        def __init__(self):
            self.device = torch.device("cpu")
            self.model = LightweightCNN(num_classes=2)
            self.model.eval()
            self.classes = ['malignant', 'normal_benign']
            # Mock preprocessor
            class MockPrep:
                def preprocess(self, path):
                    return np.random.rand(1, 128, 128).astype(np.float32)
            self.preprocessor = MockPrep()

    predictor = DummyPredictor()
    result = predictor.predict_image("dummy_path")
    
    assert "predicted_class" in result
    assert result["predicted_class"] in predictor.classes
    assert "probabilities" in result
    assert "confidence" in result
    assert len(result["probabilities"]) == 2

def test_augmentations():
    # 12, 13. Training has aug, Validation/test is deterministic
    train_aug = get_train_transforms()
    val_aug = get_val_test_transforms()
    
    assert train_aug is not None, "Training augmentation is missing"
    assert val_aug is None, "Validation/test augmentation must be deterministic (None)"
