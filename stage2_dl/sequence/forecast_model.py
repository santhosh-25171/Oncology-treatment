import torch
import torch.nn as nn
from stage2_dl.sequence.lstm_model import LongitudinalLSTM
from stage2_dl.sequence.transformer_model import SequenceTransformer

class LSTMForecaster(LongitudinalLSTM):
    """
    Subclasses LongitudinalLSTM for longitudinal trajectory forecasting (regression).
    Replaces the final classification layer with a regression head:
    Linear(hidden_size -> output_dim).
    
    Default output_dim=1 for next-step trajectory forecasting (e.g. ctDNA level or % change),
    or output_dim=3 for 3-feature vector forecasting.
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=1, output_dim=1, dropout_rate=0.5):
        super(LSTMForecaster, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=output_dim,
            dropout_rate=dropout_rate
        )
        # Replace classification layer with regression head
        self.fc = nn.Linear(hidden_size, output_dim)

class TransformerForecaster(SequenceTransformer):
    """
    Subclasses SequenceTransformer for longitudinal trajectory forecasting (regression).
    Replaces the final classification layer with a regression head:
    Linear(d_model -> output_dim).
    """
    def __init__(self, input_size=3, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_dim=1):
        super(TransformerForecaster, self).__init__(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=output_dim
        )
        # Replace classification layer with regression head
        self.fc = nn.Linear(d_model, output_dim)
