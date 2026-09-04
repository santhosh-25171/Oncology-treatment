import torch
import torch.nn as nn

class LongitudinalLSTM(nn.Module):
    """
    Modest LSTM architecture for longitudinal sequence classification.
    Input: (batch_size, seq_length, num_features)
    Output: (batch_size, num_classes)
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=1, num_classes=2, dropout_rate=0.5):
        super(LongitudinalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass with optional sequence length masking.
        Args:
            x (torch.Tensor): Padded input sequences of shape (B, T, F)
            lengths (torch.Tensor, optional): True sequence lengths of shape (B,)
        """
        if lengths is not None and len(lengths) > 0:
            # Enforce CPU lengths for pack_padded_sequence and int64
            lengths_cpu = lengths.cpu().to(torch.int64)
            # Pack the padded sequence to avoid interpreting padded values
            packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_input)
            
            # The last valid hidden state for each sequence in the batch
            # hidden is (num_layers, B, hidden_size). Take the last layer.
            final_hidden = hidden[-1]
        else:
            # If lengths are not provided, process normally and take the last time step
            lstm_out, (hidden, cell) = self.lstm(x)
            final_hidden = hidden[-1]
            
        out = self.dropout(final_hidden)
        out = self.fc(out)
        return out
