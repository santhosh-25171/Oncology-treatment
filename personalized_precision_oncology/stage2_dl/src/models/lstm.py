import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMProgressionModel(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super(LSTMProgressionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths):
        # x shape: [batch_size, seq_len, features]
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.lstm(packed_x)
        
        # hn shape: [num_layers, batch_size, hidden_size]
        last_hidden = hn[-1, :, :]
        
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits
