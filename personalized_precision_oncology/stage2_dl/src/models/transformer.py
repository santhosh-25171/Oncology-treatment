import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerProgressionModel(nn.Module):
    def __init__(self, input_size=30, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2, num_classes=2):
        super(TransformerProgressionModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x, lengths):
        # x shape: [batch_size, max_seq_len, input_size]
        # lengths shape: [batch_size]
        batch_size, max_seq_len, _ = x.shape
        device = x.device
        
        # Create src_key_padding_mask
        # PyTorch requires True for padded positions that should be ignored
        # Shape: [batch_size, max_seq_len]
        mask = torch.arange(max_seq_len, device=device).expand(batch_size, max_seq_len) >= lengths.unsqueeze(1).to(device)
        
        # Project and add positional encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # src_key_padding_mask expects True where values should be ignored
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Masked Mean Pooling
        # Invert mask: True for valid positions, False for padded positions
        valid_mask = ~mask
        valid_mask = valid_mask.unsqueeze(-1).float() # [batch_size, max_seq_len, 1]
        
        # Sum valid outputs
        sum_output = torch.sum(output * valid_mask, dim=1) # [batch_size, d_model]
        
        # Divide by sequence lengths to get the mean
        lengths_float = lengths.unsqueeze(1).float().to(device) # [batch_size, 1]
        
        # Prevent division by zero just in case
        lengths_float = torch.clamp(lengths_float, min=1.0)
        mean_output = sum_output / lengths_float # [batch_size, d_model]
        
        out = self.dropout(mean_output)
        logits = self.fc(out)
        return logits
