import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding is required because the Transformer architecture
    lacks an inherent sense of sequence order. This deterministic sinusoidal
    encoding injects temporal position information into the embeddings.
    """
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """ x shape: (B, seq_len, d_model) """
        x = x + self.pe[:, :x.size(1), :]
        return x

class SequenceTransformer(nn.Module):
    def __init__(self, input_size=3, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=32)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x, lengths=None):
        B, seq_len, _ = x.size()
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        if lengths is not None and len(lengths) > 0:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
            lengths_expanded = lengths.unsqueeze(1).expand(B, seq_len)
            src_key_padding_mask = (mask >= lengths_expanded)
        else:
            src_key_padding_mask = None
            
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        if lengths is not None and len(lengths) > 0:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
            out = out * valid_mask
            sum_out = out.sum(dim=1)
            valid_lengths = torch.clamp(lengths.unsqueeze(1).float(), min=1.0)
            pooled_out = sum_out / valid_lengths
        else:
            pooled_out = out.mean(dim=1)
            
        pooled_out = self.dropout(pooled_out)
        logits = self.fc(pooled_out)
        return logits
