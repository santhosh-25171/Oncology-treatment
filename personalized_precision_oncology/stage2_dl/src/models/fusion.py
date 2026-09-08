import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class MultimodalFusionModel(nn.Module):
    def __init__(self, cnn_model, transformer_model, cnn_embed_dim=128, temporal_embed_dim=64, num_classes=2):
        super(MultimodalFusionModel, self).__init__()
        self.cnn = cnn_model
        self.transformer = transformer_model
        
        # Freeze both branches entirely
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        self.image_proj = nn.Linear(cnn_embed_dim, 64)
        self.temporal_proj = nn.Linear(temporal_embed_dim, 64)
        
        self.fusion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def extract_image_features(self, images):
        # Extract features before the final FC
        with torch.no_grad():
            x = self.cnn.block1(images)
            x = self.cnn.block2(x)
            x = self.cnn.block3(x)
            x = self.cnn.gap(x)
            x = torch.flatten(x, 1)
        return x.mean(dim=0) # Mean pooling over tiles

    def extract_temporal_features(self, x, lengths):
        # Extract features before the final FC using the Transformer structure
        with torch.no_grad():
            batch_size, max_seq_len, _ = x.shape
            device = x.device
            mask = torch.arange(max_seq_len, device=device).expand(batch_size, max_seq_len) >= lengths.unsqueeze(1).to(device)
            
            x_proj = self.transformer.input_projection(x)
            x_pos = self.transformer.pos_encoder(x_proj)
            output = self.transformer.transformer_encoder(x_pos, src_key_padding_mask=mask)
            
            valid_mask = ~mask
            valid_mask = valid_mask.unsqueeze(-1).float()
            sum_output = torch.sum(output * valid_mask, dim=1)
            lengths_float = torch.clamp(lengths.unsqueeze(1).float().to(device), min=1.0)
            mean_output = sum_output / lengths_float
        return mean_output
        
    def forward(self, images_list, temporal_x, temporal_lengths):
        device = temporal_x.device
        img_embeds = []
        for imgs in images_list:
            if imgs.numel() == 0:
                # Handle cases where a patient has no valid images
                img_embeds.append(torch.zeros(128, device=device))
            else:
                img_embeds.append(self.extract_image_features(imgs.to(device)))
                
        img_embeds = torch.stack(img_embeds) # [batch_size, 128]
        temp_embeds = self.extract_temporal_features(temporal_x, temporal_lengths) # [batch_size, 64]
        
        # Project
        img_rep = self.image_proj(img_embeds)
        temp_rep = self.temporal_proj(temp_embeds)
        
        # Concatenate and pass through fusion head
        fused = torch.cat([img_rep, temp_rep], dim=1) # [batch_size, 128]
        logits = self.fusion_head(fused) # [batch_size, 2]
        
        return logits
