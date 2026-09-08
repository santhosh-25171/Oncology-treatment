import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, cnn_feature_dim, sequence_feature_dim, num_classes):
        super(MultimodalFusion, self).__init__()
        # Combines CNN representation + Temporal representation
        
    def forward(self, image_x, temporal_x):
        pass
