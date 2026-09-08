import torch.nn as nn

class HistopathologyCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(HistopathologyCNN, self).__init__()
        # Classes: normal, benign, malignant, tumor_margin, necrotic, inflammatory
        # Input: 3 x 224 x 224
        self.num_classes = num_classes
        
    def forward(self, x):
        pass
