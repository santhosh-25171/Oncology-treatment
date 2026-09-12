import torch
from torch.utils.data import Dataset
from PIL import Image

class HistopathologyDataset(Dataset):
    def __init__(self, metadata_df, images_dir, transform=None):
        self.metadata = metadata_df
        self.images_dir = images_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        # Skeleton for RGB loading, 224x224 input
        pass
