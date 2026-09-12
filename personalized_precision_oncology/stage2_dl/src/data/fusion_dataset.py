import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

class MultimodalFusionDataset(Dataset):
    def __init__(self, spatial_df, temporal_df, targets_df, data_dir, temporal_features):
        self.data_dir = data_dir
        self.temporal_features = temporal_features
        
        # Determine valid patient subset: must have both modalities! (Actually prompt says "Use all valid patients available for fusion", intersection is 2000 anyway)
        self.patient_ids = targets_df['patient_id'].values
        self.targets_df = targets_df.set_index('patient_id')
        
        self.spatial_grouped = dict(tuple(spatial_df.groupby('patient_id')))
        self.temporal_grouped = dict(tuple(temporal_df.groupby('patient_id')))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.patient_ids)
        
    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        
        # Temporal Branch
        pat_temp = self.temporal_grouped[pid].sort_values('study_day')
        seq_tensor = torch.FloatTensor(pat_temp[self.temporal_features].values.astype(float))
        seq_len = len(seq_tensor)
        
        # Spatial Branch
        if pid in self.spatial_grouped:
            pat_spat = self.spatial_grouped[pid]
            img_tensors = []
            for _, row in pat_spat.iterrows():
                raw_path = Path(row['image_path'])
                if raw_path.exists():
                    img_path = raw_path
                elif (self.data_dir / row['image_path']).exists():
                    img_path = self.data_dir / row['image_path']
                else:
                    img_path = Path(self.data_dir).resolve().parent.parent / raw_path
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensors.append(self.transform(img))
                except Exception:
                    continue
            if img_tensors:
                imgs_tensor = torch.stack(img_tensors)
            else:
                imgs_tensor = torch.empty(0, 3, 224, 224)
        else:
            imgs_tensor = torch.empty(0, 3, 224, 224)
            
        # Target
        target = int(self.targets_df.loc[pid, 'progression_90d'])
        
        return {
            'patient_id': pid,
            'images': imgs_tensor,
            'sequence': seq_tensor,
            'length': seq_len,
            'target': target
        }

def fusion_collate_fn(batch):
    images_list = [item['images'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    lengths = torch.LongTensor([item['length'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    targets = torch.LongTensor([item['target'] for item in batch])
    
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    return {
        'images_list': images_list,
        'padded_sequences': padded_sequences,
        'lengths': lengths,
        'patient_ids': patient_ids,
        'targets': targets
    }
