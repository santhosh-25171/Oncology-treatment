import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TemporalSequenceDataset(Dataset):
    def __init__(self, temporal_df, targets_df, features):
        self.temporal_df = temporal_df
        # If 'patient_id' is already the index, don't set it again
        if targets_df.index.name != 'patient_id' and 'patient_id' in targets_df.columns:
            self.targets_df = targets_df.set_index('patient_id')
        else:
            self.targets_df = targets_df
            
        self.features = features
        self.patient_ids = self.temporal_df['patient_id'].unique()
        
        # Group by patient for fast retrieval
        self.grouped = dict(tuple(self.temporal_df.groupby('patient_id')))
        
    def __len__(self):
        return len(self.patient_ids)
        
    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        pat_data = self.grouped[pid]
        
        # Ensure chronological order
        pat_data = pat_data.sort_values('study_day')
        
        # Convert to tensor
        seq_tensor = torch.FloatTensor(pat_data[self.features].values.astype(float))
        seq_len = len(seq_tensor)
        
        # Retrieve targets
        target_row = self.targets_df.loc[pid]
        
        target_dict = {
            'progression_90d': int(target_row.get('progression_90d', 0)),
            'future_tumor_volume_cm3': float(target_row.get('future_tumor_volume_cm3', 0.0))
        }
        
        return {
            'sequence': seq_tensor,
            'length': seq_len,
            'patient_id': pid,
            'target': target_dict
        }

def temporal_collate_fn(batch):
    sequences = [item['sequence'] for item in batch]
    lengths = torch.LongTensor([item['length'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    
    # Pad sequences to max length in the batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    progression_targets = torch.LongTensor([item['target']['progression_90d'] for item in batch])
    volume_targets = torch.FloatTensor([item['target']['future_tumor_volume_cm3'] for item in batch])
    
    return {
        'padded_sequences': padded_sequences,
        'lengths': lengths,
        'patient_ids': patient_ids,
        'targets_progression': progression_targets,
        'targets_volume': volume_targets
    }
