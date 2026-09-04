import os
import pandas as pd
import numpy as np
from typing import Tuple, List

class OncologySequenceLoader:
    def __init__(self, data_path: str, seq_length: int = 8):
        self.data_path = data_path
        self.seq_length = seq_length
        self.features = ['ctdna_level', 'biomarker_2', 'tumor_volume']
        self.target = 'outcome'
        
    def load_and_split(self) -> Tuple[Tuple[np.ndarray, np.ndarray, List[str]], 
                                      Tuple[np.ndarray, np.ndarray, List[str]], 
                                      Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Loads the longitudinal dataset, sorts chronologically, groups by patient,
        and splits by PATIENT (to prevent patient leakage) into Train/Val/Test.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Sequence data not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        # Sort chronologically strictly by patient and day to preserve temporal order
        df = df.sort_values(by=['patient_id', 'day'])
        
        # Extract unique patients and convert to numpy array to safely shuffle
        patients = df['patient_id'].unique()
        if isinstance(patients, pd.core.arrays.string_.StringArray):
            patients = patients.to_numpy()
            
        # Shuffle with fixed seed for reproducible splits
        np.random.seed(42)
        np.random.shuffle(patients)
        
        # 70% Train, 15% Val, 15% Test
        n = len(patients)
        train_p = patients[:int(0.7*n)]
        val_p = patients[int(0.7*n):int(0.85*n)]
        test_p = patients[int(0.85*n):]
        
        return self._build_sequences(df, train_p), self._build_sequences(df, val_p), self._build_sequences(df, test_p)

    def _build_sequences(self, df: pd.DataFrame, patient_subset: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Groups data by patient, pads/truncates sequences to consistent length deterministically,
        and creates tensors.
        """
        subset_df = df[df['patient_id'].isin(patient_subset)]
        
        X, y = [], []
        patient_ids = []
        
        for pid, group in subset_df.groupby('patient_id'):
            features_data = group[self.features].values
            
            # Deterministic truncation/padding
            if len(features_data) > self.seq_length:
                # Truncate to the first `seq_length` observations
                features_data = features_data[:self.seq_length]
            elif len(features_data) < self.seq_length:
                # Pad with NaNs at the end
                pad_len = self.seq_length - len(features_data)
                padding = np.full((pad_len, len(self.features)), np.nan)
                features_data = np.vstack((features_data, padding))
                
            X.append(features_data)
            
            # Binary target: 1 for Responder, 0 for Non-Responder
            outcome = group[self.target].iloc[-1]
            y.append(1 if outcome == 'Responder' else 0)
            patient_ids.append(str(pid))
            
        return np.array(X), np.array(y), patient_ids

    def validate_leakage(self, train_p: List[str], val_p: List[str], test_p: List[str]) -> bool:
        """
        Checks for patient intersection across splits to confirm no patient leakage exists.
        Returns True if zero leakage is detected.
        """
        s1 = set(train_p)
        s2 = set(val_p)
        s3 = set(test_p)
        
        intersection = s1.intersection(s2).union(s1.intersection(s3)).union(s2.intersection(s3))
        return len(intersection) == 0
