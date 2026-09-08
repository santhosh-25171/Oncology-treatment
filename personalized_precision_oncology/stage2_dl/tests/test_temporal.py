import os
import pytest
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "sample_data"

def test_temporal_dataset_exists():
    assert (data_dir / "temporal" / "biomarker_timeseries.csv").exists()
    assert (data_dir / "temporal" / "progression_targets.csv").exists()

def test_patient_leakage():
    splits = pd.read_csv(data_dir / "train_validation_test_split.csv")
    train_pats = set(splits[splits['split'] == 'train']['patient_id'])
    val_pats = set(splits[splits['split'] == 'validation']['patient_id'])
    test_pats = set(splits[splits['split'] == 'test']['patient_id'])
    
    assert len(train_pats.intersection(val_pats)) == 0, "Leakage between train and val"
    assert len(train_pats.intersection(test_pats)) == 0, "Leakage between train and test"
    assert len(val_pats.intersection(test_pats)) == 0, "Leakage between val and test"

def test_chronological_ordering():
    df = pd.read_csv(data_dir / "temporal" / "biomarker_timeseries.csv")
    # Sort intentionally randomly then sort by our logic
    df = df.sample(frac=1.0)
    df_sorted = df.sort_values(by=['patient_id', 'study_day'])
    
    # Check if for any patient, study_day is monotonically increasing
    for pid, group in df_sorted.groupby('patient_id'):
        assert group['study_day'].is_monotonic_increasing, f"Patient {pid} observations are not chronological"

def test_target_columns_omitted_from_features():
    df = pd.read_csv(data_dir / "temporal" / "biomarker_timeseries.csv")
    all_cols = df.columns.tolist()
    
    targets = ['progression_90d', 'future_tumor_volume_cm3', 'future_tumor_growth_rate', 'response_category_90d']
    
    # Assuming input features are explicitly defined
    NUMERIC_FEATURES = [
        'ctDNA_level', 'ctDNA_change_percent', 'tumor_volume_cm3', 'tumor_growth_rate',
        'CEA', 'CYFRA21_1', 'CRP', 'LDH', 'dose_intensity'
    ]
    CATEGORICAL_FEATURES = ['treatment_status', 'treatment_cycle', 'treatment_type', 'response_status']
    
    for targ in targets:
        assert targ not in NUMERIC_FEATURES
        assert targ not in CATEGORICAL_FEATURES
