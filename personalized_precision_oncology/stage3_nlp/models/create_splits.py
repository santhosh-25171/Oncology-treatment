import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def create_splits():
    print("Loading data for splitting...")
    URGENCY_CSV = os.path.join(BASE_DIR, "stage3_nlp", "data", "classification", "urgency_classification.csv")
    df = pd.read_csv(URGENCY_CSV)
    
    # 70% train, 15% val, 15% test
    # We stratify by urgency label
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["urgency_label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["urgency_label"], random_state=42)
    
    splits_dir = os.path.join(BASE_DIR, "stage3_nlp", "data", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    train_path = os.path.join(splits_dir, "train_ids.csv")
    val_path = os.path.join(splits_dir, "val_ids.csv")
    test_path = os.path.join(splits_dir, "test_ids.csv")
    
    train_df[["record_id"]].to_csv(train_path, index=False)
    val_df[["record_id"]].to_csv(val_path, index=False)
    test_df[["record_id"]].to_csv(test_path, index=False)
    
    print(f"Splits created successfully:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

if __name__ == "__main__":
    create_splits()
