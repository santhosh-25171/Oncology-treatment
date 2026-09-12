import os
import pandas as pd
import re
import unicodedata
import hashlib
from sanitize_data import validate_raw_data

def compute_checksum(filepath):
    """Compute MD5 checksum of a file to verify it hasn't changed."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def clean_clinical_text(text):
    """
    Conservatively cleans clinical text:
    - Replaces NaNs with empty strings
    - Normalizes Unicode safely (NFKC)
    - Removes excessive whitespaces and linebreaks
    - Preserves all medical entities, negations, abbreviations, dosages.
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # Unicode normalization (safe for clinical text, normalizes smart quotes etc.)
    text = unicodedata.normalize('NFKC', text)
    
    # Normalize whitespace (replaces newlines, tabs, multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing
    text = text.strip()
    
    return text

def run_cleaning_pipeline(raw_path, cleaned_path):
    print(f"Loading raw dataset from {raw_path}...")
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
        
    initial_checksum = compute_checksum(raw_path)
    print(f"Raw data checksum: {initial_checksum}")
    
    # Load dataset
    df = pd.read_csv(raw_path)
    
    # Sanitize / Validate
    print("Running pre-cleaning sanitization checks...")
    raw_stats = validate_raw_data(df)
    print(f"Raw Stats: {raw_stats}")
    
    # Clean text
    print("Cleaning clinical text...")
    df_cleaned = df.copy()
    df_cleaned["text"] = df_cleaned["text"].apply(clean_clinical_text)
    
    # Also clean missing values in other required string columns
    string_cols = ["note_type", "department", "source", "language"]
    for col in string_cols:
        df_cleaned[col] = df_cleaned[col].fillna("UNKNOWN")
        
    # We do NOT drop duplicates or change record_ids based on the prompt rules.
    # We must keep 10,000 rows exactly.
    
    # Save cleaned data
    print(f"Saving cleaned dataset to {cleaned_path}...")
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df_cleaned.to_csv(cleaned_path, index=False)
    
    # Verify raw file unchanged
    final_checksum = compute_checksum(raw_path)
    if initial_checksum != final_checksum:
        raise RuntimeError("CRITICAL ERROR: The raw dataset file was modified during cleaning!")
    
    print("Success. Raw dataset integrity verified.")
    return df, df_cleaned, raw_stats

if __name__ == "__main__":
    RAW_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "clinical_text_raw.csv")
    CLEANED_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned", "clinical_text_cleaned.csv")
    run_cleaning_pipeline(RAW_CSV, CLEANED_CSV)
