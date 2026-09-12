import pandas as pd
import re

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b"
}

REQUIRED_COLUMNS = [
    "record_id", "patient_id", "note_type", "timestamp",
    "department", "text", "source", "language"
]

def check_pii(text):
    if pd.isna(text):
        return False
    text = str(text)
    for pattern in PII_PATTERNS.values():
        if re.search(pattern, text):
            return True
    return False

def validate_raw_data(df):
    """
    Validates the raw dataset and returns a dictionary of statistics.
    Does NOT modify the dataframe.
    """
    stats = {}
    
    # 1. Row/Col counts
    stats["rows"] = len(df)
    stats["columns"] = len(df.columns)
    stats["missing_values"] = df.isnull().sum().to_dict()
    stats["total_missing"] = df.isnull().sum().sum()
    
    # 2. Duplicate checks
    stats["duplicate_record_ids"] = int(df["record_id"].duplicated().sum()) if "record_id" in df else 0
    stats["duplicate_text"] = int(df["text"].duplicated().sum()) if "text" in df else 0
    
    # 3. Empty text
    if "text" in df:
        stats["empty_text"] = int(df["text"].apply(lambda x: pd.isna(x) or str(x).strip() == "").sum())
    else:
        stats["empty_text"] = 0

    # 4. PII check
    if "text" in df:
        stats["pii_flags"] = int(df["text"].apply(check_pii).sum())
    else:
        stats["pii_flags"] = 0
        
    # 5. Invalid values check (basic)
    if "language" in df:
        stats["invalid_languages"] = int((~df["language"].isin(["en"])).sum())
    
    # 6. Column check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    stats["missing_columns"] = missing_cols
    
    return stats
