import os
import pandas as pd
import re
import json

def get_urgency_label(text):
    if pd.isna(text) or not str(text).strip():
        return None
    
    text_lower = str(text).lower()
    
    # Hide negated contexts so we don't trigger urgency on them
    # e.g., "denies severe nausea", "history of severe nausea", "severe nausea resolved"
    negations = [r"denies", r"no", r"without", r"history of", r"prior"]
    for neg in negations:
        text_lower = re.sub(rf'\b{neg}\b(?:\s+\w+){{1,4}}', ' NEGATED_CONTEXT ', text_lower)
    
    text_lower = re.sub(r'(?:\w+\s+){1,3}resolved', ' NEGATED_CONTEXT ', text_lower)
    
    if re.search(r'\b(severe|grade 3|grade 4|critical|emergency|worsening|life-threatening|major toxicity)\b', text_lower):
        return "HIGH"
    elif re.search(r'\b(moderate|grade 2|persistent|significant|intervention)\b', text_lower):
        return "MODERATE"
    else:
        return "LOW"

def annotate_urgency(df):
    results = []
    excluded = 0
    for _, row in df.iterrows():
        text = row['text']
        label = get_urgency_label(text)
        if label:
            results.append({
                "record_id": row['record_id'],
                "text": text,
                "urgency_label": label
            })
        else:
            excluded += 1
    return pd.DataFrame(results), excluded

if __name__ == "__main__":
    CLEANED_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned", "clinical_text_cleaned.csv")
    OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "classification", "urgency_classification.csv")
    
    df = pd.read_csv(CLEANED_CSV, keep_default_na=False)
    
    # Exclude duplicates based on record_id to ensure clean annotations
    df = df.drop_duplicates(subset=["record_id"])
    
    # Filter empty texts
    df_valid = df[df['text'].str.strip() != ""]
    
    urgency_df, excluded = annotate_urgency(df_valid)
    
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    urgency_df.to_csv(OUT_CSV, index=False)
    print(f"Urgency annotation complete. Annotated: {len(urgency_df)}, Excluded empty/invalid: {len(df) - len(urgency_df)}")
