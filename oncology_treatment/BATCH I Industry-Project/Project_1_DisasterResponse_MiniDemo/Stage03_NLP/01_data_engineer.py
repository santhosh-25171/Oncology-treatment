"""
ROLE: DATA ENGINEER (NLP stage)
JOB: "Gather, sanitize, and annotate raw text logs for urgency
classification and NER."

WHY THIS STEP EXISTS:
Raw text is messy - extra spaces, inconsistent casing, punctuation. This
role cleans the text BEFORE anyone tries to analyze or model it.
"""

import pandas as pd

df = pd.read_csv("data/raw_dispatch_logs.csv")

# STEP 1: Basic text cleaning - lowercase + strip extra whitespace
df["clean_message"] = df["message"].str.strip().str.lower()

# STEP 2: Remove any completely empty/duplicate messages
df = df.drop_duplicates(subset="clean_message").dropna(subset=["clean_message"])

print(f"Loaded and cleaned {len(df)} dispatcher messages.")
print(df[["clean_message", "urgency_label"]].head())

df.to_csv("data/clean_logs.csv", index=False)
print("\nSaved to data/clean_logs.csv")
