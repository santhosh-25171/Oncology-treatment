"""
ROLE: EDA ENGINEER (SLM stage)
JOB: "Audit token distribution to ensure specialized radio codes are
retained."

WHY THIS STEP EXISTS:
If summarization accidentally drops important tactical words (like "zone",
"ambulance", "closed"), the briefing becomes useless to a responder. This
role checks those key words survive.
"""

import pandas as pd
from collections import Counter
import re

df = pd.read_pickle("data/prepared_reports.pkl")

IMPORTANT_TERMS = ["zone", "ambulance", "closed", "evacuat", "injur", "shelter", "dispatch"]

for _, row in df.iterrows():
    words = re.findall(r"[a-z]+", row["full_report"].lower())
    counts = Counter(words)
    print(f"{row['report_id']} — key term counts:")
    for term in IMPORTANT_TERMS:
        found = sum(c for w, c in counts.items() if term in w)
        if found:
            print(f"  '{term}': appears {found} time(s) -> must be KEPT in summary")
    print()
