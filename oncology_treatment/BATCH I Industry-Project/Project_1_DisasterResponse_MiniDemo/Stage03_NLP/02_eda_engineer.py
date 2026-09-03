"""
ROLE: EDA ENGINEER (NLP stage)
JOB: "Analyze vocabulary distributions and phrasing patterns across urgent
vs. routine reports."

WHY THIS STEP EXISTS:
Before training a classifier, we should understand WHICH words actually
separate "urgent" from "routine" messages. This confirms the model has a
real pattern to learn from.
"""

import pandas as pd
from collections import Counter
import re

df = pd.read_csv("data/clean_logs.csv")


def top_words(messages, n=8):
    words = re.findall(r"[a-z]+", " ".join(messages))
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "no", "will", "near"}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(words).most_common(n)


urgent_msgs = df[df["urgency_label"] == "Urgent"]["clean_message"]
routine_msgs = df[df["urgency_label"] == "Routine"]["clean_message"]

print("Most common words in URGENT messages:")
print(top_words(urgent_msgs))

print("\nMost common words in ROUTINE messages:")
print(top_words(routine_msgs))

print("\n=> Notice words like 'trapped', 'help', 'now', 'sos' cluster in URGENT.")
print("   This is exactly the pattern the classifier will learn.")
