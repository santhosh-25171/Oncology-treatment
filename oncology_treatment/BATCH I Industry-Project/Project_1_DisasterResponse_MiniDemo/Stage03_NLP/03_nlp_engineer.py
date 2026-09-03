"""
ROLE: NLP ENGINEER
JOB: Construct the text classification pipeline and entity extraction NER models.

TWO JOBS:
1. Classify each message as Urgent/Routine
2. Extract Location, Headcount and Resource Needed

CLASSROOM VERSION:
TF-IDF + Logistic Regression for urgency classification.
Rule-based entity extraction as a simple NER stand-in.
"""

import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/clean_logs.csv")


# =========================================================
# PART A — URGENCY TEXT CLASSIFIER
# =========================================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    df["clean_message"]
)

y = df["urgency_label"]


clf = LogisticRegression()

clf.fit(X, y)


# Save trained NLP models

joblib.dump(
    vectorizer,
    "data/vectorizer.pkl"
)

joblib.dump(
    clf,
    "data/urgency_classifier.pkl"
)


print(
    f"Urgency classifier trained. "
    f"Training accuracy: {clf.score(X, y):.0%}"
)


# =========================================================
# PART B — RULE-BASED NER
# =========================================================

def extract_entities(text):

    text_lower = text.lower()

    # -----------------------------------------------------
    # LOCATION
    # -----------------------------------------------------

    zone_match = re.search(
        r"zone\s*(\d+)",
        text_lower
    )

    location = (
        f"Zone {zone_match.group(1)}"
        if zone_match
        else "Not found"
    )


    # -----------------------------------------------------
    # HEADCOUNT
    # -----------------------------------------------------

    # Example:
    # "8 people"
    # "3 persons"
    # "2 children"

    headcount_match = re.search(
        r"(\d+)\s*(people|persons|person|children|residents)",
        text_lower
    )

    if headcount_match:

        headcount = headcount_match.group(1)

    # Explicitly handle:
    # "no people"
    # "no persons"
    # "no one"

    elif re.search(
        r"\bno\s+(people|persons|person|residents)\b",
        text_lower
    ) or re.search(
        r"\bno\s+one\b",
        text_lower
    ):

        headcount = "0"

    else:

        headcount = "Not found"


    # -----------------------------------------------------
    # RESOURCE
    # -----------------------------------------------------

    resource_keywords = [
        "boat",
        "ambulance",
        "rescue",
        "help",
        "shelter"
    ]

    found_resources = [
        word
        for word in resource_keywords
        if word in text_lower
    ]


    if found_resources:

        resource = ", ".join(
            found_resources
        )

    else:

        resource = "None"


    # -----------------------------------------------------
    # RETURN NER RESULTS
    # -----------------------------------------------------

    return {

        "Location": location,

        "Headcount": headcount,

        "Resource Needed": resource

    }


# =========================================================
# SAMPLE NER TEST
# =========================================================

print(
    "\nSample NER extraction on 2 messages:"
)


for msg in df["message"].head(2):

    print(
        f"  Message: {msg}"
    )

    print(
        f"  Extracted: {extract_entities(msg)}\n"
    )