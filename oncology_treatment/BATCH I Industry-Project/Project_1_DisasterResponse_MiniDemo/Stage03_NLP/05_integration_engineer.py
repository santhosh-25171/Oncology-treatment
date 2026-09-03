"""
ROLE: INTEGRATION ENGINEER (NLP stage)

JOB:
Wire the trained urgency classifier and entity extraction
into one live function that the dashboard can call.
"""

import re
import joblib


# =========================================================
# LOAD TRAINED NLP MODEL
# =========================================================

vectorizer = joblib.load("data/vectorizer.pkl")

clf = joblib.load(
    "data/urgency_classifier.pkl"
)


# =========================================================
# ENTITY EXTRACTION
# =========================================================

def extract_entities(text):

    text_lower = text.lower()


    # -----------------------------------------------------
    # LOCATION
    # Example: Zone 2
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
    # Examples:
    # 8 people
    # 3 persons
    # 2 children
    # No people
    # -----------------------------------------------------

    headcount_match = re.search(
        r"(\d+)\s*"
        r"(people|persons|person|children|residents)",
        text_lower
    )

    if headcount_match:

        headcount = headcount_match.group(1)

    elif re.search(
        r"\bno\s+"
        r"(people|persons|person|residents)\b",
        text_lower
    ):

        headcount = "0"

    elif re.search(
        r"\bno\s+one\b",
        text_lower
    ):

        headcount = "0"

    else:

        headcount = "Not found"


    # -----------------------------------------------------
    # RESOURCE EXTRACTION
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


    return {

        "Location": location,

        "Headcount": headcount,

        "Resource Needed": resource

    }


# =========================================================
# LIVE NLP PROCESSING FUNCTION
# =========================================================

def process_incoming_message(text):

    """
    This function is called by app.py.

    Step 1:
    Predict urgency using TF-IDF + Logistic Regression.

    Step 2:
    Extract location, people and required resources.
    """

    urgency = clf.predict(
        vectorizer.transform(
            [text.lower()]
        )
    )[0]


    entities = extract_entities(
        text
    )


    return {

        "urgency": urgency,

        **entities

    }


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    print(
        "=== LIVE DEMO: Emergency message ==="
    )


    incoming = (
        "Help! Water rising fast. "
        "8 people are trapped near Zone 4. "
        "Send a rescue boat immediately."
    )


    result = process_incoming_message(
        incoming
    )


    print(
        f"\nMessage: {incoming}\n"
    )


    for key, value in result.items():

        print(
            f"{key}: {value}"
        )