"""
ROLE: INTEGRATION ENGINEER (SLM stage)
JOB: "Package the SLM into an offline tactical briefing mobile/desktop
app."

WHY THIS STEP EXISTS:
Wraps the summarizer into ONE simple function that works with zero
internet/cloud dependency - exactly what a field commander's device needs.
"""

import re

IMPORTANT_TERMS = {"zone", "ambulance", "closed", "evacuation", "injury", "shelter",
                    "dispatch", "critical", "immediate", "rainfall", "collapse"}


def score_sentence(sentence):
    words = re.findall(r"[a-z]+", sentence.lower())
    return sum(1 for w in words if any(term in w for term in IMPORTANT_TERMS))


def get_field_briefing(full_report, keep=2):
    """This IS the offline briefing app's core function - no internet needed."""
    sentences = [s.strip() for s in full_report.split(". ") if s.strip()]
    scored = sorted(sentences, key=score_sentence, reverse=True)[:keep]
    ordered = [s for s in sentences if s in scored]
    return ". ".join(ordered) + "."


if __name__ == "__main__":
    incoming_report = (
        "Zone 6 river gauge crossed critical level at 3 AM. Rainfall totaled 88mm. "
        "Two roads closed near the market area. Ambulance unit 2 dispatched to "
        "assist stranded residents. Shelter at Zone 6 school has 15 beds available. "
        "No injuries reported so far."
    )
    print("=== LIVE DEMO: Offline 2-line briefing ===")
    print("FULL REPORT:\n", incoming_report)
    print("\nBRIEFING:\n", get_field_briefing(incoming_report))
