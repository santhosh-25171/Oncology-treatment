"""
ROLE: INTEGRATION ENGINEER
JOB (from problem statement): "Wrap the model in an ultra-fast API for
real-time dashboard consumption."

WHY THIS STEP EXISTS:
A trained model sitting in a notebook is useless to a real responder.
This role makes it USABLE - normally via a real web API (e.g. Flask/FastAPI)
that a dashboard can call. Here we simulate that with a simple Python
function that behaves exactly like an API: takes an input, returns a
clean JSON-style response.

(In the real large-scale project, this would be a proper Flask/FastAPI
server running continuously. For a 1-hour classroom demo we simulate the
same INPUT -> OUTPUT behaviour without needing a running server.)
"""

import joblib
import pandas as pd

model = joblib.load("data/risk_model.pkl")
features = ["rainfall_mm", "gauge_level_m", "call_volume", "road_closures"]


def get_zone_risk_score(rainfall_mm, gauge_level_m, call_volume, road_closures):
    """
    This function IS the API endpoint.
    In a real API this would be called like:  POST /predict-risk
    """
    input_row = pd.DataFrame([[rainfall_mm, gauge_level_m, call_volume, road_closures]],
                              columns=features)
    prediction = model.predict(input_row)[0]
    confidence = model.predict_proba(input_row).max()

    # This is the "Field Briefing Sheet" style response - simple, non-technical
    response = {
        "risk_level": prediction,
        "confidence": f"{confidence:.0%}",
        "recommendation": {
            "Low": "Monitor only.",
            "Moderate": "Alert local response team, prepare shelter.",
            "Severe": "Dispatch ambulance fleet + open shelter IMMEDIATELY.",
        }[prediction],
    }
    return response


# STEP: Demo call, exactly like the 2:00 AM scenario from the mission brief
if __name__ == "__main__":
    print("=== LIVE DEMO: New sensor reading comes in ===")
    result = get_zone_risk_score(
        rainfall_mm=92, gauge_level_m=4.7, call_volume=38, road_closures=3
    )
    print("API Response:")
    for key, value in result.items():
        print(f"  {key}: {value}")
