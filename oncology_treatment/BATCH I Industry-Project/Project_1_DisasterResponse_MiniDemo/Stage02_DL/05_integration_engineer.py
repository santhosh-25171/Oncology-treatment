"""
ROLE: INTEGRATION ENGINEER (Deep Learning stage)
JOB: "Fuse vision and time-series scores into a single unified alert API."

WHY THIS STEP EXISTS:
The vision model and the sequence model each know only PART of the story.
This role combines both into ONE alert a dashboard can show.
"""

import numpy as np
import joblib

vision_model = joblib.load("data/vision_model.pkl")
seq_params = joblib.load("data/sequence_model.pkl")


def get_flood_alert(image_flat, current_hour, current_level):
    """This IS the unified alert API endpoint."""
    vision_call = vision_model.predict([image_flat])[0]
    predicted_level_3h = seq_params["intercept"] + seq_params["slope"] * (current_hour + 3)

    alert = "SEVERE" if (vision_call == "Flooded" and predicted_level_3h > 4.5) else \
            "WATCH" if vision_call == "Flooded" or predicted_level_3h > 4.0 else "CLEAR"

    return {
        "camera_says": vision_call,
        "predicted_level_in_3h": round(predicted_level_3h, 2),
        "unified_alert": alert,
    }


if __name__ == "__main__":
    test_image = np.load("data/X_test.npy")[0]
    result = get_flood_alert(test_image, current_hour=6, current_level=4.2)
    print("=== LIVE DEMO: Unified Vision + Trend Alert ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
