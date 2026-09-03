"""
ROLE: DL ENGINEER
JOB: "Build and train the CNN vision network and LSTM/Transformer sequence
model."

WHY THIS STEP EXISTS:
Two separate deep learning jobs happen in this stage:
 1. VISION: look at an image, say Flooded or Clear (CNN's job)
 2. SEQUENCE: look at past water-level readings, predict the next one (LSTM's job)

SIMPLIFIED FOR CLASSROOM: We use scikit-learn's MLPClassifier (a small
neural network) as a stand-in for a real CNN, and simple linear trend
fitting as a stand-in for a real LSTM/Transformer. Same INPUT->OUTPUT
shape as the real thing, tiny enough to train instantly.
"""

import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ---------- PART A: Vision model (CNN stand-in) ----------
images = np.load("data/images.npy")
labels = np.load("data/labels.npy")

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.25, random_state=42
)

vision_model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=42)
vision_model.fit(X_train, y_train)
joblib.dump(vision_model, "data/vision_model.pkl")
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)
print(f"Vision model trained. Training accuracy: {vision_model.score(X_train, y_train):.0%}")

# ---------- PART B: Sequence model (LSTM/Transformer stand-in) ----------
# Toy water-level readings over the last 6 hours for one zone (rising trend)
water_levels = np.array([1.2, 1.6, 2.1, 2.9, 3.6, 4.2])
hours = np.arange(len(water_levels)).reshape(-1, 1)

# A simple straight-line trend fit stands in for a real LSTM/Transformer,
# which would learn much more complex, non-straight-line patterns.
slope, intercept = np.polyfit(hours.flatten(), water_levels, 1)
next_3_hours = [intercept + slope * (len(water_levels) + h) for h in range(1, 4)]

print(f"\nWater level trend so far: {water_levels}")
print(f"Predicted water level, next 3 hours: {[round(v, 2) for v in next_3_hours]}")

joblib.dump({"slope": slope, "intercept": intercept}, "data/sequence_model.pkl")
