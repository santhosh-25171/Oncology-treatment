"""
ROLE: EVALUATION ENGINEER (Deep Learning stage)
JOB: "Refine confusion matrices until the model stops mistaking wet asphalt
for deep floods."

WHY THIS STEP EXISTS:
A vision model that confuses "shiny wet road" with "deep flood" could send
an ambulance to the wrong place. We check exactly WHERE it gets confused.
"""

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

vision_model = joblib.load("data/vision_model.pkl")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

y_pred = vision_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Vision model accuracy on unseen test images: {acc:.0%}")

labels = sorted(set(y_test))
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("\nConfusion matrix (rows=actual, columns=predicted):")
print(f"{'':10}" + "".join(f"{l:>10}" for l in labels))
for i, row_label in enumerate(labels):
    print(f"{row_label:10}" + "".join(f"{v:>10}" for v in cm[i]))

# Flag the exact failure mode the problem statement warns about
false_negatives = ((y_test == "Flooded") & (y_pred == "Clear")).sum()
print(f"\nDangerous misses (real flood called 'Clear'): {false_negatives}")
print("This is the number the team must drive to ZERO before real deployment.")
