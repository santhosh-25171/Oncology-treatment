"""
ROLE: EDA ENGINEER (Exploratory Data Analysis)
JOB (from problem statement): "Plot leading indicators; hunt down and
eliminate historical false alarms."

WHY THIS STEP EXISTS:
Before training a model, we must LOOK at the data and check whether it
actually makes sense. If a "Severe" label was recorded by mistake (a false
alarm), and we train on it anyway, the model learns the WRONG pattern.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lets this run without a screen (saves plot to a file)
import matplotlib.pyplot as plt

df = pd.read_csv("data/master_dataset.csv")

# STEP 1: Basic pattern check - does more rainfall generally mean higher risk?
print("Average rainfall per risk level:")
print(df.groupby("risk_label")["rainfall_mm"].mean())

# STEP 2: Visualize rainfall vs call_volume, colored by risk label
# (This is exactly what "Feature Leaderboard" will later be based on)
colors = {"Low": "green", "Moderate": "orange", "Severe": "red"}
plt.figure(figsize=(6, 4))
for label, group in df.groupby("risk_label"):
    plt.scatter(group["rainfall_mm"], group["call_volume"],
                label=label, color=colors[label])
plt.xlabel("Rainfall (mm)")
plt.ylabel("911 Call Volume")
plt.title("Rainfall vs Calls, colored by Risk Level")
plt.legend()
plt.savefig("data/eda_pattern_plot.png")
print("\nPlot saved to data/eda_pattern_plot.png")

# STEP 3: Hunt down a "false alarm" - a row that doesn't fit the pattern
# Example rule: high call_volume but very low rainfall/gauge is suspicious
# (could be a prank call spree, not a real flood)
suspicious = df[(df["call_volume"] > 20) & (df["rainfall_mm"] < 30)]
print("\nSuspicious rows found (possible false alarms):")
print(suspicious if not suspicious.empty else "None found in this mini dataset - clean!")

# In this mini dataset there are none - saved as-is for the ML Engineer.
# (In the real project, EDA Engineer would DROP or FIX confirmed false alarms here.)
df.to_csv("data/eda_checked_dataset.csv", index=False)
print("\nEDA-checked dataset saved to data/eda_checked_dataset.csv")
