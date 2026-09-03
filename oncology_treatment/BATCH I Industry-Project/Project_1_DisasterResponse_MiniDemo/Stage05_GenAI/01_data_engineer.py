"""
ROLE: DATA ENGINEER (GenAI stage)
JOB: "Assemble reference baseline datasets from real historical disaster
distributions."

WHY THIS STEP EXISTS:
Before generating FAKE (synthetic) disasters, we must understand what
REAL ones look like statistically - otherwise the fake ones won't be
realistic.
"""

import pandas as pd

df = pd.read_csv("data/historical_disaster_stats.csv")

print("Historical data loaded:")
print(df)

# STEP: Compute the statistical baseline (mean + spread) per risk level
# This baseline is what the GenAI Engineer will sample new scenarios from.
baseline = df.groupby("risk_label")[["rainfall_mm", "gauge_level_m", "call_volume"]].agg(["mean", "std"])
print("\nStatistical baseline per risk level (mean, std):")
print(baseline)

baseline.to_pickle("data/baseline_stats.pkl")
df.to_pickle("data/historical_df.pkl")
print("\nSaved baseline to data/baseline_stats.pkl")
