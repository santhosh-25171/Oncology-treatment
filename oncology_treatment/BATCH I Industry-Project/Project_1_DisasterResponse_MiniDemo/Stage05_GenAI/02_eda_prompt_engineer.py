"""
ROLE: EDA / PROMPT ENGINEER
JOB: "Identify blind spots in historical data and craft prompts for
extreme edge scenarios."

WHY THIS STEP EXISTS:
Historical data alone won't include RARE, extreme combinations (e.g. very
high rainfall AND very high call volume happening together). This role
identifies which combinations are missing/rare, so the GenAI Engineer
knows what to specifically target when generating synthetic data.
"""

import pandas as pd

df = pd.read_pickle("data/historical_df.pkl")

# STEP 1: Find the observed range of each signal
print("Observed ranges in historical data:")
for col in ["rainfall_mm", "gauge_level_m", "call_volume"]:
    print(f"  {col}: min={df[col].min()}, max={df[col].max()}")

# STEP 2: Define a "blind spot" - a compound extreme not yet seen
# e.g. rainfall > max observed AND call_volume > max observed together
max_rain = df["rainfall_mm"].max()
max_calls = df["call_volume"].max()

print(f"\nBLIND SPOT identified: no historical case has rainfall > {max_rain}mm "
      f"AND call_volume > {max_calls} happening TOGETHER.")
print("This is exactly the kind of compound extreme scenario the GenAI "
      "Engineer should generate next, to stress-test the pipeline.")

# A "prompt" for a real LLM-based generator would look like this:
prompt = (f"Generate a disaster scenario with rainfall above {max_rain}mm, "
          f"call volume above {max_calls}, combined with a power outage.")
print(f"\nExample prompt for a real GenAI system:\n  '{prompt}'")
