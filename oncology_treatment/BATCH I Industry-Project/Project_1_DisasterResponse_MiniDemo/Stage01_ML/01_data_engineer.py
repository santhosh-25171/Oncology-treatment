"""
ROLE: DATA ENGINEER
JOB (from problem statement): "Stitch gauge, weather, and 911 logs into a
clean, timestamped master dataset."

WHY THIS STEP EXISTS:
Raw data from different sensors/systems never arrives clean. Here we have
3 separate files (gauge readings, 911 calls, past outcomes). Before ANY
model can be trained, someone has to merge them into ONE table and fix
missing/duplicate values. If this step is skipped or done wrong, every
role after this one inherits the mistake.
"""

import pandas as pd

# STEP 1: Load the 3 raw source files (this mimics 3 different real systems)
gauge_df = pd.read_csv("data/raw_river_gauge.csv")        # rainfall + river level
calls_df = pd.read_csv("data/raw_911_calls.csv")          # emergency call volume
flood_log_df = pd.read_csv("data/raw_historical_flood_log.csv")  # past outcome labels

print("Raw 911 calls file BEFORE cleaning:")
print(calls_df)

# STEP 2: Fix real-world messiness
# 2a. Remove exact duplicate rows (Z1 appears twice in raw_911_calls.csv)
calls_df = calls_df.drop_duplicates()

# 2b. Handle missing values (Z10 has a blank call_volume/road_closures)
#     Simple beginner-safe fix: fill missing numeric values with the column average
calls_df["call_volume"] = calls_df["call_volume"].fillna(calls_df["call_volume"].mean())
calls_df["road_closures"] = calls_df["road_closures"].fillna(0)

# STEP 3: Merge all 3 sources into ONE master table, matched by zone_id
# (In the real project this would also be matched by timestamp, not just zone)
master_df = gauge_df.merge(calls_df, on="zone_id").merge(flood_log_df, on="zone_id")

# STEP 4: Save the clean master dataset for the next role (EDA Engineer) to use
master_df.to_csv("data/master_dataset.csv", index=False)

print("\nCleaned MASTER dataset (saved to data/master_dataset.csv):")
print(master_df)
print(f"\nRows before cleaning: {len(calls_df) + 1} (with duplicate) -> Rows after: {len(master_df)}")
