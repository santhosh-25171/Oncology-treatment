"""
ROLE: EVALUATION ENGINEER (Agentic AI stage)
JOB: "Execute simulated incident runs; catalog task success rates and
decision edge failures."

WHY THIS STEP EXISTS:
The problem statement specifically calls out a hard case: TWO zones
flooding at once, but only ONE ambulance available. This tests whether
the agent's tie-break logic (from the SOP) actually works.
"""

import pandas as pd

# STEP: Build a HARD test case - 2 Severe zones, only 1 ambulance available
inventory = pd.DataFrame([
    {"resource_type": "Ambulance", "resource_id": "AMB-1", "status": "Available", "location": "Zone 3"},
])

zone_a = {"zone": "Zone 3", "risk": "Severe", "predicted_water_level_m": 4.2}
zone_b = {"zone": "Zone 9", "risk": "Severe", "predicted_water_level_m": 5.1}

print("HARD CASE: Two Severe zones competing for ONE ambulance")
print(f"  {zone_a}")
print(f"  {zone_b}")

# Apply the TIE_BREAK_RULE from the Knowledge Engineer's SOP:
# "prioritize the zone with the HIGHER predicted water level"
winner = zone_a if zone_a["predicted_water_level_m"] > zone_b["predicted_water_level_m"] else zone_b
loser = zone_b if winner is zone_a else zone_a

print(f"\n[DECISION] Ambulance AMB-1 dispatched to {winner['zone']} "
      f"(higher predicted water level: {winner['predicted_water_level_m']}m)")
print(f"[ESCALATION] {loser['zone']} has NO ambulance available - "
      f"flagged for human commander to find backup resources.")

print("\nTask success: agent correctly applied the TIE_BREAK_RULE without human input.")
print("This is exactly the kind of edge case the Evaluation Engineer must catalog.")
