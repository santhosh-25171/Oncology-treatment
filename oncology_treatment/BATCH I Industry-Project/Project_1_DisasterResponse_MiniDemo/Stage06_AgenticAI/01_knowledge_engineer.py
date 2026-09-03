"""
ROLE: KNOWLEDGE ENGINEER
JOB: "Construct the SOP knowledge base and interface tools for shelter
registries & fleet APIs."

WHY THIS STEP EXISTS:
An agent can't reason without knowledge. This role builds the "facts" the
agent will use: Standard Operating Procedures (SOPs) + live resource
inventory (ambulances, shelter beds).
"""

import pandas as pd

# STEP 1: Load the live resource inventory (like a real fleet/shelter API would return)
inventory = pd.read_csv("data/resource_inventory.csv")
print("Resource Inventory loaded:")
print(inventory)

# STEP 2: Build a small SOP knowledge base (rules the agent must follow)
# In a real project this comes from actual safety documents / SOP PDFs.
sop_knowledge_base = {
    "SEVERE_ZONE_RULE": "A Severe zone MUST receive an ambulance within priority order.",
    "TIE_BREAK_RULE": "If two Severe zones compete for one ambulance, prioritize the "
                       "zone with the HIGHER predicted water level (from Stage 2 DL).",
    "SHELTER_RULE": "Never dispatch people to a shelter that is already at full capacity.",
    "OVERRIDE_RULE": "A human commander can pause or override any agent decision at any time.",
}

print("\nSOP Knowledge Base:")
for rule, text in sop_knowledge_base.items():
    print(f"  [{rule}] {text}")

# Save both for the next role (Workflow Engineer) to use
inventory.to_pickle("data/inventory.pkl")
import json
with open("data/sop_knowledge_base.json", "w") as f:
    json.dump(sop_knowledge_base, f, indent=2)
print("\nSaved inventory.pkl and sop_knowledge_base.json")
