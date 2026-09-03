"""
ROLE: EVALUATION ENGINEER (GenAI stage)
JOB: "Audit generated scenarios to verify they realistically stress
system logic."

WHY THIS STEP EXISTS:
Generated data is only useful if it's REALISTIC. A scenario with negative
rainfall or an impossible gauge level would be useless (or misleading) for
testing. This role sanity-checks the output.
"""

import pandas as pd

df = pd.read_pickle("data/historical_df.pkl")
synthetic = pd.read_csv("data/synthetic_scenarios.csv")

print("Sanity checks on generated scenarios:\n")
for i, row in synthetic.iterrows():
    checks = []
    checks.append("rainfall in plausible range" if 0 <= row["rainfall_mm"] <= 150 else "UNREALISTIC rainfall!")
    checks.append("gauge level in plausible range" if 0 <= row["gauge_level_m"] <= 10 else "UNREALISTIC gauge!")
    checks.append("call volume in plausible range" if 0 <= row["call_volume"] <= 100 else "UNREALISTIC calls!")

    print(f"Scenario {i+1}: {dict(row)}")
    for c in checks:
        print(f"   - {c}")
    print()

print("If all checks pass, these scenarios are ready to stress-test the ML/DL/NLP pipeline.")
