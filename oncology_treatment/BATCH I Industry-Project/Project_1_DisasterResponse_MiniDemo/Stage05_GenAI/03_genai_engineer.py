"""
ROLE: GENAI ENGINEER
JOB: "Construct the generative scenario pipeline (GAN/VAE or LLM-based
synthetic generation)."

WHY THIS STEP EXISTS:
Generate NEW, realistic-looking disaster scenarios that never actually
happened, to test the system against situations we don't have real data
for.

SIMPLIFIED FOR CLASSROOM: A real GAN/VAE/LLM pipeline is heavy. Here we
sample new values from the historical mean/spread (Gaussian sampling) -
the same underlying idea ("learn the distribution, sample new realistic
points") at a beginner-friendly scale.
"""

import pandas as pd
import numpy as np

baseline = pd.read_pickle("data/baseline_stats.pkl")
np.random.seed(1)


def generate_scenario(risk_label, n=1):
    """Samples n new synthetic rows that statistically resemble the given risk level."""
    stats = baseline.loc[risk_label]
    scenarios = []
    for _ in range(n):
        rainfall = max(0, np.random.normal(stats[("rainfall_mm", "mean")], stats[("rainfall_mm", "std")] + 1))
        gauge = max(0, np.random.normal(stats[("gauge_level_m", "mean")], stats[("gauge_level_m", "std")] + 0.2))
        calls = max(0, np.random.normal(stats[("call_volume", "mean")], stats[("call_volume", "std")] + 2))
        scenarios.append({"rainfall_mm": round(rainfall, 1),
                           "gauge_level_m": round(gauge, 1),
                           "call_volume": round(calls),
                           "risk_label": risk_label})
    return scenarios


# STEP: Generate 3 new synthetic "Severe" scenarios (the rare, high-impact case)
synthetic_scenarios = generate_scenario("Severe", n=3)

print("Generated synthetic SEVERE disaster scenarios (never actually happened):")
for s in synthetic_scenarios:
    print(" ", s)

pd.DataFrame(synthetic_scenarios).to_csv("data/synthetic_scenarios.csv", index=False)
print("\nSaved to data/synthetic_scenarios.csv")
