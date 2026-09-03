# Stage 01 — Machine Learning (Mini Demo)

**Mission (from the problem statement):** Turn raw sensor data (rainfall, river gauge,
911 calls, road closures) into an instant risk score for a city zone: Low / Moderate / Severe.

**Algorithm used:** Decision Tree Classifier (simple, explainable — matches the study
material's "classification fundamentals, decision trees" requirement).

## Run order (= the 5 squad roles, in the exact order they work in real life)

| Script | Role | What it does |
|---|---|---|
| `01_data_engineer.py` | Data Engineer | Merges 3 raw source files into one clean master table |
| `02_eda_engineer.py` | EDA Engineer | Plots patterns, finds and removes a false alarm row |
| `03_ml_engineer.py` | ML Engineer | Trains the Decision Tree risk classifier |
| `04_evaluation_engineer.py` | Evaluation Engineer | Tests the model on unseen data, checks for overconfidence |
| `05_integration_engineer.py` | Integration Engineer | Wraps the trained model in a simple "API" function |

Run them in this exact order — each script reads the file the previous script created.
