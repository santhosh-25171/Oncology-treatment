# Stage 03 — Natural Language Processing (Mini Demo)

**Mission:** Extract structure from unstructured 911 texts/dispatcher logs -
classify urgency, and pull out Location / Resource Needed / Headcount.

## Run order

| Script | Role | What it does |
|---|---|---|
| `01_data_engineer.py` | Data Engineer | Loads + cleans raw text logs, adds urgency labels |
| `02_eda_engineer.py` | EDA Engineer | Compares word usage in urgent vs routine messages |
| `03_nlp_engineer.py` | NLP Engineer | Trains an urgency text classifier + a simple rule-based NER |
| `04_evaluation_engineer.py` | Evaluation Engineer | Checks precision/recall, inspects wrong predictions |
| `05_integration_engineer.py` | Integration Engineer | Live intake function: text in -> tagged entities + urgency out |
