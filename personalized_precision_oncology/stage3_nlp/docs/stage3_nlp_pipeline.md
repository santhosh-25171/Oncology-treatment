# Stage 3 — Natural Language Processing (NLP) Pipeline

## Overview
Stage 03 implements an end-to-end clinical NLP system for extracting structured oncology insights from unstructured clinical notes and consultation summaries.

```text
                     UNSTRUCTURED CLINICAL TEXT
                                 │
                     1. Data Preprocessing
                   (clean_text.py: sanitization,
                    preserves negation & medical terms)
                                 │
                     2. Annotation & Splits
                  (annotate_urgency.py & annotate_ner.py:
                   LOW/MODERATE/HIGH + 4 entity classes)
                                 │
                     3. Exploratory Text Analysis
                   (run_eda.py: 7 figures, 8 tables)
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
 4a. Urgency Classifier                     4b. Named Entity Recognizer
 (TF-IDF + Balanced Logistic Reg)          (Custom SpaCy NER Pipeline)
 (urgency_model.pkl)                       (models/ner/)
            │                                         │
            └────────────────────┬────────────────────┘
                                 │
                     5. Evaluation & Auditing
                   (evaluate_classifier.py, evaluate_ner.py:
                    Accuracy: 89.0%, NER F1: 98.7%,
                    238-item Misinterpretation Audit Log)
                                 │
                     6. Integration Layer
                   (FastAPI Microservice + Streamlit UI)
                   [Pending integration into root application]
```

---

## Key Artifacts & Directories

| Subsystem | Location | Description |
| :--- | :--- | :--- |
| **Raw & Cleaned Data** | `stage3_nlp/data/` | Raw text (`clinical_text_raw.csv`), cleaned text, annotations, splits |
| **Models** | `stage3_nlp/models/` | Urgency Classifier (`urgency_model.pkl`), SpaCy NER (`models/ner/`), `predict.py` |
| **EDA Artifacts** | `stage3_nlp/artifacts/eda/` | Word count, vocabulary, and entity distribution plots & tables |
| **Evaluation Metrics** | `stage3_nlp/artifacts/evaluation/` | Confusion matrix, metrics JSON, `misinterpretation_audit_log.csv` |
| **Unit Tests** | `stage3_nlp/tests/` | 27 automated tests (100% passing) |
