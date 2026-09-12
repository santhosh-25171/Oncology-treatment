# Stage 03 — NLP

This stage implements the NLP pipeline for extracting structured insights from unstructured oncology clinical text.

## Pipeline Architecture

```text
Raw Clinical Text
        ↓
Cleaning / Sanitization
        ↓
Annotation
        ↓
EDA
        ↓
Urgency Classifier + NER
        ↓
Evaluation
        ↓
Integration
```

**Note:** The current implementation consists ONLY of:
```text
Raw Dataset
      ↓
Cleaning / Sanitization
```

The NLP models, evaluations, and API integration are placeholders for future development steps and have not been implemented yet.
