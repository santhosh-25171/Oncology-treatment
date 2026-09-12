# Stage 03 NLP Model Development Report

## 1. Objective
To construct a text classification pipeline and medical entity extraction (NER) models for the synthetic oncology clinical-text dataset.

## 2. PDF Stage 03 Requirement
**"Construct text classification pipeline and medical entity extraction NER models."**
This fulfills the NLP Engineer role before hand-off to the Evaluation Engineer.

## 3. Existing Input Datasets
- **Cleaned Texts:** `stage3_nlp/data/cleaned/clinical_text_cleaned.csv`
- **Urgency Labels:** `stage3_nlp/data/classification/urgency_classification.csv`
- **NER Labels:** `stage3_nlp/data/ner/ner_annotations.json`

## 4. Data Splitting
Using `sklearn.model_selection.train_test_split`, reproducible stratified splits were created (70% train, 15% validation, 15% test) focusing on preserving the `urgency_label` class distribution. The ID lists are stored in `stage3_nlp/data/splits/` to guarantee no data leakage during subsequent evaluation.

## 5. Urgency Classification Approach
A classical NLP approach was adopted. Given the relatively constrained vocabulary typical of synthetic clinical data, a Bag-of-Words paradigm is a strongly explainable and computationally efficient baseline. 

## 6. Text Representation
- **TF-IDF Vectorizer**: `ngram_range=(1,2)` and `max_features=10000` combined with standard English stop words removal. 

## 7. Classification Model
- **Logistic Regression**: Linear models perform well with sparse TF-IDF matrices, yielding directly interpretable feature weights.

## 8. Class Imbalance Handling
The `LOW` urgency class significantly outnumbers `MODERATE` and `HIGH`. To handle this, Logistic Regression was configured with `class_weight='balanced'`, which assigns inversely proportional weights to class frequencies, improving macro-recall for minority classes.

## 9. NER Approach
A blank English `spaCy` pipeline (`spacy.blank("en")`) was initialized, and a fresh Named Entity Recognition (`ner`) component was appended. We trained it natively using `spacy.training.Example`, which natively enforces no-overlapping-spans.

## 10. Entity Labels
Strictly confined to:
- `GENE_MUTATION`
- `DRUG_NAME`
- `DOSAGE_LEVEL`
- `ADVERSE_EVENT`

## 11. Training Process
- **Classification**: Fitted on the 70% train split. (Validation and Test hold-outs are strictly preserved).
- **NER**: Spacy SGD optimization executed over the training slice in minibatches for 5 iterations.

## 12. Model Artifacts
Stored sequentially inside:
- `stage3_nlp/models/classification/urgency_model.pkl` (TF-IDF + LR pipeline).
- `stage3_nlp/models/ner/` (SpaCy directory artifacts).

## 13. Prediction Pipeline
A unified local interface (`predict.py`) exposes both models. Given a raw clinical text snippet, it concurrently infers the urgency string and the structured dictionary of detected entity spans.

## 14. Tests
Verified using `pytest` inside `test_models.py`, ensuring:
- Model payloads deserialize.
- Inferences yield valid schemas.
- Entity offsets exactly match underlying strings.

## 15. Limitations
The classical TF-IDF approach lacks deep contextual awareness (e.g., struggling with complex negations that cross sentence boundaries). The shallow spacy NER model requires far more tuning and epochs to generalize beyond verbatim strings. 

## 16. Synthetic-Data Limitation
**IMPORTANT:** The underlying dataset and annotations are 100% synthetically generated. Thus, **model performance does NOT establish any clinical validity**. These pipelines serve purely as a software engineering proof-of-concept for the architecture.

## 17. What Comes Next
**Evaluation Engineer:** Extensive evaluation over the held-out test splits (Precision, Recall, F1, confusion matrices, and detailed Misinterpretation Audit Logs).
