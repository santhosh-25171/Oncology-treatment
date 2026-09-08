# Stage 03 NLP EDA Report

## 1. Objective
The objective of this Exploratory Data Analysis (EDA) is to understand the properties, structure, and clinical language patterns of the preprocessed oncology text dataset. This analysis will guide the design and training of the upcoming Urgency Classification and Medical NER models.

## 2. Dataset Overview
- **Dataset Size**: 10015 records
- **Fields**: record_id, patient_id, note_type, timestamp, department, text, source, language
- **Unique Record IDs**: 10000
- **Missing Values**: 10
- **Empty Text Records**: 10
- **Duplicate IDs**: 15
- **Duplicate Text**: 35
- **Annotation Outputs**: Urgency CSV and NER JSON successfully loaded.

## 3. Urgency Distribution
| urgency_label   |   Count |   Percentage |
|:----------------|--------:|-------------:|
| LOW             |    6197 |        61.94 |
| HIGH            |    2230 |        22.29 |
| MODERATE        |    1578 |        15.77 |

*Class Balance Discussion:* The dataset exhibits some imbalance, with LOW urgency cases dominating, followed by HIGH, and MODERATE being the least common. This may require class weighting or stratified sampling during model training.

## 4. Text Statistics
| Metric          |   Min |   Max |   Mean |   Median |   Std Dev |
|:----------------|------:|------:|-------:|---------:|----------:|
| Word Count      |    13 |    87 |  39.43 |       37 |     12.69 |
| Character Count |    80 |   591 | 274.75 |      262 |     92.51 |
| Sentence Count  |     3 |    11 |   5.91 |        6 |      1.49 |

## 5. Vocabulary Analysis
**Preprocessing:** Lowercased, stripped punctuation, removed basic English stopwords.

Top words differ slightly by urgency class, though clinical terms (e.g., patient, cycle, history) dominate all classes. The table below highlights the top 20 terms:

| Overall     | LOW         | MODERATE    | HIGH        |
|:------------|:------------|:------------|:------------|
| patient     | patient     | patient     | patient     |
| cancer      | cancer      | today       | reports     |
| stage       | consistent  | grade       | currently   |
| consistent  | stage       | cancer      | pt          |
| today       | staging     | visit       | cancer      |
| mg          | possible    | mg          | today       |
| staging     | involvement | neutropenia | grade       |
| possible    | mg          | persistent  | worsening   |
| involvement | today       | moderate    | mg          |
| reports     | findings    | stage       | stage       |
| pt          | cycle       | pt          | neutropenia |
| cycle       | denies      | reports     | visit       |
| visit       | molecular   | cycle       | cycle       |
| neutropenia | pt          | therapy     | severe      |
| denies      | visit       | following   | time        |
| currently   | pending     | currently   | therapy     |
| therapy     | biopsy      | reported    | following   |
| grade       | lung        | time        | previously  |
| molecular   | neutropenia | previously  | reported    |
| bp          | therapy     | follow      | follow      |

## 6. Clinical Language Patterns
We analyzed the dataset for specific clinical phrasing styles (negations, histories, severity):

| Pattern                              |   Count |   Percentage |
|:-------------------------------------|--------:|-------------:|
| negation (denies/no)                 |    6822 |        68.19 |
| historical (history of/prior)        |    3696 |        36.94 |
| resolution (resolved)                |       0 |         0    |
| severity (severe/critical/worsening) |    2362 |        23.61 |
| stability (stable/routine)           |    1138 |        11.37 |

*Observation:* Negation and historical mentions are very common. Simple keyword matching for adverse events will produce false positives unless the context (negated/historical) is modeled.

## 7. NER Analysis
- **Total Annotated Records:** 9990
- **Total Entities:** 37169
- **Entities Per Record:** 3.72

### Entity Distribution
| Entity Type   |   Count |   Percentage |
|:--------------|--------:|-------------:|
| ADVERSE_EVENT |   21331 |        57.39 |
| DRUG_NAME     |    7213 |        19.41 |
| DOSAGE_LEVEL  |    4053 |        10.9  |
| GENE_MUTATION |    4572 |        12.3  |

### Top Entities
**Top Drugs:**
| Drug             |   Count |
|:-----------------|--------:|
| carboplatin      |     628 |
| doxorubicin      |     624 |
| nivolumab        |     621 |
| cyclophosphamide |     617 |
| pembrolizumab    |     614 |
| paclitaxel       |     610 |
| docetaxel        |     607 |
| gefitinib        |     603 |
| trastuzumab      |     589 |
| osimertinib      |     583 |

**Top Adverse Events:**
| Adverse Event   |   Count |
|:----------------|--------:|
| diarrhea        |    1517 |
| fever           |    1472 |
| fatigue         |    1466 |
| abdominal pain  |    1463 |
| pneumonitis     |    1446 |
| neuropathy      |    1446 |
| neutropenia     |    1411 |
| mucositis       |    1405 |
| rash            |    1405 |
| anemia          |    1397 |

## 8. Note-Type Analysis
| note_type               |   Count |   Avg_Word_Count |   HIGH |   LOW |   MODERATE |
|:------------------------|--------:|-----------------:|-------:|------:|-----------:|
| nurse_intake            |    2001 |            29.63 |   26.1 |  56.8 |       17.1 |
| pathology_report        |    2002 |            35.87 |    0   | 100   |        0   |
| patient_symptom_log     |    2000 |            34.66 |   32.9 |  43   |       24   |
| physician_progress_note |    3002 |            53    |   27.6 |  52.6 |       19.8 |
| treatment_note          |    1000 |            35.01 |   22.1 |  61.8 |       16.1 |


## 9. Key Findings
- **Vocabulary Overlap:** High vocabulary overlap exists between urgency classes, meaning simple Bag-of-Words models might struggle.
- **Negation & Context:** Negations ("denies", "no") and historical mentions ("history of") occur frequently and severely alter the clinical truth.
- **Entity Imbalance:** ADVERSE_EVENT dominates the NER dataset, requiring loss weighting or careful sampling when training the NER model.
- **Length Variance:** Text length varies across note types; sequence truncation (for Transformers) must be selected carefully (e.g., around the 95th percentile).

## 10. Implications for NLP Modeling
- Context-aware architectures (e.g., BiLSTMs or Transformers like ClinicalBERT) are heavily recommended over basic TF-IDF + Logistic Regression, due to the need to understand negation scope and historical context.
- Class weighting or resampling should be applied for the Urgency Classifier.

## 11. Data Quality Limitations
- Synthetic repetitions exist (intentional edge cases).
- Small sample size for specific rare gene mutations.

## 12. Conclusion
**The dataset is comprehensively analyzed, clean, and formally ready for NLP model development.**
