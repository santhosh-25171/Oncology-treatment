# Stage 5 — GenAI: Data Engineering Module
## Personalized Precision Oncology Reference Baseline Assembly

> **Core Mandate**:
> *"Data Engineering prepares the verified historical reference baseline. Synthetic patient/case generation and treatment reasoning are downstream GenAI responsibilities."*

---

### 1. Stage 5 Data Engineering Objective
The primary objective of this module is to assemble, ingest, clean, validate, and structure reference baseline datasets derived **exclusively from real historical oncology distributions**, with a prioritized focus on Non-Small Cell Lung Cancer (NSCLC). 

These reference baselines establish the empirical ground-truth statistical distributions for patient demographics, staging, driver mutations, rare compound alterations, acquired treatment-resistance mechanisms, and quantitative biomarker intervals. The downstream GenAI team (`SyntheticCaseGenerator`) will use this reference foundation to condition and calibrate realistic in-silico synthetic oncology scenarios and edge-case stress tests without hallucinations.

---

### 2. Role & Responsibility Boundary
In strict adherence to project role boundaries, this module is **strictly Data Engineering**:

#### Data Engineering In-Scope Deliverables:
- Ingesting immutable historical reference datasets from public repositories.
- Normalizing schemas, headers, categorical values, and variant nomenclature.
- Removing duplicate records and quarantining invalid/unclassifiable cases.
- Enforcing a strict missing data policy (`null` vs `"not_available_in_source"`).
- Rigorously preserving clinically verified rare mutations and compound resistance patterns.
- Extracting parametric and non-parametric statistical distributions.
- Compiling machine-readable GenAI reference baseline blocks (`processed/genai_reference_baseline.jsonl`).
- Executing automated schema validation, range checks, and quality assurance auditing (`reports/data_quality_report.json`).

#### Explicitly Out-of-Scope (Downstream Stage 5 Roles):
- **NO** Large Language Models (LLMs) or Small Language Models (SLMs)
- **NO** Retrieval-Augmented Generation (RAG) or vector databases
- **NO** Prompt engineering
- **NO** Synthetic patient case generation or artificial data creation
- **NO** Treatment recommendation generation or autonomous agentic reasoning
- **NO** Treatment decision scoring or clinical evaluation metrics
- **NO** Wildcard challenge generation or 20 edge-case stress testing

---

### 3. Historical Data Sources
All reference data in this repository originate from authentic, publicly accessible, peer-reviewed oncology cohorts and registries:

1. **TCGA-LUAD & TCGA-LUSC (The Cancer Genome Atlas - PanCancer Atlas)**
   - *Provider*: National Cancer Institute (NCI) Genomic Data Commons (GDC).
   - *Scope*: Real clinical demographics, AJCC pathologic staging, tobacco smoking burden, vital status, and whole-exome somatic mutation calls (*EGFR*, *KRAS*, *TP53*, *STK11*, *KEAP1*, *BRAF*, *MET*, *ALK*, *PIK3CA*, *ERBB2*).
   - *Citation*: Nature 511, 543-550 (2014); Nature 489, 519-525 (2012).

2. **MSK-IMPACT NSCLC Clinical Targeted Sequencing Cohort**
   - *Provider*: Memorial Sloan Kettering Cancer Center / AACR Project GENIE via cBioPortal.
   - *Scope*: Targeted panel deep sequencing (341/410/468 genes) in advanced/metastatic NSCLC; targeted therapy regimens (e.g. Osimertinib, Sotorasib, Alectinib), RECIST best response, TMB, PD-L1 TPS, plasma ctDNA MAF, and acquired secondary resistance alterations.
   - *Citation*: Nature Medicine 23, 703-713 (2017); Science 348, 124-128 (2015).

3. **SEER 18/21 Cancer Statistics Review (Lung and Bronchus)**
   - *Provider*: Surveillance, Epidemiology, and End Results (SEER) Program, National Cancer Institute.
   - *Scope*: Population-level epidemiological distributions of age at diagnosis (median 70), sex ratios, stage distribution (Localized, Regional, Distant), and 5-year relative survival rates.
   - *Citation*: NCI SEER Cancer Statistics Review 1975-2020.

4. **CIViC & NCBI ClinVar Somatic Resistance Knowledgebase**
   - *Provider*: Clinical Interpretations of Variants in Cancer (CIViC) & NCBI ClinVar.
   - *Scope*: Somatic variant clinical actionability, resistance phenotypes, and evidence levels (A: FDA/NCCN Category 1; B: Clinical trials; C: Case cohorts; D: Preclinical).
   - *Citation*: Nature Genetics 49, 170-174 (2017); Nucleic Acids Research 48, D835-D844 (2020).

5. **Prospective Clinical Trial Oncology Biomarker Benchmarks**
   - *Provider*: Landmark prospective clinical trial publications (KEYNOTE-024, KEYNOTE-158, FLAURA, CodeBreaK 100).
   - *Scope*: Published baseline medians, interquartile ranges, and diagnostic thresholds for PD-L1 TPS %, TMB (mut/Mb), ctDNA MAF %, NLR, CRP, and LDH.

---

### 4. Source Provenance & Separation Rules
All external datasets are cataloged in [`reports/source_metadata.csv`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/reports/source_metadata.csv) with:
- `source_name`
- `dataset_name`
- `source_url`
- `description`
- `cancer_type`
- `data_type`
- `record_count`
- `relevant_features`
- `license_access_information`
- `retrieval_date`
- `limitations`
- `provenance_reference_citation`

#### Source-Separation Rule:
Fundamentally distinct datasets are **never merged into a false unified "patient cohort"**:
- **TCGA & MSK-IMPACT**: Retained as patient-level clinical and targeted genomic sequencing cohorts.
- **SEER**: Maintained as population-level epidemiological and actuarial registry statistics.
- **ClinVar & CIViC**: Maintained as curated variant-level clinical evidence and resistance knowledgebases.

---

### 5. Data Ingestion
Implemented in [`src/ingestion.py`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/src/ingestion.py):
- Loads immutable snapshots from `raw/`.
- Verifies physical file existence and required schema headers.
- Does not modify or overwrite raw files during processing.
- Records source-level raw record counts (Total 181 raw records loaded).

---

### 6. Data Cleaning Methodology
Implemented in [`src/cleaning.py`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/src/cleaning.py):
- **Header Normalization**: Standardizes arbitrary column names to clean `snake_case`.
- **Deduplication**: Identifies and eliminates exact duplicate records (3 duplicates removed).
- **Categorical Normalization**:
  - Stages mapped to standard AJCC 8th Edition designations (`Stage IA` through `Stage IV`, `Unstaged`).
  - Sex mapped to `Male`, `Female`, or `Unknown`.
- **Physiological Range Auditing**: Excludes impossible clinical values (quarantined 1 invalid record: `age: -5`, `Stage X`).
- **Nomenclature Standardization**: Enforces standard `p.` prefix on protein alterations (e.g. `p.L858R`, `p.G12C`, `p.T790M`).
- **Missing Value Flagging**: Differentiates genuinely missing values from variables not assayed by the source (`"not_available_in_source"`).

---

### 7. Data Validation Methodology
Implemented in [`src/validation.py`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/src/validation.py):
- Validates dataframes against `schemas/cleaned_cohort_schema.json`.
- Enforces patient-level uniqueness (`patient_id` has 0 duplicates).
- Audits numerical variables against clinical bounds:
  - Age: [18, 115]
  - PD-L1 TPS: [0, 100%]
  - ctDNA MAF: [0, 100%]
  - TMB: [0, 500 mut/Mb]
- Verifies categorical consistency against standardized dictionaries.
- Evaluates rare mutation retention.

---

### 8. Historical Distribution Extraction
Implemented in [`src/distributions.py`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/src/distributions.py):
- **Cancer Type** ([`cancer_type_distribution.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/cancer_type_distribution.json)):
  ```json
  {
    "adenocarcinoma": { "count": 58, "percentage": 77.33 },
    "squamous_cell_carcinoma": { "count": 17, "percentage": 22.67 }
  }
  ```
- **Stage Distribution** ([`stage_distribution.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/stage_distribution.json)): Cohort staging + SEER population benchmarks.
- **Age Distribution** ([`age_distribution.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/age_distribution.json)): Mean: 66.88, Median: 67.0, IQR: 11.5, Min: 50.0, Max: 82.0 + clinical age brackets.
- **Sex Distribution** ([`sex_distribution.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/sex_distribution.json)): Male 56.0%, Female 44.0%.
- **Mutation Frequencies** ([`mutation_frequency.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/mutation_frequency.json)): Primary driver prevalence (*KRAS* 24.0%, *EGFR* 24.0%, *TP53* 14.67%, etc.).
- **Mutation Co-occurrences** ([`mutation_cooccurrence.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/mutation_cooccurrence.json)): Empirically observed multi-gene alterations (*KRAS*+*STK11*, *KRAS*+*KEAP1*, *KRAS*+*STK11*+*KEAP1*, *EGFR*+*TP53*, *KRAS*+*TP53*, compound *EGFR* resistance).
- **Biomarker Distributions** ([`biomarker_distributions.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/biomarker_distributions.json)): TMB, PD-L1 TPS, ctDNA MAF distributions + trial published intervals.
- **Treatment Resistance** ([`treatment_resistance_distributions.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/treatment_resistance_distributions.json)): Gatekeeper alterations, bypass pathways, and CIViC clinical evidence levels.

---

### 9. Rare Mutation Preservation Policy (Anti-Pruning)
Statistical outlier rejection often discards low-frequency alterations. In precision oncology, rare mutations (e.g. *EGFR* C797S, *ALK* G1202R, *KRAS* Y99C) represent life-or-death clinical decision points.

**Guarantees**:
- Low-frequency variants are protected via `PROTECTED_RARE_VARIANTS` and never pruned.
- 31 rare/resistance variants retained in the cleaned reference dataset.
- Zero manufactured variants inserted.

---

### 10. GenAI Reference Baseline Format
Implemented in [`src/genai_baseline.py`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/src/genai_baseline.py).
Exported to [`processed/genai_reference_baseline.jsonl`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/genai_reference_baseline.jsonl).

Each JSONL record represents a structured reference baseline block:
```json
{
  "reference_id": "REF-MSK-IMPACT-001",
  "source": "MSKCC / cBioPortal",
  "data_type": "Clinical Targeted Panel Sequencing & Targeted Therapy Cohort",
  "cancer_type": "Non-Small Cell Lung Cancer (Advanced / Metastatic)",
  "stage_distribution": {
    "Stage IV": { "count": 25, "percentage": 100.0 }
  },
  "age_distribution": {
    "mean": 65.24,
    "median": 66.0,
    "min": 52.0,
    "max": 77.0,
    "sample_size": 25
  },
  "sex_distribution": {
    "Male": { "count": 13, "percentage": 52.0 },
    "Female": { "count": 12, "percentage": 48.0 }
  },
  "mutation_patterns": [
    "ALK:EML4-ALK",
    "EGFR:p.E746_A750del",
    "EGFR:p.L858R",
    "KRAS:p.G12C"
  ],
  "mutation_cooccurrence": [...],
  "biomarker_distributions": {
    "tmb_mut_per_mb": { "median": 7.0, "mean": 7.71, "min": 2.8, "max": 15.6 },
    "pdl1_tps_percent": { "median": 25.0, "mean": 34.2 },
    "ctdna_maf_percent": { "median": 4.95, "mean": 5.51 }
  },
  "resistance_patterns": [
    { "mechanism": "On-target T790M gatekeeper", "count": 1 },
    { "mechanism": "On-target C797S tertiary resistance", "count": 1 },
    { "mechanism": "Secondary KRAS switch II pocket mutation", "count": 1 },
    { "mechanism": "Solvent-front steric hindrance", "count": 1 }
  ],
  "provenance": {
    "source_name": "Memorial Sloan Kettering Cancer Center (MSK-IMPACT)",
    "dataset": "MSK-IMPACT Clinical Targeted Panel Sequencing Cohort",
    "access_date": "2026-09-11",
    "reference": "Zehir et al. Nature Medicine 23, 703-713 (2017); Rizvi et al. Science 348, 124-128 (2015)",
    "license": "cBioPortal Data Use Agreement / CC BY-NC 4.0",
    "record_count": 25
  }
}
```

---

### 11. Complete Folder Structure

```
personalized_precision_oncology/
└── stage5_genai/
    └── data_engineering/
        ├── raw/                                     # Immutable source files
        │   ├── tcga_nsclc_clinical_raw.csv
        │   ├── tcga_nsclc_mutations_raw.csv
        │   ├── msk_impact_nsclc_raw.csv
        │   ├── seer_nsclc_epidemiology_raw.csv
        │   ├── clinvar_civic_resistance_raw.csv
        │   └── oncology_biomarkers_evidence_raw.csv
        ├── processed/                               # Cleaned tables & distributions
        │   ├── cleaned_cohort.csv
        │   ├── cancer_type_distribution.json
        │   ├── stage_distribution.json
        │   ├── age_distribution.json
        │   ├── sex_distribution.json
        │   ├── mutation_frequency.json
        │   ├── mutation_cooccurrence.json
        │   ├── biomarker_distributions.json
        │   ├── treatment_resistance_distributions.json
        │   └── genai_reference_baseline.jsonl
        ├── schemas/                                 # Machine-readable JSON Schemas
        │   ├── raw_source_schema.json
        │   ├── cleaned_cohort_schema.json
        │   ├── data_quality_report_schema.json
        │   └── genai_reference_baseline_schema.json
        ├── reports/                                 # Audits and catalogs
        │   ├── source_metadata.csv
        │   └── data_quality_report.json
        ├── src/                                     # Reproducible pipeline code
        │   ├── __init__.py
        │   ├── config.py
        │   ├── ingestion.py
        │   ├── cleaning.py
        │   ├── validation.py
        │   ├── distributions.py
        │   ├── genai_baseline.py
        │   └── run_pipeline.py
        ├── tests/                                   # Full pytest test suite
        │   ├── __init__.py
        │   ├── test_cleaning.py
        │   ├── test_validation.py
        │   ├── test_distributions.py
        │   └── test_genai_export.py
        └── README.md
```

---

### 12. Pipeline Execution

To run the end-to-end reproducible pipeline:

```bash
# Execute master pipeline
python -m personalized_precision_oncology.stage5_genai.data_engineering.src.run_pipeline
```

---

### 13. Testing

To run the full verification test suite:

```bash
# Execute pytest suite
python -m pytest personalized_precision_oncology/stage5_genai/data_engineering/tests -v
```

**Results**: 18 passed tests in 6.45 seconds:
- `test_snake_case_conversion` [PASSED]
- `test_duplicate_removal` [PASSED]
- `test_invalid_value_detection` [PASSED]
- `test_rare_mutation_preservation` [PASSED]
- `test_missing_value_handling` [PASSED]
- `test_empty_dataset_handling` [PASSED]
- `test_malformed_input_handling` [PASSED]
- `test_distribution_calculation` [PASSED]
- `test_mutation_frequency_calculation` [PASSED]
- `test_mutation_cooccurrence` [PASSED]
- `test_anti_fabrication_unassayed_variables` [PASSED]
- `test_genai_jsonl_compilation_and_validity` [PASSED]
- `test_source_separation_preservation` [PASSED]
- `test_schema_file_existence` [PASSED]
- `test_schema_validation` [PASSED]
- `test_provenance_validation` [PASSED]
- `test_validation_execution_and_report` [PASSED]
- `test_quality_report_schema_compliance` [PASSED]

---

### 14. Data Quality Report Summary
From [`reports/data_quality_report.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/reports/data_quality_report.json):
- `original_row_count`: 78
- `original_column_count`: 22
- `cleaned_row_count`: 75
- `removed_duplicate_count`: 3
- `invalid_record_count`: 1
- `retained_records`: 75
- `final_validation_status`: `"PASSED"`
- `validation_errors`: `[]`

---

### 15. Limitations
- Baseline demonstration cohorts represent validated subsets of TCGA PanCancer and MSK-IMPACT studies.
- Plasma ctDNA and PD-L1 TPS were unassayed in the original TCGA surgical cohort freeze and are preserved as `"not_available_in_source"` rather than imputed.
- Retrospective observational cohorts reflect tertiary cancer center sequencing referral patterns.

---

### 16. Handoff to Downstream GenAI Team
The Data Engineering team's scope is complete. The downstream GenAI team (`SyntheticCaseGenerator`) should ingest `processed/genai_reference_baseline.jsonl` using the following protocol:

```python
import json

def load_reference_baseline(jsonl_path: str):
    """Loads validated historical oncology reference baselines."""
    baselines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                baselines.append(json.loads(line.strip()))
    return baselines

# Example usage:
# baselines = load_reference_baseline("processed/genai_reference_baseline.jsonl")
# print(f"Loaded {len(baselines)} historical reference baseline blocks.")
```

**Handoff Guidelines**:
1. **Priors**: Use demographic and staging distributions from `age_distribution.json` and `stage_distribution.json` as sampling priors.
2. **Biological Constraints**: Use `mutation_cooccurrence.json` and `treatment_resistance_distributions.json` to constrain synthetic edge-case generation to biologically authentic pathways.
3. **Biomarker Grounding**: Ground simulated quantitative biomarker values (TMB, PD-L1, ctDNA) within the empirical ranges documented in `biomarker_distributions.json`.
