# Stage 5 — GenAI: Data Engineering Module
## Personalized Precision Oncology Reference Baseline Assembly

> **Core Mandate**:
> *"Data Engineering prepares and validates the authentic historical oncology reference baseline and evidence distributions. This module does NOT generate synthetic patients, train LLMs, or make treatment recommendations."*

---

### 1. Stage 5 Data Engineering Objective
The primary objective of this module is to assemble, ingest, clean, validate, and structure reference baseline datasets derived **exclusively from real historical oncology distributions**, with a prioritized focus on Non-Small Cell Lung Cancer (NSCLC).

These reference baselines establish empirical ground-truth statistical distributions for patient demographics, staging, driver mutations, rare compound alterations, acquired treatment-resistance mechanisms, and quantitative biomarker intervals.

The prepared reference data are consumed downstream by:
1. **Stage 5 EDA & Prompt Engineering**: For genomic blind-spot detection and stress-test prompt template formulation.
2. **Stage 5 GenAI Engineer**: As empirical priors and constraints for `SyntheticCaseGenerator` (dual-mode LLM & template generator).
3. **Stage 5 Evaluation Engineer**: For auditing synthetic edge-case fidelity against real-world evidence.

---

### 2. Role & Responsibility Boundary
In strict adherence to project role boundaries, this module is **strictly Data Engineering**:

#### Data Engineering In-Scope Responsibilities:
- Ingesting immutable historical reference datasets from public repositories.
- Enforcing semantic source-separation across heterogeneous data sources.
- Normalizing schemas, headers, categorical values, and variant nomenclature.
- Removing duplicate records and quarantining invalid/unclassifiable cases with structured audit logging.
- Enforcing strict missing data semantics (`null` vs `"not_available_in_source"` vs `"not_observed_in_reference"` vs `"insufficient_evidence"` vs `"source_limitation"`).
- Rigorously preserving clinically verified rare mutations and compound resistance patterns without pruning.
- Extracting parametric and non-parametric statistical distributions.
- Compiling machine-readable, source-aware GenAI reference baseline blocks (`processed/genai_reference_baseline.jsonl`).
- Executing automated schema validation, range checks, and quality assurance auditing (`reports/data_quality_report.json`).

#### Explicitly Out-of-Scope (Downstream Stage 5 Roles):
- **This module does NOT generate synthetic patients.**
- **NO** Large Language Models (LLMs) or Small Language Models (SLMs) execution.
- **NO** Retrieval-Augmented Generation (RAG) or vector databases.
- **NO** Prompt engineering or template crafting.
- **NO** Synthetic patient case generation or artificial data creation.
- **NO** Treatment recommendation generation or autonomous agentic reasoning.
- **NO** Treatment decision scoring or clinical evaluation metrics.
- **NO** Wildcard challenge generation or 20 edge-case stress testing.

---

### 3. Historical Data Sources & Provenance
All reference data originate from authentic, publicly accessible, peer-reviewed oncology cohorts, registries, and published consensus guidelines:

1. **TCGA-LUAD & TCGA-LUSC (The Cancer Genome Atlas - PanCancer Atlas)**
   - *Provider*: National Cancer Institute (NCI) Genomic Data Commons (GDC).
   - *Data Domain*: Clinical & Genomic (Patient Cohort).
   - *Scope*: Real clinical demographics, AJCC pathologic staging, tobacco smoking burden, vital status, and whole-exome somatic mutation calls (*EGFR*, *KRAS*, *TP53*, *STK11*, *KEAP1*, *BRAF*, *MET*, *ALK*, *PIK3CA*, *ERBB2*).
   - *Citation*: Nature 511, 543-550 (2014); Nature 489, 519-525 (2012).

2. **MSK-IMPACT NSCLC Clinical Targeted Sequencing Cohort**
   - *Provider*: Memorial Sloan Kettering Cancer Center / AACR Project GENIE via cBioPortal.
   - *Data Domain*: Genomic & Resistance (Patient Cohort).
   - *Scope*: Targeted panel deep sequencing (341/410/468 genes) in advanced/metastatic NSCLC; targeted therapy regimens (e.g. Osimertinib, Sotorasib, Alectinib), RECIST best response, TMB, PD-L1 TPS, plasma ctDNA MAF, and acquired secondary resistance alterations.
   - *Citation*: Nature Medicine 23, 703-713 (2017); Science 348, 124-128 (2015).

3. **SEER 18/21 Cancer Statistics Review (Lung and Bronchus)**
   - *Provider*: Surveillance, Epidemiology, and End Results (SEER) Program, National Cancer Institute.
   - *Data Domain*: Epidemiology (Population Registry).
   - *Scope*: Population-level epidemiological distributions of age at diagnosis (median 70), sex ratios, stage distribution (Localized, Regional, Distant), and 5-year relative survival rates.
   - *Citation*: NCI SEER Cancer Statistics Review 1975-2020.

4. **CIViC & NCBI ClinVar Somatic Resistance Knowledgebase**
   - *Provider*: Clinical Interpretations of Variants in Cancer (CIViC) & NCBI ClinVar.
   - *Data Domain*: Resistance & Actionability (Variant Evidence).
   - *Scope*: Somatic variant clinical actionability, resistance phenotypes, and evidence levels (A: FDA/NCCN Category 1; B: Clinical trials; C: Case cohorts; D: Preclinical).
   - *Citation*: Nature Genetics 49, 170-174 (2017); Nucleic Acids Research 48, D835-D844 (2020).

5. **Prospective Clinical Trial Oncology Biomarker Benchmarks**
   - *Provider*: Landmark prospective clinical trial publications (KEYNOTE-024, KEYNOTE-158, FLAURA, CodeBreaK 100).
   - *Data Domain*: Biomarker (Biomarker Evidence).
   - *Scope*: Published baseline medians, interquartile ranges, and diagnostic cutoffs for PD-L1 TPS %, TMB (mut/Mb), ctDNA MAF %, NLR, CRP, and LDH.
   - *Citation*: Reck M et al. NEJM 2016; Marabelle A et al. Lancet Oncol 2020; Abbosh C et al. Nature 2017.

---

### 4. Source-Separation Architecture
Heterogeneous oncology sources **MUST NOT be blindly merged into one patient-level table**:

> **Source-Separation Rule**:
> SEER epidemiological data, ClinVar/CIViC variant evidence, and biomarker reference data are maintained as source-specific reference information rather than being incorrectly merged into the patient cohort table.

- **`cleaned_cohort.csv`**: Contains exclusively authentic individual patient-level clinical and genomic records (from TCGA and MSK-IMPACT; N=75).
- **`SEER`**: Maintained as population-level epidemiological and actuarial reference data (`REF-SEER-001`).
- **`ClinVar / CIViC`**: Maintained as curated variant-level clinical resistance evidence (`REF-CIVIC-CLINVAR-001`).
- **`Biomarker Literature`**: Maintained as curated clinical trial cutoff evidence (`REF-BIOMARKERS-001`).

All external datasets are cataloged with complete provenance in [`reports/source_metadata.csv`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/reports/source_metadata.csv).

---

### 5. Missing Data Semantics
To ensure scientific accuracy and prevent misleading imputations, the pipeline establishes a strict, auditable taxonomy of missing data:

| Missingness Sentinel | Exact Semantic Definition | Clinical Example |
|---|---|---|
| `null` / `np.nan` | A patient-level value is genuinely missing where the assay was expected. | Survival data lost to follow-up. |
| `"not_available_in_source"` | The original source registry does not contain or assay that variable. | ctDNA MAF in historical TCGA-LUAD exome freeze. |
| `"not_observed_in_reference"` | A genomic mutation/pattern was not observed in the assembled reference cohort. **Note: Does NOT mean biologically impossible.** | *NTRK1* or *NRG1* rare fusions in 75-patient baseline. |
| `"insufficient_evidence"` | Some clinical evidence exists, but it is insufficient to support a definitive conclusion. | Case report-level resistance without prospective validation. |
| `"source_limitation"` | The source data type fundamentally does not support the requested metric. | Calculating patient-level VAF from population-level SEER tables. |

---

### 6. Data Cleaning & Quarantine Methodology
Implemented in [`src/cleaning.py`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/src/cleaning.py):
- **Header Normalization**: Converts headers to clean `snake_case`.
- **Deduplication**: Identifies and eliminates exact duplicate records (3 duplicate rows removed).
- **Categorical Normalization**:
  - Stages mapped to standard AJCC 8th Edition designations (`Stage IA` through `Stage IV`, `Unstaged`).
  - Sex mapped to `Male`, `Female`, or `Unknown`.
- **Physiological Range Auditing & Quarantine**:
  - Excludes biologically invalid values (e.g. `age < 18` or `> 115`, impossible stages).
  - Invalid records are **quarantined rather than silently deleted**.
  - Quarantined records are exported to [`reports/quarantine_report.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/reports/quarantine_report.json) with record ID, source, validation rule, and reason (e.g. `TCGA-ERR-9999` quarantined for `age=-5.0`, `Stage X`).
- **Nomenclature Standardization**: Enforces standard `p.` prefix on protein alterations (e.g. `p.L858R`, `p.G12C`, `p.T790M`).
- **Final Cleaned Patient Cohort**: Exactly 75 unique, fully validated patient records preserved in [`processed/cleaned_cohort.csv`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/cleaned_cohort.csv).

---

### 7. Rare Mutation Preservation Policy (Anti-Pruning)
Statistical outlier filtering often inadvertently strips rare clinical alterations. In precision oncology, rare variants represent critical decision points:
- **Guarantees**:
  - Low-frequency variants are protected via `PROTECTED_RARE_VARIANTS` and never pruned.
  - 31 rare/resistance variants retained in the cleaned reference dataset (e.g., *EGFR* C797S, *EGFR* T790M, *EGFR* L718Q, *KRAS* Y99C, *KRAS* Q61H, *ALK* G1202R, *ALK* I1171N, *ROS1* G2032R, *MET* p.D1010H).
  - Retention rate: **86.11%** of monitored oncology variants retained.
  - Zero fabricated mutations inserted.

---

### 8. Historical Baseline Distribution Extraction
Implemented in [`src/distributions.py`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/src/distributions.py):
- **Cancer Type Distribution** ([`cancer_type_distribution.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/cancer_type_distribution.json)): LUAD (77.33%) vs LUSC (22.67%).
- **Stage Distribution** ([`stage_distribution.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/stage_distribution.json)): Cohort staging + SEER population benchmarks.
- **Age Distribution** ([`age_distribution.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/age_distribution.json)): Mean: 66.88, Median: 67.0, IQR: 11.5, Min: 50.0, Max: 82.0.
- **Sex Distribution** ([`sex_distribution.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/sex_distribution.json)): Male 56.0%, Female 44.0%.
- **Mutation Frequencies** ([`mutation_frequency.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/mutation_frequency.json)): Primary driver prevalence (*KRAS* 24.0%, *EGFR* 24.0%, *TP53* 14.67%, rare drivers < 5%).
- **Mutation Co-occurrences** ([`mutation_cooccurrence.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/mutation_cooccurrence.json)): Empirically observed multi-gene alterations (*KRAS*+*STK11*, *KRAS*+*KEAP1*, *KRAS*+*STK11*+*KEAP1*, *EGFR*+*TP53*, *KRAS*+*TP53*).
- **Biomarker Distributions** ([`biomarker_distributions.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/biomarker_distributions.json)): TMB, PD-L1 TPS, ctDNA MAF distributions + trial published intervals.
- **Treatment Resistance** ([`treatment_resistance_distributions.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/treatment_resistance_distributions.json)): Gatekeeper alterations, bypass pathways, and CIViC clinical evidence levels.

---

### 9. Source-Aware GenAI Reference Baseline JSONL
Implemented in [`src/genai_baseline.py`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/src/genai_baseline.py).
Exported to [`processed/genai_reference_baseline.jsonl`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/processed/genai_reference_baseline.jsonl).

Every reference record is explicitly source-aware, adhering to [`schemas/genai_reference_baseline_schema.json`](file:///c:/Users/svpoo/OneDrive/Desktop/New%20folder%20%283%29/personalized_precision_oncology/stage5_genai/data_engineering/schemas/genai_reference_baseline_schema.json):
- `reference_id` (e.g. `REF-TCGA-LUAD-001`, `REF-MSK-IMPACT-001`, `REF-SEER-001`, `REF-CIVIC-CLINVAR-001`, `REF-BIOMARKERS-001`)
- `source_name`
- `source_type`: `patient_cohort` | `epidemiology` | `variant_evidence` | `biomarker_evidence` | `literature_reference`
- `source_record_id`
- `data_domain`: `clinical` | `genomic` | `epidemiology` | `resistance` | `biomarker` | `treatment_response`
- `evidence_status`: `prospective_clinical_cohort` | `retrospective_surgical_cohort` | `population_registry` | `curated_clinical_evidence` | `published_guideline_benchmark`
- `provenance` (source dataset, study, publication, access information, license, record count)

The JSONL acts as a **unified reference baseline**, NOT as an artificial single-table patient merge.

---

### 10. Complete Folder Structure

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
        │   ├── cleaned_cohort.csv                   # Patient cohort (N=75)
        │   ├── cancer_type_distribution.json
        │   ├── stage_distribution.json
        │   ├── age_distribution.json
        │   ├── sex_distribution.json
        │   ├── mutation_frequency.json
        │   ├── mutation_cooccurrence.json
        │   ├── biomarker_distributions.json
        │   ├── treatment_resistance_distributions.json
        │   └── genai_reference_baseline.jsonl       # Source-aware reference blocks (6 lines)
        ├── schemas/                                 # Machine-readable JSON Schemas
        │   ├── raw_source_schema.json
        │   ├── cleaned_cohort_schema.json
        │   ├── data_quality_report_schema.json
        │   └── genai_reference_baseline_schema.json
        ├── reports/                                 # Audits and catalogs
        │   ├── source_metadata.csv                  # Full provenance catalog
        │   ├── data_quality_report.json             # Automated QA validation report
        │   └── quarantine_report.json               # Quarantined record logs
        ├── src/                                     # Reproducible pipeline code
        │   ├── __init__.py
        │   ├── config.py
        │   ├── ingestion.py
        │   ├── cleaning.py
        │   ├── validation.py
        │   ├── distributions.py
        │   ├── genai_baseline.py
        │   └── run_pipeline.py
        ├── tests/                                   # Full pytest test suite (30 tests)
        │   ├── __init__.py
        │   ├── test_cleaning.py
        │   ├── test_validation.py
        │   ├── test_distributions.py
        │   ├── test_genai_export.py
        │   └── test_quality_and_provenance.py
        └── README.md
```

---

### 11. Pipeline Execution & Testing

```bash
# Execute master pipeline
python -m personalized_precision_oncology.stage5_genai.data_engineering.src.run_pipeline

# Execute complete test suite
python -m pytest personalized_precision_oncology/stage5_genai/data_engineering/tests -v
```

**Verification Results**: **30 passed in 2.46s** (100% pass rate).

---

### 12. Downstream Handoff Architecture
The verified outputs flow directly into the next Stage 5 modules:

```
Data Engineering (Baseline Reference Data)
        ↓
genai_reference_baseline.jsonl + Baseline Distributions
        ↓
EDA / Prompt Engineering (Blind Spot Detection & Prompt Templates)
        ↓
blind_spot_report.json + Prompt Templates
        ↓
GenAI Scenario Generator (SyntheticCaseGenerator)
        ↓
20 Validated Extreme Drug-Resistance Scenarios (synthetic_edge_cases.jsonl)
        ↓
Evaluation Engineer (Stress-Test & Realism Audit)
```
