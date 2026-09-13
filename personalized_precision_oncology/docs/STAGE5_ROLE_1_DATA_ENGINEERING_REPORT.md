# STAGE 5 ROLE REPORT 1: DATA ENGINEERING
## Personalized Precision Oncology — Historical Reference Baseline Assembly & Quality Assurance

```
======================================================================================================
ROLE:               Stage 5 Data Engineer
MODULE:             stage5_genai/data_engineering/
PRIMARY MISSION:    Assemble, harmonize, clean, validate, and structure authentic historical oncology
                    reference baselines (TCGA, MSK-IMPACT, SEER, CIViC, ClinVar, Trial Benchmarks)
                    WITHOUT generating synthetic patients, training LLMs, or pruning rare variants.
VERIFICATION:       Pipeline: PASSED | Tests: 30 / 30 Passed (100% Success Rate in 2.46s)
======================================================================================================
```

---

## 1. Executive Mission & Role Definition

In precision oncology AI systems, **generative models are only as safe and valid as the empirical evidence baselines that constrain them**. The cardinal failure mode of GenAI in oncology is unconstrained hallucination—inventing non-existent molecular mechanisms, misrepresenting mutation co-occurrences, or calculating treatment decisions on distorted demographic distributions.

The **Stage 5 Data Engineer** is strictly tasked with:
1. Ingesting immutable historical reference datasets from peer-reviewed public oncology registries.
2. Enforcing a strict **source-separation architecture** to prevent the dangerous merging of population-level epidemiology tables into individual patient sequencing records.
3. Establishing an auditable **missing-data taxonomy** with explicit clinical semantics.
4. Implementing a **rare mutation preservation policy (anti-pruning)** to protect vital drug-resistance signals (<3% frequency) from standard statistical outlier deletion.
5. Normalizing nomenclature to international HGVS protein standards (`p.`).
6. Exporting verified machine-readable GenAI reference baseline blocks (`genai_reference_baseline.jsonl`) and baseline distributions to serve as empirical priors for downstream generative modules.

### Strict Role Boundary (What Data Engineering Does NOT Do):
* **NO** Large Language Model (LLM) or Small Language Model (SLM) prompting or training.
* **NO** Synthetic patient case generation or artificial data creation.
* **NO** Clinical treatment decision scoring, drug ranking, or prescriptive medical advice.
* **NO** RAG vector database indexing or open-ended embedding generation.

---

## 2. Historical Oncology Data Sources & Lineage

The reference baseline was compiled exclusively from authentic, published, and peer-reviewed oncology cohorts:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             HISTORICAL DATA SOURCE REPOSITORY MATRIX                             │
├──────────────────────────────┬──────────────────┬────────────────┬───────────────────────────────┤
│ Source Name                  │ Provider / Repo  │ Data Domain    │ Clinical Scope & Dimensions   │
├──────────────────────────────┼──────────────────┼────────────────┼───────────────────────────────┤
│ 1. TCGA-LUAD & TCGA-LUSC     │ NCI GDC PanCancer│ Clinical &     │ 50 Primary surgical resections│
│    (The Cancer Genome Atlas) │ Atlas via cBio   │ Genomic        │ WES somatic mutations, stage, │
│                              │                  │                │ smoking pack-years, survival. │
├──────────────────────────────┼──────────────────┼────────────────┼───────────────────────────────┤
│ 2. MSK-IMPACT NSCLC Cohort   │ MSKCC / AACR     │ Genomic &      │ 25 Advanced/metastatic cases  │
│    (Targeted Panel 468 Genes)│ Project GENIE    │ Resistance     │ Targeted TKI lines, RECIST,   │
│                              │                  │                │ TMB, PD-L1 TPS, ctDNA MAF.    │
├──────────────────────────────┼──────────────────┼────────────────┼───────────────────────────────┤
│ 3. SEER 18/21 Cancer Review  │ National Cancer  │ Epidemiology & │ Population incidence, median  │
│    (Lung & Bronchus Registry)│ Institute (NCI)  │ Actuarial      │ age 70, sex ratios, 5-year    │
│                              │                  │                │ relative survival benchmarks. │
├──────────────────────────────┼──────────────────┼────────────────┼───────────────────────────────┤
│ 4. CIViC & NCBI ClinVar      │ CIViC / ClinVar  │ Variant &      │ Somatic clinical resistance,  │
│    Somatic Resistance DB     │ Public DB        │ Actionability  │ NCCN Category 1/2A evidence,  │
│                              │                  │                │ gatekeeper/bypass phenotype.  │
├──────────────────────────────┼──────────────────┼────────────────┼───────────────────────────────┤
│ 5. Landmark Phase III Trials │ KEYNOTE-024/158, │ Biomarker      │ Published clinical trial      │
│    (Prospective Trial Bench) │ FLAURA, CodeBreaK│ Reference      │ cutoff medians & IQR for      │
│                              │                  │                │ PD-L1, TMB, and ctDNA MAF %.  │
└──────────────────────────────┴──────────────────┴────────────────┴───────────────────────────────┘
```

---

## 3. Source-Separation Architecture

A frequent error in bioinformatics data pipelines is merging disparate registries into a single flat patient CSV. Population-level SEER tables, variant-level ClinVar evidence, and clinical trial medians do not map 1:1 to individual clinical encounters.

The Data Engineer enforces **Source-Separation**:
* **`cleaned_cohort.csv`**: Contains exclusively authentic individual patient records (N=75; 50 from TCGA, 25 from MSK-IMPACT).
* **Reference Blocks**: Non-patient evidence is cataloged as source-specific reference blocks with unique identifiers:
  - `REF-SEER-001` (Epidemiology)
  - `REF-CIVIC-CLINVAR-001` (Variant Evidence)
  - `REF-BIOMARKERS-001` (Biomarker Literature)
* Full lineage, DOIs, licenses, and record counts are maintained in `reports/source_metadata.csv`.

---

## 4. Five-Tier Missing-Data Taxonomy

In healthcare data engineering, naive mean/median imputation causes severe distribution drift. Stage 5 formalized a strict 5-tier semantic taxonomy:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                FIVE-TIER MISSING-DATA TAXONOMY                                   │
├──────────────────────────────┬──────────────────────────────┬────────────────────────────────────┤
│ Sentinel Label               │ Formal Scientific Definition │ Concrete Pipeline Example          │
├──────────────────────────────┼──────────────────────────────┼────────────────────────────────────┤
│ 1. null / np.nan             │ Value expected at patient    │ Survival months lost to follow-up  │
│                              │ level but missing in assay.  │ in retrospective records.          │
├──────────────────────────────┼──────────────────────────────┼────────────────────────────────────┤
│ 2. "not_available_in_source" │ Variable was never collected │ Plasma ctDNA MAF in historical     │
│                              │ by the original registry.    │ 2012-2016 TCGA surgical exomes.    │
├──────────────────────────────┼──────────────────────────────┼────────────────────────────────────┤
│ 3. "not_observed_in_         │ Alteration was not observed  │ Rare NTRK1 or NRG1 fusions in the  │
│    reference_baseline"       │ in cohort. DOES NOT mean     │ 75-case baseline.                  │
│                              │ biologically impossible!     │                                    │
├──────────────────────────────┼──────────────────────────────┼────────────────────────────────────┤
│ 4. "insufficient_evidence"   │ Published case report exists │ Single-case novel compound mutant  │
│                              │ but lacks consensus trials.  │ with uncharacterized TKI response. │
├──────────────────────────────┼──────────────────────────────┼────────────────────────────────────┤
│ 5. "source_limitation"       │ Source data structure cannot │ Calculating patient-level VAF from │
│                              │ support requested metric.    │ population aggregate tables.       │
└──────────────────────────────┴──────────────────────────────┴────────────────────────────────────┘
```

---

## 5. Cleaning, Deduplication & Structured Quarantine

Implemented in `src/cleaning.py`:
1. **Deduplication**: Audited 78 raw rows; identified and eliminated **3 duplicate rows** (100% duplicate resolution).
2. **Biological Range Sanitization**:
   - $18 \le \text{age} \le 115$ (0 violations in clean data)
   - $0 \le \text{PD-L1 TPS} \le 100\%$ (25 assayed, 0 violations)
   - $0 \le \text{ctDNA MAF} \le 100\%$ (24 assayed, 0 violations)
   - $0 \le \text{TMB} \le 500\text{ mut/Mb}$ (25 assayed, 0 violations)
3. **Structured Quarantine (Zero Silent Dropping)**:
   Invalid records are never silently deleted. Record `TCGA-ERR-9999` (containing an impossible `age: -5.0` and invalid `stage: Stage X`) was quarantined into `reports/quarantine_report.json` with execution timestamp, failing validation rule, and column-level values.
4. **Nomenclature Standardization**: Enforced HGVS `p.` prefixing across all variants (e.g., `p.L858R`, `p.G12C`, `p.T790M`).
5. **Final Cleaned Patient Cohort**: Exactly **75 verified, non-duplicated patient records** saved to `processed/cleaned_cohort.csv`.

---

## 6. Rare Mutation Preservation Policy (Anti-Pruning)

Standard outlier filtering frequently strips rare genomic variants (<3% frequency), falsely sanitizing the dataset. In oncology, rare variants are the primary determinants of drug sensitivity and secondary resistance:

* **Protected Whitelist (`PROTECTED_RARE_VARIANTS`)**: Whitelisted 36 actionable oncogenic and resistance mutations.
* **Retention Audit**: **31 rare variants retained (86.11% preservation rate)**.
* **Preserved Alterations Sample**:
  - *EGFR*: `p.C797S`, `p.T790M`, `p.L718Q`, `p.A763_Y764insFQEA`
  - *KRAS*: `p.Y99C`, `p.G12D`, `p.Q61H`, `p.G12A`
  - *ALK*: `p.G1202R`, `p.I1171N`, `EML4-ALK`
  - *ROS1*: `p.G2032R`, `CD74-ROS1`
  - *MET*: `p.D1010H`, `p.D1010Y`
  - *ERBB2*: `p.Y772_A775dup`
* **Zero Fabricated Mutations**: No synthetic alterations were inserted into historical cohort tables.

---

## 7. Statistical Baseline Distributions

Extracted by `src/distributions.py` and saved as machine-readable JSON files:
* **`cancer_type_distribution.json`**: Lung Adenocarcinoma (LUAD, 77.33%) vs Lung Squamous Cell Carcinoma (LUSC, 22.67%).
* **`stage_distribution.json`**: Stage I (28.0%), Stage II (14.67%), Stage III (13.33%), Stage IV (44.0%).
* **`age_distribution.json`**: Mean: 66.88 | Median: 67.0 | IQR: 11.5 | Range: 50.0 to 82.0.
* **`sex_distribution.json`**: Male: 56.0% (n=42) | Female: 44.0% (n=33).
* **`mutation_frequency.json`**: *KRAS* (24.0%), *EGFR* (24.0%), *TP53* (14.67%), *ALK* (6.67%), *MET* (5.33%), *BRAF* (4.0%).
* **`mutation_cooccurrence.json`**: Empirically observed multi-driver pairs: *KRAS*+*STK11* (n=3), *KRAS*+*KEAP1* (n=2), *KRAS*+*STK11*+*KEAP1* (n=1), *EGFR*+*TP53* (n=4).
* **`biomarker_distributions.json`**: TMB median 7.0 mut/Mb, PD-L1 TPS median 15%, ctDNA MAF median 4.95%.
* **`treatment_resistance_distributions.json`**: Gatekeeper alterations, bypass pathways, and CIViC clinical evidence tiers.

---

## 8. Unit Testing Suite & Verification

The Data Engineering module is verified by **30 automated pytest tests**:
```
tests/test_cleaning.py              6 passed (Deduplication, invalid ranges, rare variants)
tests/test_validation.py            6 passed (Biological boundaries, schemas, categories)
tests/test_distributions.py         6 passed (Frequencies, co-occurrences, intervals)
tests/test_genai_export.py          6 passed (JSONL schema, provenance, required keys)
tests/test_quality_and_provenance.py 6 passed (Metadata catalog, source-separation, quarantine)
==================================== 30 passed in 2.46s ====================================
```
