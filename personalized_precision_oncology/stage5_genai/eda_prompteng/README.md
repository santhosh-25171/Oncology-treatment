# Stage 5 — Genomic EDA & Prompt Engineering Module

**Project**: Personalized Precision Oncology  
**Repository**: [https://github.com/santhosh-25171/Oncology-treatment](https://github.com/santhosh-25171/Oncology-treatment)  
**Role**: Exploratory Data Analysis & Prompt Engineering  
**Directory**: `personalized_precision_oncology/stage5_genai/eda_prompteng/`

---

## 1. Module Overview & Role Boundaries

The **EDA & Prompt Engineering** module sits directly between the verified **Data Engineering** reference baseline and downstream **GenAI Synthetic Case Generation**.

### Strict Architectural Boundaries
- **IN SCOPE**:
  1. Rigorous Exploratory Data Analysis (EDA) of the 75-patient multi-source reference baseline.
  2. Data-driven detection and cataloging of genomic blind spots, rare variants (<3%), unrepresented co-occurrences, resistance gaps, and source limitations.
  3. Construction of specialized, safe prompt templates for extreme drug-resistance scenarios enforcing three-tier evidence separation (`[SUPPORTED EVIDENCE]`, `[SYNTHETIC ASSUMPTION]`, `[UNKNOWN / INSUFFICIENT EVIDENCE]`).
  4. Non-fabrication integrity, source provenance tracking (TCGA vs. MSK-IMPACT vs. SEER vs. CIViC), and feature availability auditing.
- **OUT OF SCOPE** (Deferred to downstream GenAI and clinical agent modules):
  - Ingestion/pipeline re-creation (consumed read-only from `data_engineering/processed/`).
  - Synthetic patient case record generation.
  - LLM fine-tuning, training, or vector database embedding.
  - Autonomous treatment recommendations, dosage prescriptions, or clinical decision-making.

---

## 2. Directory Structure

```
stage5_genai/eda_prompteng/
├── schemas/
│   ├── blind_spot_schema.json          # JSON Schema validating blind spots taxonomy
│   └── prompt_schema.json              # JSON Schema validating prompt coverage
├── eda/
│   ├── __init__.py
│   ├── genomic_coverage.py             # Feature availability & stage distribution audit
│   ├── mutation_analysis.py            # VAF audit, rare variants (<3%), check-target audit
│   ├── cooccurrence_analysis.py        # Co-occurrence evaluation & multi-driver matrix
│   ├── resistance_analysis.py          # On-target, bypass RTK, solvent-front, lineage switch
│   ├── biomarker_analysis.py           # TMB, PD-L1, ctDNA MAF, inflammatory marker audit
│   ├── blind_spot_detection.py         # Orchestrator classifying 24 blind spots
│   ├── visualize.py                    # Matplotlib-only visualization generator
│   └── prompt_coverage.py              # Prompt coverage auditor & schema validator
├── visualizations/
│   ├── mutation_frequency.png          # Gene-level frequency distribution (pure Matplotlib)
│   ├── mutation_cooccurrence.png       # 8x8 Driver mutation co-occurrence heatmap
│   ├── stage_genomic_coverage.png      # Source-stratified stage representation (TCGA vs MSK)
│   └── resistance_coverage.png         # Targeted resistance status breakdown
├── prompts/
│   ├── system_prompt.txt               # Role boundaries & three-tier evidence governance
│   ├── extreme_resistance_prompt.txt   # Tertiary C797S + MET bypass resistance template
│   ├── compound_mutation_prompt.txt    # Multi-driver (KRAS+STK11+KEAP1) co-occurrence template
│   ├── conflicting_evidence_prompt.txt # Discordant TMB-High vs PD-L1 0% + STK11 loss template
│   ├── sparse_evidence_prompt.txt      # Ultra-rare / unobserved pan-cancer fusion template
│   └── wildcard_prompt.txt             # Complex multi-factorial lineage switch (SCLC) template
├── reports/
│   ├── genomic_eda_report.json         # Comprehensive empirical EDA findings
│   ├── blind_spot_report.json          # 24 cataloged blind spots with justification
│   └── prompt_coverage_report.json     # 5 stress prompts linked to blind spot IDs
├── tests/
│   ├── __init__.py
│   ├── test_eda.py                     # EDA computation & non-fabrication tests
│   ├── test_blind_spots.py             # Blind spot taxonomy & schema compliance tests
│   └── test_prompts.py                 # Prompt placeholders & evidence tier tests
└── README.md
```

---

## 3. Data Inputs & Provenance

This module reads exclusively from the validated historical baseline outputs produced in `data_engineering/processed/`:
1. `cleaned_cohort.csv`: 75 harmonized patient records from TCGA PanCancer Atlas (N=50, primary resections) and MSK-IMPACT (N=25, advanced/metastatic).
2. `mutation_frequency.json`: 15 gene-level and 48 variant-level historical somatic frequencies.
3. `survival_distribution.json`: Overall survival benchmarks stratified by stage and histology.
4. `recurrence_risk.json`: Recurrence risk benchmarks derived from SEER historical registries.
5. `treatment_response.json`: Standard-of-care clinical trial response rates and resistance alterations.

---

## 4. Empirical Genomic Findings & Baseline Distributions

| Metric / Dimension | TCGA PanCancer (N=50) | MSK-IMPACT (N=25) | Unified Baseline (N=75) |
| :--- | :--- | :--- | :--- |
| **Cohort Type** | Primary surgical resections | Advanced / metastatic | Harmonized NSCLC |
| **Assay Modality** | Whole Exome Sequencing (WES) | Deep Targeted Panel (468 genes) | Multi-panel |
| **Top Drivers** | *KRAS* (24.0%), *EGFR* (22.7%), *TP53* (17.3%) | *EGFR* (40.0%), *KRAS* (28.0%) | *KRAS*, *EGFR*, *TP53*, *PIK3CA* |
| **Stage Distribution** | Stage I (42.0%), Stage II (22.0%), Stage III (20.0%), Stage IV (16.0%) | Stage IV (100.0%) | Stage I-III (56.0%), Stage IV (44.0%) |
| **Median TMB** | Unassayed in unified matrix | 7.0 mut/Mb (range: 2.8 - 15.6) | 7.0 mut/Mb (N=25 assayed) |
| **PD-L1 Status** | Unassayed in surgical cohort | 60% Low (1-49%), 28% High (>=50%) | 25 assayed |
| **Plasma ctDNA** | Unassayed (source limitation) | Median MAF 4.95% (range: 1.2 - 14.5%) | 24 assayed |

---

## 5. Feature Availability Auditing & Source Limitations

Before performing statistical evaluations, variables were audited for direct dataset presence:
- **Variant Allele Fraction (VAF)**: Not available across unified cohort records. Cataloged as:
  ```json
  {
    "status": "source_limitation",
    "reason": "Variable 'variant_allele_fraction' not available in unified patient cohort reference baseline"
  }
  ```
- **Early-Stage Plasma ctDNA MAF**: TCGA surgical resections (2010–2016 era) did not collect longitudinal plasma ctDNA. Absence is noted as an assay era limitation, not biological absence.
- **Peripheral Inflammatory Markers (NLR, CRP, LDH)**: Documented in clinical trial literature benchmarks but uncollected at the patient-level in genomic sequencing cohorts.

---

## 6. Blind-Spot Taxonomy & Classification

All identified data gaps and candidate patterns are strictly categorized using the seven-tier taxonomy:

| Taxonomy Classification | Formal Definition | Reference Baseline Count |
| :--- | :--- | :---: |
| `well_represented` | Observed with adequate representation (>=3.0% cohort prevalence) | 12 |
| `rare` | Observed with low representation (<3.0% cohort prevalence or <=2 instances) | 9 |
| `sparse` | Documented in literature/evidence registries but observed only as singletons (n=1) | 6 |
| `not_observed` | Evaluated against baseline and unobserved (0 records); labeled `not_observed_in_reference_baseline` | 6 |
| `insufficient_evidence` | Published data lacks statistical consensus | 0 |
| `conflicting_evidence` | Divergent biomarker signals documented across Phase III clinical trials | 1 |
| `source_limitation` | Variable unassayed or unavailable due to historical assay era or panel design | 2 |

> **Unobserved Pattern Rule**: Unobserved patterns (e.g., *NTRK1/2/3* fusions, *NRG1* fusions, ADC payload resistance) are never claimed as "biologically absent". They are explicitly cataloged as `"not_observed_in_reference_baseline"` with the note: *"unobserved in dataset does not imply biological impossibility"*.

---

## 7. Catalog of Identified Blind Spots (Summary)

The engine cataloged **24 distinct genomic blind spots** in `reports/blind_spot_report.json`:
- **BS001–BS006 (Rare Mutations)**: *BRAF* p.V600E (2.67%), *MET* p.D1010H (2.67%), *TP53* p.R175H (2.67%), *ERBB2* p.Y772_A775dup (2.67%), *RET* KIF5B-RET (2.67%), *FGFR1* Amp (2.67%).
- **BS007–BS010 (Unobserved Pan-Cancer Drivers)**: *NTRK1*, *NTRK2*, *NTRK3*, and *NRG1* fusions (`not_observed`).
- **BS011–BS013 (Compound Alterations)**: *KRAS*+*STK11*+*KEAP1*, *KRAS* G12C + Y99C switch II mutation, *ALK* EML4-ALK + G1202R solvent front.
- **BS014–BS020 (Acquired Resistance)**: *EGFR* C797S tertiary resistance, *MET* bypass amplification, *KRAS* Y99C, *ALK* G1202R, Histological SCLC transformation, 4th-Gen EGFR allosteric resistance, and ADC internalization failure (`not_observed`).
- **BS021–BS022 (Source Limitations)**: Early-stage plasma ctDNA MAF and patient-level inflammatory markers (NLR/CRP/LDH).
- **BS023 (Conflicting Evidence)**: TMB-High (>=10 mut/Mb) with PD-L1 TPS 0% and *STK11* loss.
- **BS024 (Stage-Specific Gap)**: Acquired TKI resistance alterations in early-stage (Stage I/II) NSCLC (`not_observed`).

---

## 8. Matplotlib Visualizations

Four publication-quality charts were generated using **native Matplotlib only** (`matplotlib.use("Agg")`, NO Seaborn):
1. `visualizations/mutation_frequency.png`: Bar chart of top 11 gene frequencies annotated with patient counts and cohort percentages.
2. `visualizations/mutation_cooccurrence.png`: 8x8 driver mutation co-occurrence matrix heatmap with exact cell annotations.
3. `visualizations/stage_genomic_coverage.png`: Grouped bar chart illustrating source stratification by AJCC stage (TCGA surgical vs MSK-IMPACT metastatic).
4. `visualizations/resistance_coverage.png`: Horizontal bar chart breaking down representation status for 7 major targeted resistance mechanisms.

---

## 9. Prompt Engineering & Evidence Governance

All prompt templates enforce the **Three-Tier Evidence Separation Mandate**:
1. `[SUPPORTED EVIDENCE]`: Grounded strictly in FDA labels, Phase III trials, NCCN guidelines, CIViC, or baseline distributions.
2. `[SYNTHETIC ASSUMPTION]`: Explicitly marked edge-case parameters introduced for algorithmic stress testing.
3. `[UNKNOWN / INSUFFICIENT EVIDENCE]`: Explicit identification of dataset blind spots, unrepresented combinations, or absent randomized trial data.

### Stress-Testing Scenario Prompts:
- `prompts/system_prompt.txt`: Core governance mandate, safety boundaries, and three-tier rules.
- `prompts/extreme_resistance_prompt.txt` (Target: `BS014`): Tertiary *EGFR* C797S covalent disruption + *MET* bypass amplification under 3rd-Gen TKI.
- `prompts/compound_mutation_prompt.txt` (Target: `BS011`): Dual/triple driver co-occurrence (*KRAS* + *STK11* + *KEAP1*) violating mutual exclusivity.
- `prompts/conflicting_evidence_prompt.txt` (Target: `BS023`): Hypermutation (TMB >=10 mut/Mb) vs PD-L1 TPS 0% + *STK11* loss immune exclusion.
- `prompts/sparse_evidence_prompt.txt` (Target: `BS007`): Pan-cancer unobserved fusion (*NTRK1/2/3*, *NRG1*) handling without biological fabrication.
- `prompts/wildcard_prompt.txt` (Target: `BS018`): Lineage plasticity (adenocarcinoma to SCLC phenotype transformation) with *RB1*/*TP53* double-null loss.

---

## 10. Prompt Coverage Validation

`reports/prompt_coverage_report.json` was generated and strictly validated against `schemas/prompt_schema.json`:
- Total Blind Spots Evaluated: 24
- Stress Scenarios Formulated: 5
- Covered Dimensions: Extreme resistance, compound drivers, conflicting evidence, unobserved fusions, and wildcard lineage switch.
- Schema Status: **PASSED (100% compliant)**.

---

## 11. Verification & Test Suite

The test suite in `tests/` contains **16 automated pytest unit tests**:
- `tests/test_blind_spots.py` (4 tests): Validates JSON schema compliance of `blind_spot_report.json`, taxonomy enforcement, unobserved phrasing integrity, and summary tally consistency.
- `tests/test_eda.py` (7 tests): Validates feature availability auditing, VAF limitation handling, rare variant filtering (<3%), candidate target evaluations, co-occurrence matrices, resistance audits, and biomarker calculations.
- `tests/test_prompts.py` (5 tests): Validates presence of all 6 prompt templates, required placeholders (`{{...}}`), three-tier evidence headers, `prompt_schema.json` compliance, and valid cross-references to detected blind spot IDs.

**Test Execution Result**:
```
============================= 16 passed in 6.45s ==============================
```

---

## 12. Clinical Safety & Research Disclaimer

> [!CAUTION]
> **Research & Stress-Testing Use Only**:  
> This software module and its associated prompt templates are designed exclusively for computational oncology research, AI agent benchmarking, and algorithmic robustness evaluation. They do not constitute medical advice, clinical decision support, or prescriptive recommendations. Direct dosage instructions or clinical administration decisions must never be derived from these research templates.
