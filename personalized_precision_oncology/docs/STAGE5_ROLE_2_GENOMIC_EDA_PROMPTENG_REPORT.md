# STAGE 5 ROLE REPORT 2: GENOMIC EDA & PROMPT ENGINEERING
## Personalized Precision Oncology — Blind-Spot Detection, Evidence Governance & Prompt Design

```
======================================================================================================
ROLE:               Stage 5 Genomic EDA & Prompt Engineer
MODULE:             stage5_genai/eda_prompteng/
PRIMARY MISSION:    Audit feature availability, identify data-driven empirical genomic blind spots (BS01-24),
                    formulate 3-tier evidence governance prompt templates, and generate publication-grade
                    native Matplotlib visualizations without clinical fabrication.
VERIFICATION:       Tests: 16 / 16 Passed (100% Success Rate in 6.45s)
======================================================================================================
```

---

## 1. Executive Mission & Role Definition

The **Stage 5 Genomic EDA & Prompt Engineer** bridges the verified historical reference baseline (Role 1) and synthetic case generation (Role 3). 

In oncology generative modeling, prompts written without empirical grounding produce generic, unhelpful cases or medically absurd combinations (such as proposing concurrent *EGFR* TKI and *ALK* TKI without evidence of dual activation). The primary mandate of this role is to:
1. Conduct an exhaustive exploratory data analysis (EDA) of the 75-patient multi-source reference baseline.
2. Systematically audit feature availability and catalog historical assay limitations (e.g. absent VAF or early-stage ctDNA).
3. Establish a rigorous **seven-tier blind-spot taxonomy** to classify gaps in representation.
4. Discover and document **24 distinct empirical blind spots** (`BS001` through `BS024`).
5. Design robust, parameterized prompt templates enforcing the **Three-Tier Evidence Separation Mandate** (`[SUPPORTED EVIDENCE]`, `[SYNTHETIC ASSUMPTION]`, `[UNKNOWN / INSUFFICIENT EVIDENCE]`).
6. Generate 4 publication-quality visualizations using pure Matplotlib.

### Strict Role Boundary:
* **Read-only** consumption of historical baseline data (`data_engineering/processed/`).
* **NO** synthetic patient generation or JSONL export.
* **NO** model fine-tuning, training, or vector database embedding.
* **NO** prescriptive medical dosing or clinical recommendations.

---

## 2. Feature Availability Auditing & Source Limitations

Before conducting downstream modeling, variables were audited across the 75 baseline cases:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                FEATURE AVAILABILITY & ASSAY AUDIT                                │
├──────────────────────────────┬───────────────┬─────────────────┬─────────────────────────────────┤
│ Feature Dimension            │ Status        │ Coverage %      │ Clinical Limitation Context     │
├──────────────────────────────┼───────────────┼─────────────────┼─────────────────────────────────┤
│ Age, Sex, Histology, Stage   │ Available     │ 75/75 (100.0%)  │ Complete clinical demographic.  │
│ Primary Driver Gene          │ Available     │ 75/75 (100.0%)  │ Uniformly called across panels. │
│ Smoking Pack-Years           │ Available     │ 50/75 (66.67%)  │ Available in TCGA surgical.     │
│ TMB (mut/Mb)                 │ Available     │ 25/75 (33.33%)  │ Assayed in MSK-IMPACT panel.    │
│ PD-L1 TPS %                  │ Available     │ 25/75 (33.33%)  │ Assayed in MSK-IMPACT cohort.   │
│ Plasma ctDNA MAF %           │ Available     │ 24/75 (32.00%)  │ Assayed in advanced MSK cohort. │
│ Variant Allele Fraction (VAF)│ Source Limit  │ 0/75 (0.00%)    │ Unreported across unified calls.│
│ Early-Stage ctDNA MRD        │ Source Limit  │ 0/50 (0.00%)    │ Assay era limitation (2012-16). │
│ Inflammatory Markers (NLR)   │ Trial Bench   │ 0/75 (0.00%)    │ Trial literature reference only.│
└──────────────────────────────┴───────────────┴─────────────────┴─────────────────────────────────┘
```

---

## 3. Seven-Tier Blind-Spot Taxonomy

To prevent subjective assertions, data gaps were classified under a formal 7-tier taxonomy:

1. **`well_represented`** ($\ge 3.0\%$ cohort prevalence): Primary drivers (*KRAS* G12C, *EGFR* L858R, *TP53*).
2. **`rare`** ($< 3.0\%$ prevalence, $\le 2$ cases): Low-frequency alterations (*BRAF* V600E, *MET* Exon 14, *ERBB2* Exon 20).
3. **`sparse`** ($n=1$ in cohort): Singletons documented in registries (*EGFR* C797S, *KRAS* Y99C, *ALK* G1202R).
4. **`not_observed`** (0 cases in cohort): Evaluated and unobserved (*NTRK1/2/3*, *NRG1* fusions).
   > **The Unobserved Pattern Rule**: Labeled `"not_observed_in_reference_baseline"`, explicitly documenting: *"unobserved in dataset does not imply biological impossibility."*
5. **`insufficient_evidence`**: Case reports without guideline consensus.
6. **`conflicting_evidence`**: Divergent trial signals (e.g. TMB-High vs PD-L1 0% + *STK11* loss).
7. **`source_limitation`**: Unassayed due to historical technology or panel design (e.g. VAF).

---

## 4. The 24 Empirical Genomic Blind Spots

From `reports/blind_spot_report.json`:

```
  BS001: BRAF p.V600E (2.67% prevalence, rare kinase activation)
  BS002: MET p.D1010H (2.67% prevalence, exon 14 juxtamembrane splice alteration)
  BS003: TP53 p.R175H (2.67% prevalence, structural DNA-binding contact mutation)
  BS004: ERBB2 p.Y772_A775dup (2.67% prevalence, exon 20 insertion refractory to standard TKIs)
  BS005: RET KIF5B-RET (2.67% prevalence, kinase domain inversion fusion)
  BS006: FGFR1 Amplification (2.67% prevalence, squamous cell receptor amplification)
  BS007: NTRK1 Fusions (0.0% in baseline; not_observed_in_reference_baseline)
  BS008: NTRK2 Fusions (0.0% in baseline; not_observed_in_reference_baseline)
  BS009: NTRK3 Fusions (0.0% in baseline; not_observed_in_reference_baseline)
  BS010: NRG1 Fusions (0.0% in baseline; not_observed_in_reference_baseline)
  BS011: KRAS + STK11 + KEAP1 Triad (1.33% prevalence, extreme immunoresistance co-occurrence)
  BS012: KRAS G12C + Y99C Switch II (Acquired covalent inhibitor binding resistance)
  BS013: ALK EML4-ALK + G1202R (Acquired solvent-front steric hindrance)
  BS014: EGFR C797S Tertiary Resistance (Acquired covalent bond disruption under osimertinib)
  BS015: MET Bypass Amplification (EGFR-independent RTK bypass signaling, CN=14)
  BS016: KRAS Q61H Secondary Mutation (GTPase active site locking)
  BS017: ALK Compound L1196M + G1202R (Dual gatekeeper + solvent-front resistance)
  BS018: Histological Transformation (Adenocarcinoma to Small Cell Neuroendocrine / SCLC switch)
  BS019: 4th-Gen EGFR Allosteric Resistance (EGFR L858R + C797S + L718Q triple mutant)
  BS020: ADC Internalization Failure (Target downregulation + efflux pump upregulation)
  BS021: Stage I ctDNA MRD Absence (Assay era limitation in historical surgical resections)
  BS022: Inflammatory Markers NLR/CRP (Unassayed in genomic sequencing cohorts)
  BS023: TMB-High (>=10 mut/Mb) with PD-L1 0% and STK11 Loss (Conflicting immunotherapy signals)
  BS024: Acquired TKI Resistance in Stage I/II (Unobserved pattern in early-stage surgical baselines)
```

---

## 5. Three-Tier Evidence Governance & Prompt Design

Every prompt template enforces three distinct structural tiers:
1. `[SUPPORTED EVIDENCE]`: Grounded in FDA approvals, NCCN guidelines, CIViC, and baseline frequencies.
2. `[SYNTHETIC ASSUMPTION]`: Hypothetical edge-case parameters introduced to stress test reasoning.
3. `[UNKNOWN / INSUFFICIENT EVIDENCE]`: Explicit catalog of data blind spots or unrepresented combinations.

### The 6 Standardized Prompt Files:
* **`system_prompt.txt`**: Governs role boundaries, non-hallucination, and evidence segregation.
* **`extreme_resistance_prompt.txt`**: C797S tertiary resistance and MET bypass under 3rd-Gen TKI (`BS014`).
* **`compound_mutation_prompt.txt`**: Triple driver (*KRAS*+*STK11*+*KEAP1*) primary immunoresistance (`BS011`).
* **`conflicting_evidence_prompt.txt`**: High TMB vs PD-L1 0% with *STK11* loss (`BS023`).
* **`sparse_evidence_prompt.txt`**: Unobserved pan-cancer fusions (*NTRK1/2/3*, *NRG1*) (`BS007`).
* **`wildcard_prompt.txt`**: Biphasic neuroendocrine lineage switch with *RB1*/*TP53* double-null loss (`BS018`).

---

## 6. Publication-Quality Visualizations

Generated using native Matplotlib (`matplotlib.use("Agg")`, zero Seaborn dependencies):
1. **`visualizations/mutation_frequency.png`**: Top 11 drivers with patient counts and cohort percentages.
2. **`visualizations/mutation_cooccurrence.png`**: 8x8 Driver co-occurrence heatmap with cell value annotations.
3. **`visualizations/stage_genomic_coverage.png`**: Grouped bar chart comparing surgical TCGA vs metastatic MSK stages.
4. **`visualizations/resistance_coverage.png`**: Horizontal representation status across 7 major targeted resistance mechanisms.

---

## 7. Unit Testing Suite & Verification

The Genomic EDA test suite features **16 automated pytest unit tests**:
```
tests/test_eda.py         7 passed (Feature availability, VAF handling, rare variants, co-occurrence)
tests/test_blind_spots.py 4 passed (Taxonomy compliance, unobserved phrasing, schema validation)
tests/test_prompts.py     5 passed (Placeholders, three-tier headers, schema conformance)
========================== 16 passed in 6.45s ==========================
```
