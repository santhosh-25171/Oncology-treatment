# Stage 5 — GenAI Generative Scenario Pipeline

**Project**: Personalized Precision Oncology  
**Repository**: [https://github.com/santhosh-25171/Oncology-treatment](https://github.com/santhosh-25171/Oncology-treatment)  
**Role**: Stage 5 GenAI Engineer (Generative Scenario Pipeline)  
**Directory**: `personalized_precision_oncology/stage5_genai/genai/`

---

## 1. Module Purpose & Strict Role Boundaries

The **Generative Scenario Pipeline** module consumes historical baseline reference datasets (`data_engineering/processed/`) and genomic blind spots (`eda_prompteng/reports/`) to synthesize realistic oncology edge-case profiles for downstream AI stress testing and robustness benchmarking.

### Non-Negotiable Research Mandates:
- **Synthetic Research Scenarios Only**: All records are explicitly labeled with `"synthetic": true`. They must **NEVER** be represented as real historical patients, actual clinical trials, or validated clinical treatment plans.
- **Evidence-Constrained Biological Plausibility**: Does not claim LLM validation proves biology; uses verifiable empirical constraints and marks unsupported combinations with explicit uncertainty.
- **Three-Tier Evidence Separation**: Rigorously separates historical baseline evidence (`reference_evidence`) from stress-test hypotheses (`synthetic_assumptions`) and catalogs scientific `uncertainty`.
- **Zero-Cost Local Execution**: Features a robust, deterministic, seed-controlled template fallback generator that runs locally with zero external API dependencies.

---

## 2. Directory Structure

```
stage5_genai/genai/
├── config/
│   └── generation_config.json              # Generation configuration (mode, seed, temperature, tokens)
├── schemas/
│   ├── synthetic_patient_schema.json       # JSON Schema validating 13 top-level scenario attributes
│   └── generation_config_schema.json       # JSON Schema validating runtime config
├── prompts/
│   ├── system_prompt.txt                   # LLM role boundaries, synthetic mandate & safety rules
│   ├── scenario_generation_prompt.txt      # Structured generation prompt with placeholders
│   └── output_validation_prompt.txt        # LLM self-audit / consistency check prompt
├── src/
│   ├── __init__.py
│   ├── reference_loader.py                 # Read-only loader for historical reference baselines
│   ├── blind_spot_loader.py                # Prioritizer & loader for EDA blind spots
│   ├── prompt_loader.py                    # Placeholder injection & prompt formatting
│   ├── scenario_validator.py               # Comprehensive schema & semantic consistency validator
│   ├── provenance.py                       # Audit trail, source tracking, timestamps, & seeds
│   └── pipeline.py                         # Master generation & validation pipeline orchestrator
├── generators/
│   ├── __init__.py
│   ├── mutation_sampler.py                 # Constrained frequency & co-occurrence sampling
│   ├── template_generator.py               # Deterministic fallback generator (20 edge cases)
│   ├── llm_generator.py                    # OpenAI-compatible LLM caller with graceful fallback
│   └── scenario_generator.py               # Unified facade routing between LLM and Template
├── scenarios/
│   ├── synthetic_edge_cases.jsonl          # Exactly 20 validated synthetic edge-case records
│   ├── generation_metadata.json            # Pipeline execution metadata & parameters
│   └── generation_summary.json             # Categorical & blind spot coverage summary
├── tests/
│   ├── __init__.py
│   ├── test_reference_loader.py            # Baseline loading integrity tests
│   ├── test_scenario_generation.py         # 20-scenario generation & seed reproducibility tests
│   ├── test_schema.py                      # JSON Schema compliance tests
│   ├── test_provenance.py                  # Provenance & synthetic label tests
│   └── test_validation.py                  # Malformed / unlabelled record rejection tests
├── .env.example                            # Safe environment variable configuration template
└── README.md                               # Technical documentation
```

---

## 3. The 20 Synthetic Edge-Case Scenarios

The generator constructed **exactly 20 diverse synthetic edge-case scenarios**, each targeting an empirical blind spot identified during Stage 5 EDA:

| Scenario ID | Category | Target Blind Spot | Primary Alteration & Mechanism | Clinical Dilemma / Stress Dimension |
| :--- | :--- | :--- | :--- | :--- |
| **`EDGE_001`** | `rare_mutation` | `BS001` (*BRAF* V600E) | *BRAF* p.V600E (2.67% cohort prevalence) | Monotherapy vs combination BRAF/MEK targeting in NSCLC. |
| **`EDGE_002`** | `rare_mutation` | `BS002` (*MET* Exon 14) | *MET* p.D1010H juxtamembrane splice | Sensitivity to selective MET TKIs (capmatinib/tepotinib). |
| **`EDGE_003`** | `rare_mutation` | `BS004` (*ERBB2* Exon 20) | *ERBB2* p.Y772_A775dup insertion | Limited standard TKI efficacy vs HER2 ADC (T-DXd). |
| **`EDGE_004`** | `rare_mutation` | `BS005` (*RET* Fusion) | *RET* KIF5B-RET fusion | CNS-penetrant RET selective inhibitor selection. |
| **`EDGE_005`** | `sparse_evidence` | `BS007` (*NTRK1* Fusion) | *NTRK1* TPR-NTRK1 fusion | Pan-cancer tissue-agnostic TRK inhibitor under baseline data gap. |
| **`EDGE_006`** | `sparse_evidence` | `BS010` (*NRG1* Fusion) | *NRG1* CD74-NRG1 fusion transcript | HER3/HER2 bispecific antibody (zenocutuzumab) therapeutic rationale. |
| **`EDGE_007`** | `compound_mutation` | `BS011` (*KRAS*+*STK11*+*KEAP1*) | *KRAS* G12C + *STK11* + *KEAP1* triad | Extreme primary immunoresistance despite high neoantigen load. |
| **`EDGE_008`** | `compound_resistance` | `BS012` (*KRAS* G12C + Y99C) | *KRAS* p.G12C + p.Y99C switch II mutation | Acquired covalent binding disruption under sotorasib/adagrasib. |
| **`EDGE_009`** | `compound_resistance` | `BS013` (*ALK* EML4-ALK + G1202R) | *ALK* G1202R solvent-front mutation | Refractoriness to 1st/2nd-gen TKIs; 3rd-gen lorlatinib rescue. |
| **`EDGE_010`** | `on_target_resistance` | `BS014` (*EGFR* C797S) | *EGFR* L858R + T790M + C797S (in cis) | Tertiary covalent bond disruption refractory to all approved TKIs. |
| **`EDGE_011`** | `bypass_resistance` | `BS015` (MET Amplification) | *EGFR* Ex19del + *MET* Amplification (CN=14) | Bypass activation without secondary EGFR mutation under osimertinib. |
| **`EDGE_012`** | `multi_factor_resistance`| `BS018` (SCLC Switch) | Lineage plasticity + *RB1*/*TP53* loss | Complete phenotypic shift from adenocarcinoma to neuroendocrine SCLC. |
| **`EDGE_013`** | `on_target_resistance` | `BS019` (4th-Gen Resistance) | *EGFR* L858R + C797S + L718Q | ATP-binding cleft mutation resisting investigational 4th-gen TKIs. |
| **`EDGE_014`** | `conflicting_evidence` | `BS023` (TMB vs PD-L1) | TMB-High (16.4 mut/Mb) vs PD-L1 TPS 0% | Conflicting signals: neoantigen burden vs STK11 cold microenvironment. |
| **`EDGE_015`** | `missing_biomarker` | `BS021` (ctDNA in Stage I) | Early-stage *EGFR* NSCLC post-R0 resection | Adjuvant targeted therapy decision without post-op ctDNA MRD baseline. |
| **`EDGE_016`** | `unusual_stage_combo` | `BS024` (Adjuvant Resistance) | Stage IIA *EGFR* with acquired C797S | Adjuvant targeted resistance occurring before metastatic dissemination. |
| **`EDGE_017`** | `rare_combination` | `BS011` (*EGFR* + *KRAS*) | Concurrent *EGFR* Ex19del + *KRAS* G12V | Violation of mutual exclusivity; intrinsic downstream TKI resistance. |
| **`EDGE_018`** | `prior_treatment_res` | `BS017` (Multi-line ALK) | Compound *ALK* G1202R + L1196M | Progression after crizotinib, alectinib, and lorlatinib. |
| **`EDGE_019`** | `bypass_resistance` | `BS020` (ADC Payload Failure) | *ERBB2* Exon 20 ins + SLFN11 loss + ABCB1 | Dual target antigen downregulation and drug efflux pump upregulation. |
| **`EDGE_020`** | `wildcard` | `BS018` (The Wildcard Challenge) | *EGFR* L858R + C797S + MET amp + SCLC switch | Fulminant biphasic progression: concurrent tertiary TKI resistance, bypass RTK, neuroendocrine transformation, and discordant immunogenomics. |

---

## 4. Scenario Structure & Three-Tier Evidence Separation

Every generated scenario in `scenarios/synthetic_edge_cases.jsonl` conforms strictly to `schemas/synthetic_patient_schema.json`:

```json
{
  "scenario_id": "EDGE_010",
  "synthetic": true,
  "generation_method": "template",
  "scenario_category": "on_target_resistance",
  "target_blind_spot": {
    "blind_spot_id": "BS014",
    "category": "resistance_mechanism",
    "coverage_status": "sparse",
    "reason": "Resistance pattern EGFR C797S Tertiary Resistance audited in MSK-IMPACT NSCLC baseline."
  },
  "patient_context": {
    "age_group": "65-69",
    "sex": "Female",
    "cancer_type": "NSCLC",
    "histology": "Lung Adenocarcinoma",
    "stage": "Stage IV",
    "prior_treatment_context": "Erlotinib (18 mo), Osimertinib 2nd-line (14 mo)"
  },
  "genomic_profile": {
    "alterations": [
      { "gene": "EGFR", "variant": "p.L858R", "alteration_type": "Exon 21 Activating", "status": "historically_observed" },
      { "gene": "EGFR", "variant": "p.T790M", "alteration_type": "Exon 20 Gatekeeper", "status": "historically_observed" }
    ],
    "cooccurring_alterations": [],
    "resistance_related_features": [
      { "gene": "EGFR", "variant": "p.C797S (in cis)", "alteration_type": "Exon 20 Covalent Disruption", "status": "synthetic_combination_not_observed_in_reference" }
    ]
  },
  "biomarkers": { "tmb": 5.4, "tmb_status": "low", "pdl1_tps": 0, "ctdna_maf_percent": 11.2 },
  "clinical_context": {
    "disease_status": "Acquired Progression",
    "progression_context": "Multifocal pulmonary progression and pericardial effusion."
  },
  "synthetic_assumptions": [
    "Modeled cis-allelic configuration of C797S with T790M, rendering tumor completely refractory to all approved 1st, 2nd, and 3rd gen TKIs."
  ],
  "reference_evidence": [
    "EGFR C797S tertiary resistance is documented in MSK-IMPACT resistance distributions (1 cohort instance) and CIViC registry."
  ],
  "uncertainty": {
    "level": "high",
    "reason": "Cis-C797S/T790M is completely resistant to osimertinib; requires 4th-gen allosteric inhibitors in clinical trials."
  },
  "provenance": {
    "reference_sources": [
      "TCGA PanCancer Atlas (GDC)",
      "MSK-IMPACT Targeted Sequencing (cBioPortal)",
      "CIViC / ClinVar Curated Registries",
      "SEER Oncology Benchmarks"
    ],
    "blind_spot_source": "Stage 5 EDA / blind_spot_report.json",
    "generation_method": "template",
    "generation_timestamp": "2026-09-11T17:13:19.686414+00:00",
    "random_seed": 42
  }
}
```

---

## 5. Execution Modes & Graceful Fallback

### Dual Mode Architecture
1. **Primary Mode (`llm`)**:
   - Uses an OpenAI-compatible API configured through environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`).
   - If credentials are not set or the network is unavailable, it **automatically falls back to template mode** and transparently sets `"fallback_reason": "LLM configuration unavailable"`.
   - Never commits API keys or credentials to the repository.
2. **Fallback Mode (`template`)**:
   - Executes deterministically with fixed random seeds (`random_seed: 42`).
   - Generates 100% reproducible, schema-compliant records locally without external network access.

---

## 6. Verification & Automated Test Suite

The test suite in `tests/` features **19 automated unit tests** across 5 test modules:
- `tests/test_reference_loader.py` (4 tests): Verifies read-only loading of historical frequencies, co-occurrences, and baseline blocks.
- `tests/test_scenario_generation.py` (4 tests): Verifies generation count (20), scenario ID uniqueness, deterministic seed reproducibility, and LLM-to-template fallback.
- `tests/test_schema.py` (2 tests): Verifies presence of schema and validates all 20 scenarios against `synthetic_patient_schema.json`.
- `tests/test_provenance.py` (3 tests): Verifies mandatory `synthetic: true`, complete provenance blocks, and evidence/assumption isolation.
- `tests/test_validation.py` (6 tests): Verifies `ScenarioValidator` acceptance of valid records and rejection of missing synthetic flags, missing blind spots, empty assumptions, missing provenance, and duplicate IDs.

**Test Run Output**:
```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
collected 19 items

test_provenance.py::test_synthetic_flag_true PASSED                       [  5%]
test_provenance.py::test_provenance_structure PASSED                      [ 10%]
test_provenance.py::test_evidence_and_assumption_separation PASSED        [ 15%]
test_reference_loader.py::test_reference_loader_initialization PASSED    [ 21%]
test_reference_loader.py::test_reference_loader_gene_frequency PASSED    [ 26%]
test_reference_loader.py::test_reference_loader_cooccurrences PASSED      [ 31%]
test_reference_loader.py::test_reference_loader_summary_text PASSED      [ 36%]
test_scenario_generation.py::test_template_generator_count PASSED        [ 42%]
test_scenario_generation.py::test_scenario_unique_ids PASSED             [ 47%]
test_scenario_generation.py::test_reproducibility_with_same_seed PASSED   [ 52%]
test_scenario_generation.py::test_llm_generator_graceful_fallback PASSED  [ 57%]
test_schema.py::test_schema_file_exists PASSED                            [ 63%]
test_schema.py::test_scenarios_comply_with_schema PASSED                  [ 68%]
test_validation.py::test_validator_accepts_valid_scenario PASSED          [ 73%]
test_validation.py::test_validator_rejects_missing_synthetic_flag PASSED [ 78%]
test_validation.py::test_validator_rejects_missing_blind_spot PASSED     [ 84%]
test_validation.py::test_validator_rejects_empty_assumptions PASSED       [ 89%]
test_validation.py::test_validator_rejects_missing_provenance PASSED      [ 94%]
test_validation.py::test_validator_detects_duplicates PASSED             [100%]

============================= 19 passed in 0.75s ==============================
```

---

## 7. Downstream Handoff to Evaluation

The generated output files in `scenarios/`:
- `synthetic_edge_cases.jsonl`: 20 standardized JSONL edge cases.
- `generation_metadata.json`: Full execution provenance.
- `generation_summary.json`: Category distribution and blind spot coverage.

These outputs are ready for consumption by the downstream **Stage 5 Evaluation Engine** to benchmark AI treatment recommendations, diagnostic reasoning, and edge-case stress-testing metrics.
