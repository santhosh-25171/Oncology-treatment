# STAGE 5 ROLE REPORT 3: GENAI SCENARIO GENERATOR ENGINEER
## Personalized Precision Oncology — Generative Scenario Pipeline & Synthetic Edge-Case Synthesis

```
======================================================================================================
ROLE:               Stage 5 GenAI Scenario Generator Engineer
MODULE:             stage5_genai/genai/
PRIMARY MISSION:    Synthesize exactly 20 evidence-constrained oncology edge-case profiles (EDGE_001-020)
                    targeting empirical blind spots, implementing a dual-mode generator with zero-cost
                    deterministic local fallback (seed 42), mandatory synthetic labeling, and schema enforcement.
VERIFICATION:       Tests: 19 / 19 Passed (100% Success Rate in 3.56s)
======================================================================================================
```

---

## 1. Executive Mission & Role Definition

The **Stage 5 GenAI Scenario Generator Engineer** is responsible for synthesizing structured, biologically plausible edge-case scenarios designed specifically to stress test downstream clinical decision agents.

In computational oncology, AI treatment-recommendation engines frequently encounter out-of-distribution presentations. The primary mandate of this role is to:
1. Consume the historical baseline distributions (`data_engineering/processed/`) and empirical blind spots (`eda_prompteng/reports/blind_spot_report.json`).
2. Implement a **dual-mode architecture**: an OpenAI-compatible LLM generator alongside a deterministic, local, seed-controlled template fallback.
3. Enforce **mandatory synthetic labeling** (`synthetic: true`), rejecting any case missing this safety flag.
4. Guarantee **zero-cost reproducibility** via fixed random seeds (`random_seed: 42`).
5. Synthesize **exactly 20 diverse edge cases** (`EDGE_001` through `EDGE_020`) targeting empirical blind spots.
6. Validate all scenarios against a strict 13-attribute JSON schema (`schemas/synthetic_patient_schema.json`).

### Strict Role Boundary:
* **Research & stress-testing scenarios only**. Never represented as real historical patients or clinical trials.
* **NO** unconstrained hallucinations. Every alteration is labeled as `historically_observed` or `synthetic_combination_not_observed_in_reference`.
* **NO** modification of upstream Data Engineering reference baselines.

---

## 2. Dual-Mode Generation Architecture & Graceful Fallback

The generator features an enterprise-grade dual-mode architecture:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             DUAL-MODE GENERATION ARCHITECTURE                                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     ScenarioGenerator Facade     │
                              └────────────────┬─────────────────┘
                                               │
                     ┌─────────────────────────┴─────────────────────────┐
                     │                                                   │
                     ▼                                                   ▼
       ┌───────────────────────────┐                       ┌───────────────────────────┐
       │   1. Primary Mode (LLM)   │                       │ 2. Fallback Mode (Template)│
       │ • OpenAI-compatible API   │   Fallback Trigger:   │ • Pure local execution    │
       │ • Config via env vars     │ ────────────────────► │ • random_seed: 42         │
       │ • Structured JSON mode    │   No key / network err│ • Bit-for-bit reproducible│
       │ • output_validation_prompt│                       │ • Zero API cost           │
       └───────────────────────────┘                       └───────────────────────────┘
```

1. **Primary LLM Mode (`llm`)**: Calls OpenAI-compatible endpoints with temperature 0.2 and structured JSON formatting.
2. **Deterministic Fallback Mode (`template`)**: Automatically activated when API keys or network connections are unavailable. Uses `TemplateGenerator` with fixed pseudorandom seeds (`random_seed: 42`), ensuring complete reproducibility without cloud dependencies.
3. **Audit Transparency**: The generation method and fallback reasons are recorded in `provenance.generation_method`.

---

## 3. Scenario Schema Specification (13 Attributes)

Every scenario in `scenarios/synthetic_edge_cases.jsonl` complies with `schemas/synthetic_patient_schema.json` across 13 required attributes:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                SYNTHETIC PATIENT SCHEMA FIELDS                                   │
├────────────────────────────┬─────────────┬───────────────────────────────────────────────────────┤
│ Attribute Key              │ Data Type   │ Purpose & Clinical Requirement                        │
├────────────────────────────┼─────────────┼───────────────────────────────────────────────────────┤
│ 1. scenario_id             │ String      │ Unique identifier (EDGE_001 to EDGE_020).             │
│ 2. synthetic               │ Boolean     │ MANDATORY: Strictly true for all synthetic cases.     │
│ 3. generation_method       │ String      │ "template" or "llm".                                  │
│ 4. scenario_category       │ String      │ Category (e.g. on_target_resistance, compound_res).   │
│ 5. target_blind_spot       │ Object      │ Contains blind_spot_id (BS001-BS024), category, etc.  │
│ 6. patient_context         │ Object      │ Age group, sex, cancer type, histology, stage.        │
│ 7. genomic_profile         │ Object      │ Primary alterations, co-occurrences, and resistance.  │
│ 8. biomarkers              │ Object      │ TMB (mut/Mb), PD-L1 TPS %, ctDNA MAF %.               │
│ 9. clinical_context        │ Object      │ Disease status, progression history, prior lines.     │
│ 10. synthetic_assumptions  │ Array[str]  │ Explicitly marked hypothetical stress parameters.     │
│ 11. reference_evidence     │ Array[str]  │ Real baseline citations (TCGA, MSK, CIViC, NCCN).     │
│ 12. uncertainty            │ Object      │ Uncertainty level ("low", "moderate", "high") & reason│
│ 13. provenance             │ Object      │ Reference sources, timestamp, seed, generator version.│
└────────────────────────────┴─────────────┴───────────────────────────────────────────────────────┘
```

---

## 4. The 20 Synthetic Edge-Case Scenarios

The generator created exactly 20 diverse edge cases targeting the empirical blind spots:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               THE 20 SYNTHETIC EDGE CASES CATALOG                                │
├──────────┬────────────────────────┬────────────┬─────────────────────────────────────────────────┤
│ ID       │ Category               │ Blind Spot │ Primary Alteration & Mechanism                  │
├──────────┼────────────────────────┼────────────┼─────────────────────────────────────────────────┤
│ EDGE_001 │ rare_mutation          │ BS001      │ BRAF p.V600E (2.67% cohort prevalence)          │
│ EDGE_002 │ rare_mutation          │ BS002      │ MET p.D1010H juxtamembrane splice alteration    │
│ EDGE_003 │ rare_mutation          │ BS004      │ ERBB2 p.Y772_A775dup exon 20 insertion          │
│ EDGE_004 │ rare_mutation          │ BS005      │ RET KIF5B-RET fusion kinase inversion           │
│ EDGE_005 │ sparse_evidence        │ BS007      │ NTRK1 TPR-NTRK1 tissue-agnostic fusion          │
│ EDGE_006 │ sparse_evidence        │ BS010      │ NRG1 CD74-NRG1 chimeric transcript              │
│ EDGE_007 │ compound_mutation      │ BS011      │ KRAS G12C + STK11 + KEAP1 primary cold triad    │
│ EDGE_008 │ compound_resistance    │ BS012      │ KRAS p.G12C + p.Y99C switch II pocket mutation  │
│ EDGE_009 │ compound_resistance    │ BS013      │ ALK EML4-ALK + G1202R solvent-front hindrance   │
│ EDGE_010 │ on_target_resistance   │ BS014      │ EGFR L858R + T790M + C797S (in cis) tertiary    │
│ EDGE_011 │ bypass_resistance      │ BS015      │ EGFR Ex19del + MET Amplification (CN=14) bypass │
│ EDGE_012 │ multi_factor_resistance│ BS018      │ Adenocarcinoma to SCLC transformation + RB1/TP53│
│ EDGE_013 │ on_target_resistance   │ BS019      │ EGFR L858R + C797S + L718Q 4th-Gen TKI resistant│
│ EDGE_014 │ conflicting_evidence   │ BS023      │ TMB-High (16.4 mut/Mb) vs PD-L1 0% + STK11 loss │
│ EDGE_015 │ missing_biomarker      │ BS021      │ Stage I EGFR post-resection without ctDNA MRD   │
│ EDGE_016 │ unusual_stage_combo    │ BS024      │ Stage IIA EGFR with acquired C797S on adjuvant  │
│ EDGE_017 │ rare_combination       │ BS011      │ Concurrent EGFR Ex19del + KRAS G12V activation  │
│ EDGE_018 │ prior_treatment_res    │ BS017      │ ALK Compound G1202R + L1196M post 3 TKI lines   │
│ EDGE_019 │ bypass_resistance      │ BS020      │ ERBB2 Ex20 ins + SLFN11 loss + ABCB1 ADC failure│
│ EDGE_020 │ wildcard               │ BS018      │ The Wildcard: EGFR C797S + MET amp + SCLC switch│
└──────────┴────────────────────────┴────────────┴─────────────────────────────────────────────────┘
```

---

## 5. Scenario Validation & Rejection Engine

Implemented in `src/scenario_validator.py`, enforcing 6 strict safety rules:
* Rejects records where `synthetic` is missing or `false`.
* Rejects records with empty `synthetic_assumptions`.
* Rejects records with unmapped or invalid `blind_spot_id`.
* Rejects duplicate scenario IDs.
* Rejects missing provenance blocks (missing seed, timestamp, or sources).
* Verifies HGVS nomenclature conformance.

---

## 6. Unit Testing Suite & Verification

The GenAI Scenario Generator is verified by **19 automated unit tests**:
```
tests/test_reference_loader.py    4 passed (Baseline frequencies, co-occurrences)
tests/test_scenario_generation.py 4 passed (20 scenarios generated, unique IDs, seed reproducibility)
tests/test_schema.py              2 passed (Schema presence, 13 required attributes)
tests/test_provenance.py          3 passed (synthetic: true, provenance completeness)
tests/test_validation.py          6 passed (Rejection of invalid flags, missing assumptions)
================================== 19 passed in 3.56s ==================================
```
