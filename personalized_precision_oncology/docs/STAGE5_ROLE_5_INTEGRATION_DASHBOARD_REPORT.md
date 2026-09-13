# STAGE 5 ROLE REPORT 5: INTEGRATION & DASHBOARD ENGINEER
## Personalized Precision Oncology — Contamination Protection, REST API & Interactive UI Dashboard

```
======================================================================================================
ROLE:               Stage 5 Integration & Dashboard Engineer
MODULE:             stage5_genai/integration/
PRIMARY MISSION:    Connect upstream synthetic scenarios to an interactive testing dashboard and FastAPI
                    REST service, enforce strict contamination protection shields (ID disjointness,
                    cohort inviolability), interactive seed-guided GenAI patient generation, realism/
                    discriminator telemetry analytics, append-only persistence, and SHA-256 change detection.
VERIFICATION:       Tests: 29 / 29 Passed (100% Success Rate in 18.45s)
======================================================================================================
```

---

## 1. Executive Mission & Role Definition

The **Stage 5 Integration & Dashboard Engineer** operationalizes the outputs of Data Engineering, EDA, GenAI Generation, and Evaluation. 

In clinical AI deployment, having static JSONL files is insufficient; clinical researchers and software testers require an intuitive, real-time interface to inspect edge cases, execute live audits, generate custom seed-guided synthetic patients on demand, and observe decision stresses. Crucially, the Integration Engineer must ensure that **synthetic research scenarios never contaminate historical clinical training sets**, and that interactive generations do not corrupt the baseline 20 benchmark test scenarios.

The primary mandate of this role is to:
1. Ingest generated scenarios safely with duplicate rejection and schema verification (`ScenarioLoader`).
2. Implement an impenetrable **Contamination Protection Shield** preventing synthetic records from mixing with real patient cohorts.
3. Transform raw scenario records into secure, badge-protected UI models (`ScenarioAdapter`).
4. Build high-performance **FastAPI REST endpoints** for scenario querying, live interactive generation, analytics, and single/batch audits (`src/api.py`).
5. Deliver a standalone **HTML5/Tailwind/JavaScript Web Dashboard** (`dashboard.html` / `src/dashboard_app.py`) featuring interactive patient generation and realism inspection cards.
6. Maintain an **append-only longitudinal evaluation run history** (`history/evaluation_history.jsonl`).
7. Implement automated **SHA-256 change detection** to seamlessly reload upstream scenario updates without modifying disk files.

### Strict Role Boundary:
* **DO NOT** generate synthetic patient cases without explicit user seeds or alter biological assumptions.
* **DO NOT** rewrite or alter the core evaluation rules of `ScenarioEvaluator`.
* **DO NOT** fabricate missing fields (unassayed values are surfaced as `"NOT EVALUATED"`).
* **DO NOT** merge synthetic test cases into the historical cohort.
* **DO NOT** corrupt the frozen 20 benchmark scenarios with dynamic interactive runs.

---

## 2. Contamination Protection Shield & Anti-Mixing Guarantees

The integration layer enforces 5 non-negotiable isolation rules verified by `tests/test_contamination_protection.py`:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            CONTAMINATION PROTECTION ARCHITECTURE                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
      ┌────────────────────────────────────────┴────────────────────────────────────────┐
      │                                                                                 │
      ▼                                                                                 ▼
┌───────────────────────────────────────────┐                     ┌───────────────────────────────────────────┐
│     REAL PATIENT COHORT (IMMUTABLE)       │                     │    SYNTHETIC RESEARCH CASES (ISOLATED)    │
│ • Source: TCGA & MSK-IMPACT               │                     │ • Source: GenAI Scenario Generator        │
│ • File: data_engineering/cleaned_cohort   │                     │ • File: genai/scenarios/synthetic_edge    │
│ • ID Pattern: TCGA-*, MSK-*               │                     │ • ID Pattern: EDGE_001 to EDGE_020, SYN-* │
│ • Use: Baseline statistical distributions │                     │ • Use: Algorithmic decision stress testing│
└───────────────────────────────────────────┘                     └───────────────────────────────────────────┘
                      │                                                                 │
                      └───────────────────────────────┬─────────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │   ISOLATION AIR GAP SHIELD  │
                                       │ • Zero Cross-File Writes    │
                                       │ • Strict ID Disjointness    │
                                       │ • Mandatory synthetic: true │
                                       │ • Adapter Warning Badges    │
                                       └─────────────────────────────┘
```

1. **Mandatory Boolean Flag**: Every scenario must have `synthetic: true`. Records with `synthetic: false` or missing flags are rejected immediately.
2. **Strict ID Disjointness**: Synthetic IDs (`EDGE_001` to `EDGE_020`, `SYN-XXXXXX`) share zero overlap with real patient IDs (`TCGA-*`, `MSK-*`).
3. **Queue Separation**: Historical records cannot enter or be parsed by the synthetic scenario queue.
4. **Cohort Inviolability**: Synthetic scenarios are never written into or merged with `cleaned_cohort.csv`.
5. **Adapter Safety**: All views wrap payloads with `[SYNTHETIC TEST SCENARIO - RESEARCH ONLY]`.

---

## 3. Production FastAPI REST API Architecture

Implemented in `src/api.py`, providing asynchronous endpoints for clinical AI benchmarking:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FASTAPI REST API ENDPOINTS                                     │
├────────┬──────────────────────────────────┬──────────────────────────────────────────────────────┤
│ Method │ Endpoint Route                   │ Purpose & Response Payload                           │
├────────┼──────────────────────────────────┼──────────────────────────────────────────────────────┤
│ GET    │ /health                          │ Health status, pipeline version, scenario count.     │
│ GET    │ /stage5/scenarios                │ Query-filtered scenarios (?include_generated=true).  │
│ GET    │ /stage5/scenarios/{scenario_id}  │ Detailed scenario (demographics, genomics, evidence).│
│ POST   │ /stage5/scenarios/{id}/evaluate  │ Runs live ScenarioEvaluator, updates run history.    │
│ POST   │ /stage5/scenarios/evaluate-all   │ Batch-audits all 20 scenarios, aggregates KPIs.      │
│ POST   │ /stage5/generate                 │ Interactive seed-based patient generator + audit.    │
│ GET    │ /stage5/analytics                │ Comprehensive realism, similarity, & pass analytics. │
│ GET    │ /stage5/evaluation/summary       │ Aggregated KPIs (Pass/Review/Fail, Stress, Realism). │
│ GET    │ /stage5/evaluation/history       │ Longitudinal evaluation history logs.                │
│ GET    │ /dashboard or /                  │ Serves the interactive testing dashboard UI.         │
└────────┴──────────────────────────────────┴──────────────────────────────────────────────────────┘
```

---

---

## 4. Interactive Web Dashboard UI Features

Served by `src/dashboard_app.py` from `dashboard.html`:
* **Prominent Warning Banner**: Unmissable red header declaring: `[SYNTHETIC TEST SCENARIO - RESEARCH ONLY]`.
* **Real-Time KPI Cards**:
  - Total Scenarios: **20** (Benchmark Baseline)
  - Passed: **16 (80%)**
  - Review Required: **4 (20%)**
  - Failed: **0 (0%)**
  - Average Decision-Stress Score: **3.23 / 5.0** (Strong Stress)
  - Average Realism Score: **4.78 / 5.0** (Evidence-Constrained)
  - Blind-Spot Target Coverage: **18/24 targets (75.0%)**
* **Metric Distinction**: Strictly separates the **Scenario Pass Rate** (16/20, 80%) from **Blind-Spot Target Coverage** (18/24 targets, 75.0%).
* **Category Filter Pills**: One-click filtering across `rare_mutation`, `compound_resistance`, `bypass_resistance`, `conflicting_biomarkers`, `unobserved_fusions`, `lineage_switch`, and `wildcard_extreme`.
* **Interactive Generation Modal ("🧬 Generate Patient")**:
  - User can configure specific seed constraints: Age, Sex, Cancer Type, Stage, Histology, Smoking Status, Primary Driver, Variant, TMB, PD-L1, Target Blind Spot, and Prior Treatments.
  - Calls `POST /stage5/generate` to trigger `InteractiveScenarioGenerator`, routes output through `ScenarioEvaluator`, and immediately surfaces results.
* **Telemetry & Realism Analytics Modal ("📊 Analytics")**:
  - Engine Mix: LLM vs. Deterministic Fallback percentage.
  - Longitudinal Pass / Review / Fail distributions.
  - Average Statistical Similarity and Cohort Similarity dials.
  - Recent dynamic generations table with direct scenario inspection links.
* **Dedicated Synthetic Realism / Discriminator Inspector Card**:
  - Surfaces real-time statistical metrics: Data Type, Real-Patient Identity (Never Disclosed / Not Applicable), Statistical Similarity ($0.0 - 1.0$), Cohort Similarity ($0.0 - 1.0$), Calibrated Realism Score ($0.0 - 5.0$), Anomaly Status, Reference Memorization Duplicate Check, Blind Spot Status, and Clinical Interpretation.
* **Scenario Inspector**: Deep dive into demographics, genomics, biomarkers, synthetic assumptions, reference evidence, and uncertainty rationale.
* **Live Action Triggers**:
  - `Audit`: Executes instantaneous single-scenario evaluation.
  - `Batch Evaluate (Run All)`: Re-audits all 20 scenarios and recalculates dashboard KPIs.
  - `Reload Files`: Re-checks disk files via SHA-256 hash comparison.
* **Append-Only History Log**: Displays historical runs, run IDs (`RUN_EDGE_001_001`), timestamps, and score trends.

---

## 5. Benchmark vs. Dynamic Interactive Generation Isolation

To prevent synthetic data drift and test pollution:
1. **Benchmark Baseline Inviolability**: The 20 benchmark scenarios (`EDGE_001` through `EDGE_020`) in `synthetic_edge_cases.jsonl` remain strictly frozen. Regression benchmarks and baseline KPIs are never inflated by interactive test runs.
2. **Distinct Namespace**: All dynamically generated synthetic patients receive unique IDs in the `SYN-XXXXXX` namespace (e.g., `SYN-000001`).
3. **Isolated Persistence**: Generated patients are saved to `stage5_genai/genai/scenarios/generated_scenarios.jsonl`.
4. **Controlled Exposure**: API and Dashboard consumers query benchmark scenarios by default, with `?include_generated=true` required to display interactive records.

---

## 6. Append-Only Persistence & Change Detection

1. **SHA-256 Hash Change Detection**:
   `DashboardService.check_for_updates()` computes the SHA-256 hash and modification time of `synthetic_edge_cases.jsonl`. When upstream files update, the integration layer safely re-ingests without modifying disk files.
2. **Append-Only History Tracking**:
   All audit runs are persisted to `history/evaluation_history.jsonl` with structured IDs (`run_id: RUN_<SCENARIO_ID>_<RUN_NUMBER>`). Historical runs are never overwritten, allowing long-term regression tracking.
3. **Input Immutability**:
   Upstream scenario files are opened in read-only mode, guaranteeing that integration activities never mutate upstream scenario records.

---

## 7. How to Launch and Test the Dashboard

### Launch Dashboard Server:
```powershell
python -m personalized_precision_oncology.stage5_genai.integration.src.dashboard_app
```
Access in your browser:
* Dashboard UI: `http://localhost:8080/dashboard`
* Interactive Swagger Docs: `http://localhost:8080/docs`

---

## 8. Unit Testing Suite & Verification

The Integration test suite features **29 automated pytest unit tests** (100% passing):
```
tests/test_scenario_loader.py          5 passed (Loading, ID retrieval, rejection of synthetic: false)
tests/test_adapter.py                  4 passed (Dashboard item mapping, missing field handling)
tests/test_evaluation_integration.py   3 passed (Single evaluation, batch evaluation, cached loading)
tests/test_history_manager.py          3 passed (Append-and-retrieve, input immutability)
tests/test_dashboard_service.py        9 passed (KPIs, filtering, detail views, all REST endpoints)
tests/test_contamination_protection.py 2 passed (Cohort isolation, target coverage distinction)
tests/test_dashboard_generation.py     3 passed (Interactive generator, analytics KPIs, REST routes)
====================================== 29 passed in 18.45s ======================================
```
