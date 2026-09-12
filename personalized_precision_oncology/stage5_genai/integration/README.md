# Stage 5 — Synthetic Scenario Integration Layer

> **Critical Notice**: The Integration layer does not generate synthetic scenarios. It consumes generated and evaluated synthetic scenarios and connects them to the existing testing/evaluation dashboard.
>
> Synthetic scenarios are research-only test artifacts and are not real patients, real clinical outcomes, or clinical validation data.

---

## 1. Overview & Role Responsibilities

The **Stage 5 Integration Engineer** bridges the upstream Stage 5 deliverables:
1. **GenAI Generation** (`genai/scenarios/synthetic_edge_cases.jsonl`) — 20 verified synthetic edge cases.
2. **Evaluation Engine & Reports** (`evaluation/` and `evaluation/reports/`) — Realism and decision-stress audits.

into an interactive testing dashboard and continuous monitoring pipeline.

### Strict System Boundaries
- **No Synthetic Data Generation**: Scenarios are ingested strictly as immutable inputs from `genai/scenarios/`.
- **No Evaluation Logic Rewrite**: Live audits directly invoke the existing `ScenarioEvaluator` from `stage5_genai.evaluation.src.evaluator`.
- **No Data Fabrication**: Unassayed or missing fields are explicitly surfaced as `"NOT EVALUATED"` or `"NOT AVAILABLE"`.
- **Mandatory Synthetic Labeling**: Every payload and view prominently displays `[SYNTHETIC TEST SCENARIO - RESEARCH ONLY]` with `synthetic: true`.
- **Contamination Protection**: Synthetic scenarios cannot mix with or silently enter the historical/reference patient cohort.

---

## 2. Directory Structure

```
personalized_precision_oncology/stage5_genai/integration/
├── __init__.py
├── README.md
├── schemas/
│   ├── integration_scenario_schema.json     # Schema enforcing required synthetic fields
│   └── dashboard_result_schema.json          # Schema validating audit outputs & KPIs
├── src/
│   ├── __init__.py
│   ├── scenario_loader.py                   # Safe JSONL ingestion with duplicate & schema validation
│   ├── scenario_adapter.py                  # Standardized data transformation & badge enforcement
│   ├── evaluation_adapter.py                # Wrapper around official ScenarioEvaluator
│   ├── history_manager.py                   # Append-only evaluation persistence (evaluation_history.jsonl)
│   ├── dashboard_service.py                 # Core coordinator for KPIs, filters, & change detection
│   ├── api.py                               # FastAPI REST endpoints
│   └── dashboard_app.py                     # Standalone HTML/JS testing dashboard & runner
├── history/
│   └── evaluation_history.jsonl             # Longitudinal record of synthetic scenario evaluation runs
└── tests/
    ├── __init__.py
    ├── test_scenario_loader.py              # Tests for loading, validation, duplicate rejection
    ├── test_adapter.py                      # Tests for view mapping & missing-field fallback
    ├── test_evaluation_integration.py       # Tests for live single/batch evaluation
    ├── test_history_manager.py              # Tests for append-only runs & input immutability
    ├── test_dashboard_service.py            # Tests for KPIs, filters, change detection, & REST API
    └── test_contamination_protection.py    # Explicit isolation & anti-contamination tests
```

---

## 3. Data Flow Architecture

```
[ genai/scenarios/synthetic_edge_cases.jsonl ] (Immutable Input)
                      │
                      ▼
             ScenarioLoader
       (Line-by-line JSON validation,
    synthetic: true check, duplicate check)
                      │
                      ▼
              ScenarioAdapter
     (Enforces [SYNTHETIC TEST SCENARIO],
    maps patient, genomics, biomarkers, prov)
                      │
                      ▼
              DashboardService
    (Caches state, computes KPIs & filters,
       monitors file hash for auto-reload)
           │                     │
           ▼                     ▼
      FastAPI API        Interactive Dashboard (HTML/JS)
  (/stage5/scenarios,    (KPI Cards, Table, Inspector,
   /stage5/evaluation)    Live Audit & Run All Buttons)
           │                     │
           └──────────┬──────────┘
                      │ Triggers scenario evaluation
                      ▼
              EvaluationAdapter
    (Invokes ScenarioEvaluator without modification)
                      │
                      ▼
               HistoryManager
   (Appends run to history/evaluation_history.jsonl)
```

---

## 4. API Endpoints

The integration API is implemented using **FastAPI** (`src/api.py`):

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and version. |
| `GET` | `/stage5/scenarios` | Lists scenarios with query filters (`category`, `blind_spot`, `status`, `uncertainty`, `method`). |
| `GET` | `/stage5/scenarios/{scenario_id}` | Detailed scenario record including demographics, alterations, biomarkers, assumptions, evidence, and audit. |
| `POST` | `/stage5/scenarios/{scenario_id}/evaluate` | Executes evaluation via `ScenarioEvaluator`, records run in history, updates caches. |
| `POST` | `/stage5/scenarios/evaluate-all` | Batch-evaluates all available synthetic scenarios, aggregates metrics, records runs. |
| `GET` | `/stage5/evaluation/summary` | Returns aggregated KPIs (Total, Passed, Review, Failed, Avg Stress, Avg Realism, Blind-Spot Coverage). |
| `GET` | `/stage5/evaluation/history` | Retrieves longitudinal evaluation run logs (optional `?scenario_id=EDGE_...`). |
| `GET` | `/` or `/dashboard` | Interactive Web Dashboard interface. |

---

## 5. Web Testing Dashboard

The web dashboard is served directly from `src/dashboard_app.py`:
- **Warning Header**: Red banner highlighting synthetic research-only status.
- **KPI Summary**: Real-time display of Total Scenarios (20), Passed (17), Review (3), Failed (0), Average Stress (3.23), Average Realism (4.78), Blind-Spot Target Coverage (18/24 targets, 75.0%).
- **Blind-Spot Metric Distinction**: Explicitly labeled as `Blind-Spot Target Coverage: 18/24 targets (75.0%)` representing documented EDA blind spots covered, strictly distinguished from scenario pass rate (17/20).
- **Category Filter Pills**: One-click filtering across `rare_mutation`, `compound_resistance`, `bypass_resistance`, `conflicting_biomarkers`, `unobserved_fusions`, `lineage_switch`, `wildcard_extreme`.
- **Dynamic Search & Filters**: Live filtering by evaluation status, uncertainty, and search text.
- **Scenario Inspector**: Deep dive into demographics, genomics, biomarkers, target blind spot, synthetic assumptions, and reference evidence.
- **Live Audit Sub-Checklist**: Inspects Schema, Provenance, Blind Spot, Genomic Consistency, and Clinical Consistency.
- **Execution Actions**:
  - `Audit`: Live evaluations per scenario.
  - `Batch Evaluate (Run All)`: Automated execution across all loaded valid scenarios.
  - `Reload Files`: Re-checks disk files and refreshes caches.
- **Evaluation History Log**: Displays historical runs, run IDs (`RUN_EDGE_001_001`), timestamps, and score trends.

---

## 6. Synthetic Data Isolation & Contamination Protection

The integration layer enforces strict isolation guarantees verified in `tests/test_contamination_protection.py`:
1. **Mandatory Synthetic Flag**: Every record must have `synthetic: true`. Records with `synthetic: false` or missing synthetic flags are rejected.
2. **ID Disjointness**: Synthetic scenario IDs (`EDGE_001` through `EDGE_020`) are completely disjoint from historical patient IDs (`TCGA-*`, `MSK-*`).
3. **Queue Separation**: Real patient records cannot be submitted to or loaded by `ScenarioLoader`.
4. **Adapter Safety**: All views wrap scenarios with `[SYNTHETIC TEST SCENARIO - RESEARCH ONLY]` and explicit clinical warnings.
5. **Cohort Inviolability**: Synthetic scenarios are never written into or merged with the cleaned historical patient cohort.

---

## 7. Continuous Evaluation & Persistence

1. **Change Detection**: `DashboardService.check_for_updates()` checks the SHA-256 hash and modification time of `synthetic_edge_cases.jsonl`. When upstream files update, the integration layer safely re-ingests and validates them without recreating synthetic data.
2. **Append-Only History**: Results are stored in `history/evaluation_history.jsonl` under synthetic scenario evaluation runs. No history is overwritten; every evaluation appends an incremental run record (`run_id: RUN_<SCENARIO_ID>_<RUN_NUMBER>`), allowing regression tracking (e.g. `EDGE_005 Run 1 -> REVIEW`, `EDGE_005 Run 2 -> PASS`).
3. **Input Immutability**: The upstream file `genai/scenarios/synthetic_edge_cases.jsonl` is opened in read-only mode and is guaranteed never to be mutated by integration actions.

---

## 8. Running the Dashboard & Tests

### Launch the Dashboard Server
```bash
python -m personalized_precision_oncology.stage5_genai.integration.src.dashboard_app
```
Access in browser:
- Dashboard: `http://localhost:8080/dashboard`
- OpenAPI Docs: `http://localhost:8080/docs`

### Run Integration Test Suite
```bash
python -m pytest personalized_precision_oncology/stage5_genai/integration/tests -v
```
All 26 integration tests verify schema conformance, loader robustness, duplicate detection, adapter transformations, live single/batch evaluation, history tracking, contamination protection, and REST endpoints.
