"""
FastAPI REST API for Stage 5 Synthetic Oncology Testing Dashboard.
Exposes endpoints for scenario retrieval, inspection, single/batch evaluation,
KPI summaries, and longitudinal evaluation history.
"""

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dashboard_service import DashboardService

app = FastAPI(
    title="Personalized Precision Oncology - Stage 5 Synthetic Scenario Testing API",
    description="Integration API for executing, evaluating, and monitoring synthetic oncology edge cases.",
    version="5.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = DashboardService()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "HEALTHY", "stage": "Stage 5 Integration", "version": "5.0.0"}


@app.get("/stage5/scenarios")
def list_scenarios(
    category: Optional[str] = Query(None, description="Filter by scenario category"),
    blind_spot: Optional[str] = Query(None, description="Filter by target blind spot ID"),
    status: Optional[str] = Query(None, description="Filter by evaluation status (PASS, REVIEW, FAIL)"),
    uncertainty: Optional[str] = Query(None, description="Filter by uncertainty level (low, moderate, high)"),
    method: Optional[str] = Query(None, description="Filter by generation method (template, llm)")
) -> List[Dict[str, Any]]:
    """Lists synthetic scenarios with optional metadata filters."""
    return service.get_scenarios(
        category=category,
        blind_spot=blind_spot,
        status=status,
        uncertainty=uncertainty,
        method=method
    )


@app.get("/stage5/scenarios/{scenario_id}")
def get_scenario(scenario_id: str) -> Dict[str, Any]:
    """Retrieves complete detailed representation of a synthetic scenario."""
    detail = service.get_scenario_detail(scenario_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")
    return detail


@app.post("/stage5/scenarios/{scenario_id}/evaluate")
def evaluate_scenario(scenario_id: str) -> Dict[str, Any]:
    """Triggers live evaluation of a single scenario through the Stage 5 evaluation engine."""
    try:
        return service.evaluate_single_scenario(scenario_id)
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(exc)}")


@app.post("/stage5/scenarios/evaluate-all")
def evaluate_all() -> Dict[str, Any]:
    """Triggers batch evaluation across all loaded synthetic scenarios."""
    try:
        return service.evaluate_all_scenarios()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(exc)}")


@app.get("/stage5/evaluation/summary")
def get_summary() -> Dict[str, Any]:
    """Returns aggregated Stage 5 testing KPIs and distributions."""
    return service.get_summary()


@app.get("/stage5/evaluation/history")
def get_history(scenario_id: Optional[str] = Query(None, description="Filter history by scenario ID")) -> List[Dict[str, Any]]:
    """Returns longitudinal evaluation history records."""
    return service.get_history(scenario_id=scenario_id)
