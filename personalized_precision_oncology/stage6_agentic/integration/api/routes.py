"""
FastAPI Routes for Stage 6 Agentic AI Integration Layer.

Exposes thin, robust endpoints for:
- Case deliberation analysis (POST /stage6/analyze)
- Physician review override recording (POST /stage6/review/override)
- Structured audit trail querying (GET /stage6/audit/{case_id})
- Component health check (GET /stage6/health)

Adheres to:
1. Thin routes delegating directly to Stage6IntegrationService.
2. Structured JSON error responses without leaking Python stack traces.
3. Strict clinical decision-support semantics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import JSONResponse

from personalized_precision_oncology.stage6_agentic.integration.api.schemas import (
    ErrorResponse,
    PatientCaseRequest,
    PatientCaseResponse,
    ReviewOverrideRequest,
    ReviewOverrideResponse,
    Stage6HealthResponse,
)
from personalized_precision_oncology.stage6_agentic.integration.api.service import (
    Stage6IntegrationService,
    default_integration_service,
)

logger = logging.getLogger("Stage6Routes")

stage6_router = APIRouter(prefix="/stage6", tags=["Stage 6 Agentic AI"])


@stage6_router.get("/health", response_model=Stage6HealthResponse)
def get_stage6_health() -> Stage6HealthResponse:
    """Returns component health and operational status for the Stage 6 agentic service."""
    try:
        return default_integration_service.get_health()
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stage 6 health check failed",
        )


@stage6_router.post("/analyze", response_model=PatientCaseResponse)
def analyze_case(request: PatientCaseRequest) -> PatientCaseResponse:
    """
    Executes multidisciplinary tumor board case deliberation.
    Gathers evidence across specialist agents, assesses safety, applies the
    physician review gate, and returns structured decision-support output.
    """
    try:
        return default_integration_service.analyze_case(request)
    except ValueError as ve:
        logger.warning(f"Validation error in Stage 6 analysis: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Unexpected error in Stage 6 case analysis: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="DeliberationWorkflowError",
                detail="An internal error occurred during case deliberation. Please consult system logs.",
                case_id=request.case_id or request.patient_id,
                status_code=500,
            ).model_dump(),
        )


@stage6_router.post("/review/override", response_model=ReviewOverrideResponse)
def record_physician_override(request: ReviewOverrideRequest) -> ReviewOverrideResponse:
    """
    Records an oncologist's decision-support review, approval, modification, or rejection.
    Preserves original AI output immutably in the audit log.
    """
    try:
        return default_integration_service.record_physician_override(request)
    except ValueError as ve:
        logger.warning(f"Validation error in physician override: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Failed to record physician override: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="PhysicianOverrideError",
                detail="Failed to record physician override. Please verify submission format.",
                case_id=request.case_id,
                status_code=500,
            ).model_dump(),
        )


@stage6_router.get("/audit/{case_id}", response_model=List[Dict[str, Any]])
def get_case_audit_trail(case_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves chronological structured audit events recorded for a specific case.
    """
    try:
        return default_integration_service.get_audit_trail(case_id)
    except Exception as e:
        logger.error(f"Failed to query audit trail for case {case_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit trail",
        )
