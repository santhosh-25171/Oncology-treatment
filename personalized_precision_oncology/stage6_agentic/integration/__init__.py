"""
Stage 6 Agentic AI Integration Layer.

Provides:
- FastAPI REST Microservice routes and schemas
- Streamlit Clinician Workstation Prototype
- Physician Review Gate & Traceable Override Governance
- Structured Audit Trail Logger
"""

from personalized_precision_oncology.stage6_agentic.integration.api.schemas import (
    ErrorResponse,
    PatientCaseRequest,
    PatientCaseResponse,
    ReviewOverrideRequest,
    ReviewOverrideResponse,
    Stage6HealthResponse,
    TreatmentCandidateResponse,
)
from personalized_precision_oncology.stage6_agentic.integration.api.service import (
    Stage6IntegrationService,
    default_integration_service,
)
from personalized_precision_oncology.stage6_agentic.integration.api.routes import (
    stage6_router,
)
from personalized_precision_oncology.stage6_agentic.integration.physician_review.review_gate import (
    ReviewDecision,
    ReviewStatusSummary,
    PhysicianReviewGate,
)
from personalized_precision_oncology.stage6_agentic.integration.physician_review.override import (
    PhysicianOverride,
    PhysicianOverrideManager,
    default_override_manager,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_schema import (
    AuditEvent,
    AuditEventType,
)
from personalized_precision_oncology.stage6_agentic.integration.audit.audit_logger import (
    AuditLogger,
    default_audit_logger,
)

__all__ = [
    # API
    "ErrorResponse",
    "PatientCaseRequest",
    "PatientCaseResponse",
    "ReviewOverrideRequest",
    "ReviewOverrideResponse",
    "Stage6HealthResponse",
    "TreatmentCandidateResponse",
    "Stage6IntegrationService",
    "default_integration_service",
    "stage6_router",
    # Physician Review
    "ReviewDecision",
    "ReviewStatusSummary",
    "PhysicianReviewGate",
    "PhysicianOverride",
    "PhysicianOverrideManager",
    "default_override_manager",
    # Audit
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "default_audit_logger",
]
