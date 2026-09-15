"""
Risk Agent for Stage 6 Agentic AI.

Wraps the calibrated Stage 1 Machine Learning pipeline (OncologyPredictionPipeline)
to assess overall patient mortality/progression risk, toxicity risk, and therapy response.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.evidence_store import adapt_stage1_to_evidence
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


class RiskAgent(BaseAgent):
    """
    Specialist Agent wrapping the Stage 1 Tabular Machine Learning Pipeline.
    Evaluates calibrated overall patient risk, toxicity propensity, and therapy response.
    """

    def __init__(self, pipeline: Optional[Any] = None) -> None:
        super().__init__(
            agent_id="agent_risk_stratification",
            agent_name="Clinical Risk Stratification Specialist",
            clinical_role=ClinicalRole.RISK_STRATIFICATION,
            version="1.0.0"
        )
        self._pipeline = pipeline

    def _get_pipeline(self) -> Any:
        """Lazily initialize the Stage 1 pipeline to prevent heavy imports during startup."""
        if self._pipeline is None:
            from personalized_precision_oncology.stage1_ml.prediction.prediction import OncologyPredictionPipeline
            self._pipeline = OncologyPredictionPipeline()
        return self._pipeline

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates presence of clinical features or pre-computed Stage 1 results.
        """
        if "stage1_result" in input_data and isinstance(input_data["stage1_result"], dict):
            return True, []

        patient_dict = input_data.get("patient_dict") or input_data.get("patient_features") or input_data
        # Check minimum essential fields
        core_fields = ["cancer_type", "age", "tumor_size", "performance_status", "comorbidity_score"]
        missing = [f for f in core_fields if f not in patient_dict]

        # Require at least 4 of the 5 core clinical fields
        if len(missing) > 1:
            return False, missing

        return True, []

    def _run(self, input_data: Dict[str, Any]) -> AgentResult:
        patient_id = str(input_data.get("patient_id", "PT_UNKNOWN"))
        stage1_dict = input_data.get("stage1_result")

        # 1. Compute prediction if not pre-provided
        if stage1_dict is None:
            pipeline = self._get_pipeline()
            patient_features = input_data.get("patient_dict") or input_data.get("patient_features") or input_data
            stage1_dict = pipeline.predict(patient_features)

        # 2. Extract Stage 1 outputs
        ov_risk = stage1_dict.get("overall_patient_risk", {})
        overall_pred = ov_risk.get("prediction", "Unknown")
        overall_prob = float(ov_risk.get("risk_probability", ov_risk.get("confidence", 0.5)))
        threshold = float(ov_risk.get("threshold", 0.48))
        important_factors = ov_risk.get("important_factors", [])

        tox_risk = stage1_dict.get("toxicity_risk", {})
        tox_pred = tox_risk.get("prediction", "Unknown")
        tox_conf = float(tox_risk.get("confidence", 0.5))

        ther_resp = stage1_dict.get("therapy_response", {})
        ther_pred = ther_resp.get("prediction", "Unknown")
        ther_conf = float(ther_resp.get("confidence", 0.5))

        # Overall confidence is calibrated probability max
        confidence = float(ov_risk.get("confidence", overall_prob))

        # 3. Generate Evidence Records via Stage 6 Adapter
        evidence_records = adapt_stage1_to_evidence(patient_id, stage1_dict)
        evidence_ids = [r.evidence_id for r in evidence_records]

        # 4. Synthesize clinical narrative and warnings
        warnings: List[str] = []
        if overall_pred == "High":
            warnings.append(
                f"High overall patient risk (calibrated probability: {overall_prob:.4f} >= threshold {threshold:.2f})."
            )
        if tox_pred == "High":
            warnings.append(f"High chemotherapy/immunotherapy toxicity risk predicted by CatBoost (confidence: {tox_conf:.2f}).")
        if ther_pred == "Non-Responder":
            warnings.append(f"Patient predicted as Non-Responder to standard systemic therapy (confidence: {ther_conf:.2f}).")

        # Format factor summary
        top_factors_desc = ", ".join([f"{f.get('feature')} ({f.get('direction')})" for f in important_factors[:3]])

        summary = (
            f"Stage 1 ML Stratification: {overall_pred} Overall Clinical Risk "
            f"(calibrated probability {overall_prob:.2%}, decision threshold {threshold:.2f}). "
            f"Toxicity Risk: {tox_pred} ({tox_conf:.1%}). Therapy Response: {ther_pred} ({ther_conf:.1%}). "
            f"Key driving factors: {top_factors_desc or 'Standard baseline distribution'}."
        )

        next_action = (
            "Prioritize multidisciplinary tumor board review for aggressive risk-adapted strategy."
            if overall_pred == "High" or ther_pred == "Non-Responder"
            else "Standard clinical protocol with routine toxicity monitoring."
        )

        provenance = ProvenanceRecord(
            source_name="Stage 1 Tabular Machine Learning Pipeline",
            source_dataset="data/stage1_ml/processed/oncology_cleaned.csv",
            source_study="Calibrated XGBoost (Threshold=0.48), CatBoost, Random Forest",
            source_module="stage1_ml.prediction.prediction.OncologyPredictionPipeline.predict",
            publication="Stage 1 Multi-Target Calibrated Risk Stratification",
            doi_or_pmid="Internal Stage 1 Model Inference",
            citation="Stage 1 OncologyPredictionPipeline.predict()",
            access_date="2026-09-14",
            license="Internal Stage 1 Model Assets"
        )

        status = AgentStatus.WARNING if warnings else AgentStatus.SUCCESS

        return AgentResult(
            status=status,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            clinical_role=self.clinical_role,
            findings={
                "overall_patient_risk": overall_pred,
                "risk_probability": overall_prob,
                "decision_threshold": threshold,
                "toxicity_risk": tox_pred,
                "toxicity_confidence": tox_conf,
                "therapy_response": ther_pred,
                "therapy_confidence": ther_conf,
                "important_factors": important_factors,
                "raw_stage1_output": stage1_dict
            },
            summary=summary,
            confidence=confidence,
            evidence_ids=evidence_ids,
            provenance=provenance,
            warnings=warnings,
            missing_data=[],
            next_action=next_action,
            execution_time_ms=0.0,
            metadata={"patient_id": patient_id}
        )
