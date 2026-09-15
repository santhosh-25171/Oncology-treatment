"""
Multimodal Deep Learning Specialist Agent for Stage 6 Agentic AI.

Wraps the calibrated Stage 2 models (BaselineCNN histopathology classifier,
Transformer progression model, and Multimodal Fusion network).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from personalized_precision_oncology.stage6_agentic.agentic.agents.base_agent import BaseAgent
from personalized_precision_oncology.stage6_agentic.agentic.agents.schemas import (
    AgentResult,
    AgentStatus,
    ClinicalRole,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.evidence_store import adapt_stage2_to_evidence
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import ProvenanceRecord


class MultimodalAgent(BaseAgent):
    """
    Pathology & Disease Dynamics Specialist wrapping Stage 2 Deep Learning models.
    Analyzes biopsy histopathology images and longitudinal clinic biomarker sequences.
    """

    def __init__(self, dl_manager: Optional[Any] = None) -> None:
        super().__init__(
            agent_id="agent_multimodal_diagnostics",
            agent_name="Multimodal Imaging & Disease Dynamics Specialist",
            clinical_role=ClinicalRole.MULTIMODAL_DIAGNOSTICS,
            version="1.0.0"
        )
        self._dl_manager = dl_manager

    def _get_dl_manager(self) -> Any:
        """Lazily initialize Stage2DLManager."""
        if self._dl_manager is None:
            from personalized_precision_oncology.integration.api.stage2_dl_manager import Stage2DLManager
            self._dl_manager = Stage2DLManager()
        return self._dl_manager

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates presence of pre-computed Stage 2 results, image bytes, or temporal records.
        """
        if "stage2_result" in input_data and isinstance(input_data["stage2_result"], dict):
            return True, []

        has_image = bool(input_data.get("image_bytes") or input_data.get("image_file"))
        has_temporal = bool(input_data.get("temporal_records") or input_data.get("trajectory_records"))

        if not has_image and not has_temporal:
            return False, ["image_bytes or temporal_records"]

        return True, []

    def _run(self, input_data: Dict[str, Any]) -> AgentResult:
        patient_id = str(input_data.get("patient_id", "PT_UNKNOWN"))
        stage2_dict = input_data.get("stage2_result")
        missing_modalities: List[str] = []

        # 1. Compute prediction if not pre-provided
        if stage2_dict is None:
            mgr = self._get_dl_manager()
            image_bytes = input_data.get("image_bytes") or input_data.get("image_file")
            temporal_records = input_data.get("temporal_records") or input_data.get("trajectory_records")

            if image_bytes and temporal_records:
                # Joint multimodal prediction
                stage2_dict = mgr.predict_multimodal(image_bytes, temporal_records)
            elif image_bytes:
                # Image only
                stage2_dict = mgr.predict_image(image_bytes)
                missing_modalities.append("temporal_trajectory_records")
            elif temporal_records:
                # Trajectory only
                stage2_dict = mgr.predict_trajectory(temporal_records)
                missing_modalities.append("biopsy_image_bytes")
            else:
                stage2_dict = {}

        # 2. Extract Stage 2 outputs
        image_pred = "Unknown"
        image_conf = 0.0
        has_gradcam = False

        if "biopsy_image_bytes" not in missing_modalities:
            image_pred = stage2_dict.get("image_prediction", stage2_dict.get("prediction", "Unknown"))
            image_conf = float(stage2_dict.get("image_confidence", stage2_dict.get("confidence", 0.0)))
            has_gradcam = bool(stage2_dict.get("gradcam_available", False))

        prog_prob = stage2_dict.get("progression_probability")
        trajectory_pred = stage2_dict.get("temporal_prediction")
        if trajectory_pred is None and "temporal_trajectory_records" not in missing_modalities:
            trajectory_pred = stage2_dict.get("prediction")
        if prog_prob is not None and trajectory_pred is None:
            trajectory_pred = "Progression" if float(prog_prob) >= 0.5 else "Stable"

        # 3. Generate Evidence Records via Stage 6 Adapter
        evidence_records = adapt_stage2_to_evidence(patient_id, stage2_dict)
        evidence_ids = [r.evidence_id for r in evidence_records]

        # 4. Synthesize clinical findings, warnings, and confidence
        warnings: List[str] = []
        confidences: List[float] = []

        if image_pred == "malignant":
            warnings.append(f"Histopathology biopsy confirmed as malignant (confidence: {image_conf:.2%}).")
            confidences.append(image_conf)

        if prog_prob is not None:
            prob_f = float(prog_prob)
            confidences.append(prob_f if prob_f >= 0.5 else 1.0 - prob_f)
            if prob_f >= 0.60:
                warnings.append(f"Elevated 90-day progression probability ({prob_f:.1%}) indicates active disease velocity.")

        overall_confidence = float(sum(confidences) / max(len(confidences), 1)) if confidences else image_conf

        # Summary construction
        parts = []
        if "biopsy_image_bytes" not in missing_modalities and image_pred != "Unknown":
            parts.append(f"Biopsy tissue: '{image_pred}' ({image_conf:.1%}) [Grad-CAM: {'Active' if has_gradcam else 'Off'}]")
        if "temporal_trajectory_records" not in missing_modalities and prog_prob is not None:
            parts.append(f"90-day disease dynamic: '{trajectory_pred}' (progression probability: {float(prog_prob):.1%})")

        summary = (
            f"Stage 2 Multimodal Assessment: {'; '.join(parts) or 'Modality evaluated'}."
        )

        if missing_modalities:
            warnings.append(f"Incomplete multimodal assessment: missing {', '.join(missing_modalities)}.")

        next_action = (
            "Recommend restaging CT/PET scan and trajectory verification."
            if (prog_prob is not None and float(prog_prob) >= 0.60)
            else "Maintain scheduled imaging surveillance interval."
        )

        provenance = ProvenanceRecord(
            source_name="Stage 2 Multimodal Deep Learning (CNN + Transformer)",
            source_dataset="stage2_dl/sample_data/ (Biopsy Images & Longitudinal Temporal)",
            source_study="ResNet-18 Biopsy Classifier & Temporal Progression Transformer",
            source_module="integration.api.stage2_dl_manager.Stage2DLManager",
            publication="Stage 2 Spatial-Temporal Multimodal Deep Learning",
            doi_or_pmid="Internal Stage 2 Model Checkpoints",
            citation="Stage 2 Stage2DLManager (predict_image, predict_trajectory, predict_multimodal)",
            access_date="2026-09-14",
            license="Internal Stage 2 Model Assets"
        )

        status = AgentStatus.WARNING if warnings else AgentStatus.SUCCESS

        return AgentResult(
            status=status,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            clinical_role=self.clinical_role,
            findings={
                "image_prediction": image_pred,
                "image_confidence": image_conf,
                "gradcam_available": has_gradcam,
                "progression_probability": prog_prob,
                "trajectory_prediction": trajectory_pred,
                "raw_stage2_output": stage2_dict
            },
            summary=summary,
            confidence=round(overall_confidence, 4),
            evidence_ids=evidence_ids,
            provenance=provenance,
            warnings=warnings,
            missing_data=missing_modalities,
            next_action=next_action,
            execution_time_ms=0.0,
            metadata={"patient_id": patient_id}
        )
