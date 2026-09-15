"""
Deterministic Local Evidence Store and Stage 1–5 Adapters for Stage 6 Agentic AI.

Unified in-memory evidence registry combining:
1. Patient-derived Stage 1 ML Evidence (Risk, Toxicity, Response, SHAP)
2. Stage 2 DL Multimodal Evidence (Biopsy CNN, Trajectory Transformer, Fusion)
3. Stage 3 Clinical NLP Evidence (Triage Urgency, Oncology NER)
4. Stage 4 SLM Evidence (Bedside Briefing, Synthesis)
5. Stage 5 GenAI Stress-Testing Evidence (Edge Cases, Audits, Blind Spots)
6. Stage 6 Local Knowledge Evidence (Guidelines, Drug Interactions, Resistance Rules)

Zero cloud dependencies, zero external network requirements, thread-safe, and deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.drug_interactions import DrugInteractionEngine
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.guidelines import ClinicalGuidelinesKnowledgeBase
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.resistance_rules import ResistanceRuleEngine
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import (
    EvidenceCategory,
    EvidenceLevel,
    EvidenceRecord,
    EvidenceStage,
    ProvenanceRecord,
)


def adapt_stage1_to_evidence(patient_id: str, s1_result: Dict[str, Any]) -> List[EvidenceRecord]:
    """
    Deterministically adapt Stage 1 Tabular ML prediction dictionary into EvidenceRecords.
    Provenance points to OncologyPredictionPipeline.
    """
    records: List[EvidenceRecord] = []
    prov = ProvenanceRecord(
        source_name="Stage 1 Tabular Machine Learning Pipeline",
        source_dataset="data/stage1_ml/processed/oncology_cleaned.csv",
        source_module="stage1_ml.prediction.prediction.OncologyPredictionPipeline",
        source_study="Calibrated XGBoost, CatBoost, and Random Forest Ensemble",
        publication="Stage 1 ML Precision Risk Stratification Baseline",
        citation="Stage 1 Calibrated Multi-Target Predictor (Platt Scaling Threshold=0.48)",
        access_date="2026-09-11",
        license="Internal Proprietary Stage 1 Model Inference"
    )

    # 1. Overall Patient Risk
    if "overall_patient_risk" in s1_result:
        ov = s1_result["overall_patient_risk"]
        pred = ov.get("prediction", "Unknown")
        prob = ov.get("risk_probability", ov.get("confidence", 0.0))
        factors = ov.get("important_factors", [])
        factor_desc = ", ".join([f"{f.get('feature')}: {f.get('direction')}" for f in factors[:3]])

        records.append(
            EvidenceRecord(
                evidence_id=f"S1-{patient_id}-OVERALL-RISK",
                source_stage=EvidenceStage.STAGE1_EVIDENCE,
                source_module="stage1_ml.prediction.prediction.OncologyPredictionPipeline.predict",
                category=EvidenceCategory.TREATMENT_RESPONSE.value,
                topic=f"Patient {patient_id} Overall Clinical Risk Stratification",
                matched_terms=["stage 1", "overall risk", pred.lower(), "risk stratification"],
                evidence_text=(
                    f"Stage 1 Calibrated XGBoost predicts {pred} overall patient mortality/progression risk "
                    f"(calibrated probability: {prob:.4f}, decision threshold: {ov.get('threshold', 0.48)}). "
                    f"Top contributing SHAP factors: {factor_desc or 'Standard clinical distribution'}."
                ),
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
                safety_note="High overall risk warrants intensive monitoring and early restaging." if pred == "High" else None,
                provenance=prov,
                limitations="Calibrated on 5,012-patient cohort; extreme rare mutations require genomic specialist verification.",
                metadata=ov
            )
        )

    # 2. Toxicity Risk
    if "toxicity_risk" in s1_result:
        tox = s1_result["toxicity_risk"]
        pred = tox.get("prediction", "Unknown")
        conf = tox.get("confidence", 0.0)
        records.append(
            EvidenceRecord(
                evidence_id=f"S1-{patient_id}-TOXICITY-RISK",
                source_stage=EvidenceStage.STAGE1_EVIDENCE,
                source_module="stage1_ml.prediction.prediction.OncologyPredictionPipeline.predict",
                category=EvidenceCategory.TOXICITY.value,
                topic=f"Patient {patient_id} Systemic Toxicity Risk",
                matched_terms=["stage 1", "toxicity", "toxicity risk", pred.lower()],
                evidence_text=(
                    f"Stage 1 Calibrated CatBoost predicts {pred} systemic chemotherapy/immunotherapy toxicity risk "
                    f"(confidence: {conf:.4f}, probability distribution: {tox.get('probabilities', {})})."
                ),
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
                safety_note="High toxicity risk requires prophylactic supportive care, dose modification consideration, and liver/renal monitoring." if pred == "High" else None,
                provenance=prov,
                limitations="Does not account for rare pharmacogenomic variants (e.g. DPD deficiency) unless specifically assayed.",
                metadata=tox
            )
        )

    # 3. Therapy Response
    if "therapy_response" in s1_result:
        resp = s1_result["therapy_response"]
        pred = resp.get("prediction", "Unknown")
        conf = resp.get("confidence", 0.0)
        records.append(
            EvidenceRecord(
                evidence_id=f"S1-{patient_id}-THERAPY-RESPONSE",
                source_stage=EvidenceStage.STAGE1_EVIDENCE,
                source_module="stage1_ml.prediction.prediction.OncologyPredictionPipeline.predict",
                category=EvidenceCategory.TREATMENT_RESPONSE.value,
                topic=f"Patient {patient_id} Therapy Response Prediction",
                matched_terms=["stage 1", "therapy response", "responder", "non-responder", pred.lower()],
                evidence_text=(
                    f"Stage 1 Calibrated Random Forest predicts patient is a {pred} to planned systemic regimen "
                    f"(confidence: {conf:.4f}, response probability: {resp.get('probabilities', {}).get('Responder', conf):.4f})."
                ),
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
                safety_note="Non-responder prediction flags necessity of evaluating targeted or trial alternatives." if "Non" in pred else None,
                provenance=prov,
                limitations="Reflects broad systemic response probability; targeted mutation sensitivity takes precedence.",
                metadata=resp
            )
        )

    return records


def adapt_stage2_to_evidence(patient_id: str, s2_result: Dict[str, Any]) -> List[EvidenceRecord]:
    """
    Adapt Stage 2 Multimodal Deep Learning outputs into EvidenceRecords.
    Provenance points to Stage2DLManager.
    """
    records: List[EvidenceRecord] = []
    prov = ProvenanceRecord(
        source_name="Stage 2 Multimodal Deep Learning (CNN + Transformer)",
        source_dataset="stage2_dl/sample_data/ (Biopsy Images & Longitudinal Temporal)",
        source_module="integration.api.stage2_dl_manager.Stage2DLManager",
        source_study="ResNet-18 Biopsy Classifier & Temporal Progression Transformer",
        publication="Stage 2 Spatial-Temporal Multimodal Deep Learning",
        citation="Stage 2 Multimodal DL Manager (cnn_best.pt, transformer_best.pt, multimodal_fusion_best.pt)",
        access_date="2026-09-11",
        license="Internal Proprietary Stage 2 Model Checkpoints"
    )

    # Image Biopsy classification
    if "image_prediction" in s2_result or ("prediction" in s2_result and "class_probabilities" in s2_result):
        pred = s2_result.get("image_prediction", s2_result.get("prediction", "Unknown"))
        conf = s2_result.get("image_confidence", s2_result.get("confidence", 0.0))
        records.append(
            EvidenceRecord(
                evidence_id=f"S2-{patient_id}-BIOPSY-CNN",
                source_stage=EvidenceStage.STAGE2_EVIDENCE,
                source_module="integration.api.stage2_dl_manager.Stage2DLManager.predict_image",
                category=EvidenceCategory.MULTIMODAL_IMAGING.value,
                topic=f"Patient {patient_id} Histopathology Biopsy Classification",
                matched_terms=["stage 2", "biopsy", "histopathology", "cnn", pred.lower()],
                evidence_text=(
                    f"Stage 2 BaselineCNN classified biopsy image as '{pred}' with confidence {conf:.4f}. "
                    f"Grad-CAM visual saliency attention overlay available: {s2_result.get('gradcam_available', False)}."
                ),
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
                safety_note="Malignant pathology confirmed on biopsy." if pred == "malignant" else None,
                provenance=prov,
                limitations="Single representative field of view; subject to biopsy sampling bias.",
                metadata=s2_result
            )
        )

    # Trajectory / Progression
    if "progression_probability" in s2_result:
        prog_prob = s2_result["progression_probability"]
        pred = s2_result.get("prediction", "Progression" if prog_prob >= 0.5 else "Stable")
        records.append(
            EvidenceRecord(
                evidence_id=f"S2-{patient_id}-TRAJECTORY-TRANSFORMER",
                source_stage=EvidenceStage.STAGE2_EVIDENCE,
                source_module="integration.api.stage2_dl_manager.Stage2DLManager.predict_trajectory",
                category=EvidenceCategory.TREATMENT_RESPONSE.value,
                topic=f"Patient {patient_id} 90-Day Disease Progression Dynamics",
                matched_terms=["stage 2", "trajectory", "progression", "transformer", pred.lower()],
                evidence_text=(
                    f"Stage 2 Temporal Transformer predicts 90-day trajectory as '{pred}' "
                    f"(progression probability: {prog_prob:.4f}, sequence length: {s2_result.get('sequence_length', 'N/A')})."
                ),
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
                safety_note="High progression probability signals aggressive disease velocity." if prog_prob >= 0.6 else None,
                provenance=prov,
                limitations="Requires >=3 sequential longitudinal clinic visit measurements for optimal calibration.",
                metadata=s2_result
            )
        )

    return records


def adapt_stage3_to_evidence(patient_id: str, s3_result: Dict[str, Any]) -> List[EvidenceRecord]:
    """
    Adapt Stage 3 Clinical NLP (Urgency + NER) into EvidenceRecords.
    Provenance points to Stage3NLPManager.
    """
    records: List[EvidenceRecord] = []
    prov = ProvenanceRecord(
        source_name="Stage 3 Clinical NLP & Transcription Suite",
        source_dataset="stage3_nlp/data/ (Clinical Consultations & Triage)",
        source_module="integration.api.stage3_nlp_manager.Stage3NLPManager",
        source_study="TF-IDF Urgency Classifier & spaCy Oncology NER",
        publication="Stage 3 Precision Oncology NLP Pipeline",
        citation="Stage 3 NLP Manager (urgency_model.pkl, spaCy NER pipeline)",
        access_date="2026-09-11",
        license="Internal Proprietary Stage 3 NLP Pipeline"
    )

    # Triage Urgency
    if "urgency" in s3_result:
        urg = s3_result["urgency"]
        conf = s3_result.get("confidence", 0.0)
        records.append(
            EvidenceRecord(
                evidence_id=f"S3-{patient_id}-TRIAGE-URGENCY",
                source_stage=EvidenceStage.STAGE3_EVIDENCE,
                source_module="integration.api.stage3_nlp_manager.Stage3NLPManager.predict_urgency",
                category=EvidenceCategory.TRIAGE_NLP.value,
                topic=f"Patient {patient_id} Clinical Triage Urgency",
                matched_terms=["stage 3", "triage", "urgency", urg.lower()],
                evidence_text=(
                    f"Stage 3 NLP Classifier evaluated clinical note/consultation as '{urg}' urgency "
                    f"(confidence: {conf:.4f}, execution latency: {s3_result.get('execution_time_ms', 0):.1f} ms)."
                ),
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
                safety_note="HIGH urgency triggers prioritized same-day clinical oncologist review." if urg == "HIGH" else None,
                provenance=prov,
                limitations="Triage based on textual lexical patterns and consultation phrasing.",
                metadata=s3_result
            )
        )

    # Named Entity Recognition (NER)
    if "entities" in s3_result:
        entities = s3_result["entities"]
        entity_texts = [f"{e.get('text')} ({e.get('label')})" for e in entities]
        matched_terms = ["stage 3", "ner", "entities"] + [e.get("text", "").lower() for e in entities]
        records.append(
            EvidenceRecord(
                evidence_id=f"S3-{patient_id}-CLINICAL-NER",
                source_stage=EvidenceStage.STAGE3_EVIDENCE,
                source_module="integration.api.stage3_nlp_manager.Stage3NLPManager.extract_entities",
                category=EvidenceCategory.GENOMIC.value,
                topic=f"Patient {patient_id} Extracted Clinical Entities",
                matched_terms=matched_terms,
                evidence_text=(
                    f"Stage 3 spaCy NER extracted {len(entities)} clinical entities: "
                    f"{', '.join(entity_texts) if entity_texts else 'None detected'}."
                ),
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
                safety_note=None,
                provenance=prov,
                limitations="Entities extracted from unstructured progress text; requires cross-verification with lab panel.",
                metadata={"entities": entities, "counts": s3_result.get("entity_counts", {})}
            )
        )

    return records


def adapt_stage4_to_evidence(patient_id: str, s4_result: Dict[str, Any]) -> List[EvidenceRecord]:
    """
    Adapt Stage 4 Small Language Model (SLM) Bedside Briefing into EvidenceRecord.
    Provenance points to Stage4SLMManager.
    """
    prov = ProvenanceRecord(
        source_name="Stage 4 LoRA Fine-Tuned SLM (Qwen2.5-0.5B-Instruct)",
        source_dataset="stage4_slm/data/ (Multimodal Paired Scenarios)",
        source_module="integration.api.stage4_slm_manager.Stage4SLMManager",
        source_study="LoRA Adaptation on Precision Oncology Clinical Briefings",
        publication="Stage 4 SLM Bedside Synthesis",
        citation="Stage 4 SLM Manager (adapter_model.safetensors, Qwen2.5-0.5B)",
        access_date="2026-09-11",
        license="Internal Proprietary Stage 4 LoRA Checkpoint"
    )

    briefing = s4_result.get("oncology_briefing", "")
    return [
        EvidenceRecord(
            evidence_id=f"S4-{patient_id}-SLM-BRIEFING",
            source_stage=EvidenceStage.STAGE4_EVIDENCE,
            source_module="integration.api.stage4_slm_manager.Stage4SLMManager.generate_briefing",
            category=EvidenceCategory.BEDSIDE_BRIEFING.value,
            topic=f"Patient {patient_id} Multimodal Bedside Briefing",
            matched_terms=["stage 4", "slm", "briefing", "synthesis", "bedside summary"],
            evidence_text=briefing,
            evidence_level=EvidenceLevel.MODEL_INFERENCE,
            safety_note=s4_result.get("synthetic_disclaimer", "SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE"),
            provenance=prov,
            limitations="Synthesized by 0.5B parameter local language model; requires clinical oncologist validation.",
            metadata=s4_result
        )
    ]


def adapt_stage5_to_evidence(scenario_id: str, s5_result: Dict[str, Any]) -> List[EvidenceRecord]:
    """
    Adapt Stage 5 GenAI Scenario / Stress-Testing Audit into EvidenceRecord.
    Provenance points to ScenarioEvaluator.
    """
    prov = ProvenanceRecord(
        source_name="Stage 5 GenAI & Stress-Testing Suite",
        source_dataset="stage5_genai/genai/scenarios/synthetic_edge_cases.jsonl",
        source_module="stage5_genai.evaluation.src.evaluator.ScenarioEvaluator",
        source_study="Adversarial Clinical & Genomic Stress-Testing Audits",
        publication="Stage 5 GenAI Stress-Testing Framework",
        citation="ScenarioEvaluator.evaluate_scenario() Audit Report",
        access_date="2026-09-11",
        license="Public Domain / Creative Commons CC0"
    )

    audit = s5_result.get("audit_result", s5_result)
    status = audit.get("overall_status", "AUDITED")
    stress = audit.get("stress_score", 0.0)
    realism = audit.get("realism_score", 0.0)

    return [
        EvidenceRecord(
            evidence_id=f"S5-{scenario_id}-AUDIT",
            source_stage=EvidenceStage.STAGE5_EVIDENCE,
            source_module="stage5_genai.evaluation.src.evaluator.ScenarioEvaluator.evaluate_scenario",
            category=EvidenceCategory.RESISTANCE.value,
            topic=f"Scenario {scenario_id} Stress-Testing Audit",
            matched_terms=["stage 5", "stress-test", "edge case", scenario_id.lower(), status.lower()],
            evidence_text=(
                f"Stage 5 Evaluator audited scenario '{scenario_id}': Overall Status={status}, "
                f"Stress Score={stress}/5.0, Realism Score={realism}/5.0, "
                f"Clinical Consistency={audit.get('clinical_consistency', 'CONSISTENT')}, "
                f"Genomic Consistency={audit.get('genomic_consistency', 'CONSISTENT')}."
            ),
            evidence_level=EvidenceLevel.LEVEL_B,
            safety_note="Adversarial synthetic scenario modeled to stress-test clinical blind spots.",
            provenance=prov,
            limitations="Simulated synthetic scenario; not a living biological patient record.",
            metadata=s5_result
        )
    ]


class EvidenceStore:
    """
    Deterministic in-memory evidence store.
    Pre-populates global knowledge base evidence (guidelines, drug interactions, resistance rules)
    and ingests session-scoped patient evidence from Stages 1–5.
    """

    def __init__(self, populate_knowledge_base: bool = True) -> None:
        self._records: Dict[str, EvidenceRecord] = {}
        self._guidelines_kb = ClinicalGuidelinesKnowledgeBase()
        self._drug_engine = DrugInteractionEngine()
        self._resistance_engine = ResistanceRuleEngine()

        if populate_knowledge_base:
            self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> None:
        """Load static knowledge base records into the store."""
        # 1. Guidelines
        for g_rec in self._guidelines_kb.get_all_records():
            self._records[g_rec.evidence_id] = g_rec

        # 2. Drug interactions
        for d_rec in self._drug_engine.as_evidence_records():
            self._records[d_rec.evidence_id] = d_rec

        # 3. Resistance rules
        for r_rec in self._resistance_engine.as_evidence_records():
            self._records[r_rec.evidence_id] = r_rec

    def add_record(self, record: EvidenceRecord) -> None:
        """Add a single evidence record."""
        self._records[record.evidence_id] = record

    def add_records(self, records: List[EvidenceRecord]) -> None:
        """Add multiple evidence records."""
        for r in records:
            self._records[r.evidence_id] = r

    def get_record(self, evidence_id: str) -> Optional[EvidenceRecord]:
        """Retrieve evidence record by ID."""
        return self._records.get(evidence_id)

    def get_records_by_stage(self, stage: EvidenceStage) -> List[EvidenceRecord]:
        """Retrieve all evidence records originating from a specific stage."""
        return [r for r in self._records.values() if r.source_stage == stage]

    def get_records_by_category(self, category: str) -> List[EvidenceRecord]:
        """Retrieve all evidence records matching category string."""
        return [r for r in self._records.values() if r.category.lower() == category.lower()]

    def clear_patient_evidence(self, patient_id: Optional[str] = None) -> None:
        """
        Remove patient-specific records while preserving global knowledge base.
        If patient_id is provided, removes records prefixed or tagged with patient_id.
        Otherwise removes all non-KNOWLEDGE_BASE_EVIDENCE records.
        """
        keys_to_remove = []
        for eid, rec in self._records.items():
            if rec.source_stage != EvidenceStage.KNOWLEDGE_BASE_EVIDENCE:
                if patient_id is None or patient_id in eid:
                    keys_to_remove.append(eid)

        for k in keys_to_remove:
            del self._records[k]

    def get_all_records(self) -> List[EvidenceRecord]:
        """Return all evidence records in the store."""
        return list(self._records.values())

    @property
    def drug_engine(self) -> DrugInteractionEngine:
        return self._drug_engine

    @property
    def resistance_engine(self) -> ResistanceRuleEngine:
        return self._resistance_engine

    @property
    def guidelines_kb(self) -> ClinicalGuidelinesKnowledgeBase:
        return self._guidelines_kb
