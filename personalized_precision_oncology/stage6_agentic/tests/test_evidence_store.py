"""
Unit tests for Stage 6 Evidence Store, Stage 1–5 Adapters, and Source Data Immutability.
"""

import hashlib
from pathlib import Path
import pytest
from pydantic import ValidationError

from personalized_precision_oncology.stage6_agentic.agentic.knowledge.drug_interactions import DrugInteractionEngine
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.evidence_store import (
    EvidenceStore,
    adapt_stage1_to_evidence,
    adapt_stage2_to_evidence,
    adapt_stage3_to_evidence,
    adapt_stage4_to_evidence,
    adapt_stage5_to_evidence,
)
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.resistance_rules import ResistanceRuleEngine
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import (
    DrugInteractionRule,
    EvidenceCategory,
    EvidenceLevel,
    EvidenceRecord,
    EvidenceStage,
    ProvenanceRecord,
)


class TestEvidenceSchemas:
    """Test Pydantic schema validation, constraints, and data integrity."""

    def test_evidence_record_valid(self) -> None:
        prov = ProvenanceRecord(
            source_name="Test Source",
            source_dataset="test.csv",
            source_module="test_module",
            publication="Test et al. 2026",
            doi_or_pmid="DOI:10.1000/123",
            access_date="2026-09-14",
            license="Open Access"
        )
        rec = EvidenceRecord(
            evidence_id="TEST-001",
            source_stage=EvidenceStage.STAGE1_EVIDENCE,
            source_module="test_module.func",
            category=EvidenceCategory.GENOMIC.value,
            topic="Test Topic",
            matched_terms=["egfr", "l858r"],
            evidence_text="Detailed test evidence narrative.",
            evidence_level=EvidenceLevel.LEVEL_A,
            safety_note="Test safety note.",
            provenance=prov,
            limitations="No known limitations.",
            metadata={"key": "value"}
        )
        assert rec.evidence_id == "TEST-001"
        assert rec.source_stage == EvidenceStage.STAGE1_EVIDENCE
        assert rec.evidence_level == EvidenceLevel.LEVEL_A
        assert rec.provenance.source_name == "Test Source"

    def test_provenance_forbid_extra(self) -> None:
        """ProvenanceRecord must strictly forbid unrecognized/arbitrary extra fields."""
        with pytest.raises(ValidationError):
            ProvenanceRecord(
                source_name="Test Source",
                unauthorized_hallucinated_field="forbidden"
            )

    def test_drug_interaction_rule_valid(self) -> None:
        rule = DrugInteractionRule(
            drug_or_class="Osimertinib",
            interacting_drug_or_class="Rifampin",
            interaction_type="CYP3A4_INDUCTION",
            severity="CONTRAINDICATED",
            warning="Severe reduction in osimertinib exposure.",
            source="FDA Label",
            evidence_level="Level A",
            review_required=True
        )
        assert rule.severity == "CONTRAINDICATED"
        assert rule.review_required is True


class TestStageAdaptersAndSeparation:
    """Test Stage 1–5 deterministic adapter transformations and clean evidence separation."""

    @pytest.fixture
    def store_with_all_stages(self) -> EvidenceStore:
        store = EvidenceStore(populate_knowledge_base=True)
        patient_id = "SYNTH_PATIENT_TEST"

        # Mock Stage 1 Result
        s1_result = {
            "overall_patient_risk": {
                "prediction": "High",
                "risk_probability": 0.7825,
                "threshold": 0.48,
                "important_factors": [{"feature": "comorbidity_score", "direction": "increases_risk"}]
            },
            "toxicity_risk": {
                "prediction": "Moderate",
                "confidence": 0.6214,
                "probabilities": {"Moderate": 0.6214}
            },
            "therapy_response": {
                "prediction": "Non-Responder",
                "confidence": 0.5842
            }
        }
        store.add_records(adapt_stage1_to_evidence(patient_id, s1_result))

        # Mock Stage 2 Result
        s2_result = {
            "image_prediction": "malignant",
            "image_confidence": 0.8841,
            "gradcam_available": True,
            "progression_probability": 0.7412,
            "prediction": "Progression",
            "sequence_length": 5
        }
        store.add_records(adapt_stage2_to_evidence(patient_id, s2_result))

        # Mock Stage 3 Result
        s3_result = {
            "urgency": "HIGH",
            "confidence": 0.9234,
            "entities": [
                {"text": "EGFR L858R", "label": "GENE_MUTATION"},
                {"text": "cisplatin", "label": "DRUG_NAME"}
            ],
            "execution_time_ms": 18.2
        }
        store.add_records(adapt_stage3_to_evidence(patient_id, s3_result))

        # Mock Stage 4 Result
        s4_result = {
            "oncology_briefing": "Patient presents with Stage IV disease exhibiting high overall clinical risk.",
            "sentence_count": 1,
            "synthetic_disclaimer": "SYNTHETIC RESEARCH DATA"
        }
        store.add_records(adapt_stage4_to_evidence(patient_id, s4_result))

        # Mock Stage 5 Result
        s5_result = {
            "audit_result": {
                "overall_status": "PASS",
                "stress_score": 4.2,
                "realism_score": 4.6,
                "clinical_consistency": "CONSISTENT",
                "genomic_consistency": "CONSISTENT"
            }
        }
        store.add_records(adapt_stage5_to_evidence("EDGE_001", s5_result))

        return store

    def test_stage_evidence_separation(self, store_with_all_stages: EvidenceStore) -> None:
        """Verify that records are partitioned by stage without cross-contamination."""
        s1_records = store_with_all_stages.get_records_by_stage(EvidenceStage.STAGE1_EVIDENCE)
        s2_records = store_with_all_stages.get_records_by_stage(EvidenceStage.STAGE2_EVIDENCE)
        s3_records = store_with_all_stages.get_records_by_stage(EvidenceStage.STAGE3_EVIDENCE)
        s4_records = store_with_all_stages.get_records_by_stage(EvidenceStage.STAGE4_EVIDENCE)
        s5_records = store_with_all_stages.get_records_by_stage(EvidenceStage.STAGE5_EVIDENCE)
        kb_records = store_with_all_stages.get_records_by_stage(EvidenceStage.KNOWLEDGE_BASE_EVIDENCE)

        assert len(s1_records) == 3  # overall, toxicity, therapy response
        assert len(s2_records) == 2  # cnn, trajectory
        assert len(s3_records) == 2  # urgency, ner
        assert len(s4_records) == 1  # briefing
        assert len(s5_records) >= 1  # audit + Stage 5 edge-case resistance records
        assert any(r.evidence_id == "S5-EDGE_001-AUDIT" for r in s5_records)
        assert len(kb_records) > 0   # guidelines, drug interactions, resistance

    def test_provenance_preservation(self, store_with_all_stages: EvidenceStore) -> None:
        """Every stage-adapted record must preserve source module provenance."""
        s1_rec = store_with_all_stages.get_record("S1-SYNTH_PATIENT_TEST-OVERALL-RISK")
        assert s1_rec is not None
        assert "OncologyPredictionPipeline" in s1_rec.provenance.source_module

        s2_rec = store_with_all_stages.get_record("S2-SYNTH_PATIENT_TEST-BIOPSY-CNN")
        assert s2_rec is not None
        assert "Stage2DLManager" in s2_rec.provenance.source_module

        s3_rec = store_with_all_stages.get_record("S3-SYNTH_PATIENT_TEST-TRIAGE-URGENCY")
        assert s3_rec is not None
        assert "Stage3NLPManager" in s3_rec.provenance.source_module

        s4_rec = store_with_all_stages.get_record("S4-SYNTH_PATIENT_TEST-SLM-BRIEFING")
        assert s4_rec is not None
        assert "Stage4SLMManager" in s4_rec.provenance.source_module

        s5_rec = store_with_all_stages.get_record("S5-EDGE_001-AUDIT")
        assert s5_rec is not None
        assert "ScenarioEvaluator" in s5_rec.provenance.source_module

    def test_clear_patient_evidence(self, store_with_all_stages: EvidenceStore) -> None:
        """Clearing patient session records leaves static knowledge base intact."""
        initial_kb_count = len(store_with_all_stages.get_records_by_stage(EvidenceStage.KNOWLEDGE_BASE_EVIDENCE))
        assert initial_kb_count > 0

        store_with_all_stages.clear_patient_evidence(patient_id="SYNTH_PATIENT_TEST")

        assert len(store_with_all_stages.get_records_by_stage(EvidenceStage.STAGE1_EVIDENCE)) == 0
        assert len(store_with_all_stages.get_records_by_stage(EvidenceStage.STAGE2_EVIDENCE)) == 0
        assert len(store_with_all_stages.get_records_by_stage(EvidenceStage.STAGE3_EVIDENCE)) == 0
        assert len(store_with_all_stages.get_records_by_stage(EvidenceStage.STAGE4_EVIDENCE)) == 0
        # Knowledge base count must be unaltered
        assert len(store_with_all_stages.get_records_by_stage(EvidenceStage.KNOWLEDGE_BASE_EVIDENCE)) == initial_kb_count


class TestDrugInteractions:
    """Test drug interaction engine and contraindication rules."""

    def test_known_interaction_lookup(self) -> None:
        engine = DrugInteractionEngine()
        rule = engine.check_interaction("Osimertinib", "Rifampin")
        assert rule is not None
        assert rule.severity == "CONTRAINDICATED"
        assert "CYP3A" in rule.interaction_type

        # Test bidirectional lookup (Rifampin + Osimertinib)
        rule_rev = engine.check_interaction("Rifampin", "Osimertinib")
        assert rule_rev is not None
        assert rule_rev.severity == "CONTRAINDICATED"

        # Cisplatin + Gentamicin
        cis_gent = engine.check_interaction("Cisplatin", "Gentamicin")
        assert cis_gent is not None
        assert cis_gent.severity == "CONTRAINDICATED"
        assert "NEPHROTOXICITY" in cis_gent.interaction_type

    def test_unknown_interaction_returns_none(self) -> None:
        """Unknown drug pairs must return None rather than hallucinating rules."""
        engine = DrugInteractionEngine()
        rule = engine.check_interaction("Aspirin", "Vitamin C")
        assert rule is None


class TestResistanceRuleEngine:
    """Test resistance rules parsing and Stage 5 stress-test consumption."""

    def test_resistance_queries(self) -> None:
        engine = ResistanceRuleEngine()
        egfr_rules = engine.get_resistance_rules_for_gene("EGFR")
        assert len(egfr_rules) > 0
        variants = [r.variant for r in egfr_rules]
        assert any("T790M" in v for v in variants)
        assert any("C797S" in v for v in variants)

        kras_rules = engine.get_resistance_rules_for_gene("KRAS")
        assert len(kras_rules) > 0

        alk_rules = engine.get_resistance_rules_for_gene("ALK")
        assert len(alk_rules) > 0

    def test_edge_case_loading(self) -> None:
        engine = ResistanceRuleEngine()
        edge_cases = engine.get_edge_case_scenarios()
        assert len(edge_cases) >= 20  # The 20 immutable synthetic edge cases


class TestSourceDataImmutability:
    """Verify that source Stage 1–5 dataset files are NEVER modified."""

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def test_stage5_source_files_remain_unmodified(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        stage5_dir = repo_root / "personalized_precision_oncology" / "stage5_genai"

        edge_cases_file = stage5_dir / "genai" / "scenarios" / "synthetic_edge_cases.jsonl"
        baseline_file = stage5_dir / "data_engineering" / "processed" / "genai_reference_baseline.jsonl"

        assert edge_cases_file.exists()
        assert baseline_file.exists()

        hash_edge_before = self._file_hash(edge_cases_file)
        hash_base_before = self._file_hash(baseline_file)

        # Instantiate engines and run operations
        engine = ResistanceRuleEngine(repo_root=repo_root)
        _ = engine.get_all_rules()
        _ = engine.get_edge_case_scenarios()
        _ = engine.as_evidence_records()

        store = EvidenceStore(populate_knowledge_base=True)
        _ = store.get_all_records()

        hash_edge_after = self._file_hash(edge_cases_file)
        hash_base_after = self._file_hash(baseline_file)

        assert hash_edge_before == hash_edge_after, "synthetic_edge_cases.jsonl was unexpectedly modified!"
        assert hash_base_before == hash_base_after, "genai_reference_baseline.jsonl was unexpectedly modified!"
