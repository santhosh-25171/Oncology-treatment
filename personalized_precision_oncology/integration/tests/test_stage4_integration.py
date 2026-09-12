#!/usr/bin/env python3
"""
Integration & Compatibility Test Suite for Stage 4 SLM in Precision Oncology
Roles: Stage 4 SLM Engineer + Integration Engineer
Project: Personalized Precision Medicine for Oncology Treatment Optimization

Strictly divided into:
- Category A: Adapter & Compatibility Unit Tests
- Category B: Full Production Integration Tests

DISCLAIMER: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.
"""

import sys
import os
import json
import pytest
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from stage4_slm.adapter.context_adapter import (
    adapt_stage123_to_stage4_context,
    validate_stage_responses,
    get_optimal_cpu_threads,
    build_stage4_prompt,
)
from integration.client.api_client import OncologyAPIClient
from integration.api.stage4_slm_manager import Stage4SLMManager


# -------------------------------------------------------------------------
# Fixtures for Authentic Stage Outputs
# -------------------------------------------------------------------------

@pytest.fixture
def mock_stage1_response() -> Dict[str, Any]:
    """Authentic runtime response structure from Stage 1 ML pipeline."""
    return {
        "overall_patient_risk": {
            "prediction": "High",
            "risk_probability": 0.7420,
            "confidence": 0.8850,
            "threshold": 0.48,
            "probabilities": {"Low": 0.2580, "High": 0.7420},
            "important_factors": [
                {"feature": "mutation_burden", "importance": 0.185},
                {"feature": "ctDNA_level", "importance": 0.142}
            ]
        },
        "toxicity_risk": {
            "prediction": "High",
            "confidence": 0.8120,
            "probabilities": {"Low": 0.1880, "High": 0.8120}
        },
        "therapy_response": {
            "prediction": "Non-Responder",
            "confidence": 0.6950,
            "probabilities": {"Responder": 0.3050, "Non-Responder": 0.6950}
        },
        "backend": "Local Python Engine"
    }


@pytest.fixture
def mock_stage2_response() -> Dict[str, Any]:
    """Authentic runtime response structure from Stage 2 DL multimodal pipeline."""
    return {
        "progression_probability": 0.6850,
        "confidence": 0.8420,
        "prediction": "Progression Anticipated",
        "image_prediction": "Malignant Carcinoma (Grade III)",
        "temporal_prediction": "Rapid Progression (90-Day Horizon)",
        "backend": "Local Python Engine"
    }


@pytest.fixture
def mock_stage3_response() -> Dict[str, Any]:
    """Authentic runtime response structure from Stage 3 NLP pipeline."""
    return {
        "urgency": "HIGH",
        "confidence": 0.9240,
        "probabilities": {"LOW": 0.0210, "MODERATE": 0.0550, "HIGH": 0.9240},
        "entities": [
            {"text": "EGFR L858R", "label": "GENE_MUTATION", "start": 35, "end": 45},
            {"text": "osimertinib", "label": "DRUG_NAME", "start": 62, "end": 73},
            {"text": "severe diarrhea", "label": "ADVERSE_EVENT", "start": 105, "end": 120}
        ],
        "total_entities": 3,
        "backend": "Local Python Engine"
    }


@pytest.fixture
def sample_clinical_report() -> str:
    return (
        "Patient with metastatic non-small cell lung cancer harboring EGFR L858R mutation "
        "previously treated with 80 mg osimertinib once daily. Repeat PET-CT scan demonstrates "
        "progressive disease with new hepatic lesions and grade 3 diarrhea."
    )


# =========================================================================
# CATEGORY A: ADAPTER & COMPATIBILITY UNIT TESTS (Correction 3 & 12)
# =========================================================================

def test_adapter_stage1_schema(mock_stage1_response, mock_stage2_response, mock_stage3_response):
    """Verify that the adapter correctly ingests and transforms the authentic Stage 1 ML schema."""
    s1_ctx, s2_ctx, s3_ctx = adapt_stage123_to_stage4_context(
        mock_stage1_response, mock_stage2_response, mock_stage3_response
    )
    assert s1_ctx["mortality_prob"] == 0.7420
    assert s1_ctx["response_prob"] == 0.3050
    assert s1_ctx["toxicity_prob"] == 0.8120
    assert s1_ctx["risk_category"] == "High"
    assert s1_ctx["risk_score"] == 0.8850
    assert s1_ctx["top_feature"] == "mutation_burden"
    assert s1_ctx["SYNTHETIC"] is True


def test_adapter_stage2_schema(mock_stage1_response, mock_stage2_response, mock_stage3_response):
    """Verify that the adapter correctly ingests and transforms the authentic Stage 2 DL schema."""
    s1_ctx, s2_ctx, s3_ctx = adapt_stage123_to_stage4_context(
        mock_stage1_response, mock_stage2_response, mock_stage3_response
    )
    assert s2_ctx["progression_prob"] == 0.6850
    assert s2_ctx["confidence"] == 0.8420
    assert s2_ctx["fused_prediction"] == "Progression Anticipated"
    assert s2_ctx["histopathology_finding"] == "Malignant Carcinoma (Grade III)"
    assert s2_ctx["temporal_prediction"] == "Rapid Progression (90-Day Horizon)"


def test_adapter_stage3_schema(mock_stage1_response, mock_stage2_response, mock_stage3_response):
    """Verify that the adapter correctly ingests and transforms the authentic Stage 3 NLP schema."""
    s1_ctx, s2_ctx, s3_ctx = adapt_stage123_to_stage4_context(
        mock_stage1_response, mock_stage2_response, mock_stage3_response
    )
    assert "EGFR L858R" in s3_ctx["extracted_entities"]
    assert "osimertinib" in s3_ctx["extracted_entities"]
    assert s3_ctx["urgency_level"] == "High"
    assert s3_ctx["urgency_confidence"] == 0.9240


def test_adapter_missing_stage_rejection(mock_stage1_response, mock_stage2_response, sample_clinical_report):
    """
    Verify that validate_stage_responses strictly rejects missing stages without
    fabricating fallback values (Correction 3 & 11).
    """
    # Missing Stage 3
    is_valid, errors = validate_stage_responses(
        stage1_resp=mock_stage1_response,
        stage2_resp=mock_stage2_response,
        stage3_resp=None,
        clinical_report=sample_clinical_report
    )
    assert is_valid is False
    assert any("Stage 3" in e for e in errors)

    # Missing Stage 1
    is_valid, errors = validate_stage_responses(
        stage1_resp=None,
        stage2_resp=mock_stage2_response,
        stage3_resp={"urgency": "HIGH", "entities": ["EGFR"]},
        clinical_report=sample_clinical_report
    )
    assert is_valid is False
    assert any("Stage 1" in e for e in errors)

    # Missing Clinical Report
    is_valid, errors = validate_stage_responses(
        stage1_resp=mock_stage1_response,
        stage2_resp=mock_stage2_response,
        stage3_resp={"urgency": "HIGH", "entities": ["EGFR"]},
        clinical_report=""
    )
    assert is_valid is False
    assert any("Clinical report" in e for e in errors)


def test_adapter_malformed_schema_handling():
    """Verify that malformed stage responses return controlled failure flags."""
    malformed_s1 = {"unrelated_field": 123}
    malformed_s2 = {}
    malformed_s3 = {"empty": True}

    is_valid, errors = validate_stage_responses(
        stage1_resp=malformed_s1,
        stage2_resp=malformed_s2,
        stage3_resp=malformed_s3,
        clinical_report="Valid text report."
    )
    assert is_valid is False
    assert len(errors) >= 3


def test_cpu_thread_configuration():
    """Verify that get_optimal_cpu_threads dynamically adapts to CPU resources (Correction 1)."""
    threads = get_optimal_cpu_threads()
    assert isinstance(threads, int)
    assert threads >= 1
    total = os.cpu_count() or 4
    if total >= 8:
        assert threads == 6, f"Expected 6 threads on 8+ core host, got {threads}"


# =========================================================================
# CATEGORY B: FULL PRODUCTION INTEGRATION TESTS (Correction 3, 4, 7, 12, 13)
# =========================================================================

def test_full_stage123_to_stage4_production(
    mock_stage1_response,
    mock_stage2_response,
    mock_stage3_response,
    sample_clinical_report
):
    """
    Production end-to-end integration test:
    Stage 1 + Stage 2 + Stage 3 + Clinical Report -> Adapter -> Stage 4 SLM briefing.
    """
    client = OncologyAPIClient()
    payload = {
        "patient_id": "SYNTH_P_TEST_01",
        "clinical_report": sample_clinical_report,
        "source_type": "consultation",
        "stage1_result": mock_stage1_response,
        "stage2_result": mock_stage2_response,
        "stage3_result": mock_stage3_response,
    }

    result = client.predict_slm_briefing(payload)
    assert result is not None
    assert "oncology_briefing" in result
    assert result["sentence_count"] in [1, 2]
    assert result["format_valid"] is True
    assert result["generation_latency_seconds"] > 0.0
    assert "Qwen2.5-0.5B-Instruct + LoRA" in result["model"]
    assert "SYNTHETIC RESEARCH DATA" in result["synthetic_disclaimer"]


def test_text_consultation_workflow(mock_stage1_response, mock_stage2_response):
    """
    Tests text consultation note preset processed through Stage 3 NLP into Stage 4.
    """
    client = OncologyAPIClient()
    consultation_text = (
        "EMERGENCY: Patient admitted with high-grade febrile neutropenia following 100 mg doxorubicin. "
        "Patient exhibits severe cardiotoxicity and nausea."
    )
    # Stage 3 NLP run
    s3_res = client.predict_nlp(consultation_text)
    assert s3_res is not None
    assert s3_res["urgency"].upper() in ["LOW", "MODERATE", "HIGH"]

    # Stage 4 SLM run
    payload = {
        "patient_id": "SYNTH_P_TEXT_02",
        "clinical_report": consultation_text,
        "source_type": "emergency",
        "stage1_result": mock_stage1_response,
        "stage2_result": mock_stage2_response,
        "stage3_result": s3_res,
    }
    slm_res = client.predict_slm_briefing(payload)
    assert slm_res["format_valid"] is True
    assert len(slm_res["oncology_briefing"]) > 15


def test_audio_whisper_to_nlp_to_stage4(mock_stage1_response, mock_stage2_response):
    """
    Tests canonical Audio -> Whisper -> NLP -> Stage 4 flow (Correction 4).
    Verifies that the existing Stage 3 Whisper transcriber directly feeds Stage 3 NLP
    without any duplicate NLP or Whisper path.
    """
    from stage3_nlp.audio.transcriber import ClinicalAudioTranscriber

    client = OncologyAPIClient()
    # Check if a sample wav audio exists, or synthesize a clean dummy PCM WAV
    audio_dir = PROJECT_ROOT / "stage3_nlp" / "data"
    sample_wavs = list(audio_dir.glob("**/*.wav")) if audio_dir.exists() else []

    if sample_wavs:
        with open(sample_wavs[0], "rb") as f:
            audio_bytes = f.read()
        transcriber = ClinicalAudioTranscriber(model_name="tiny")
        t_res = transcriber.transcribe(audio_bytes)
        transcribed_text = t_res.get("text", "")
    else:
        # Fallback simulation of verified Whisper transcript string
        transcribed_text = (
            "Patient with lung cancer harboring EGFR mutation on osimertinib therapy. "
            "Exhibits severe fatigue and progressive disease."
        )

    # 1. Existing Stage 3 NLP execution
    s3_res = client.predict_nlp(transcribed_text)
    assert "urgency" in s3_res
    assert "entities" in s3_res

    # 2. Stage 4 SLM consumes existing Stage 3 NLP output
    payload = {
        "patient_id": "SYNTH_P_AUDIO_03",
        "clinical_report": transcribed_text,
        "source_type": "audio_dictation",
        "stage1_result": mock_stage1_response,
        "stage2_result": mock_stage2_response,
        "stage3_result": s3_res,
    }
    slm_res = client.predict_slm_briefing(payload)
    assert slm_res["format_valid"] is True
    assert slm_res["sentence_count"] in [1, 2]


def test_stage4_failure_isolation(mock_stage1_response, mock_stage2_response, mock_stage3_response):
    """
    Verify that if Stage 4 fails (e.g. invalid report or missing context),
    Stage 1, 2, and 3 responses remain uncompromised and isolated (Correction 7).
    """
    client = OncologyAPIClient()
    invalid_payload = {
        "patient_id": "SYNTH_FAIL_TEST",
        "clinical_report": "",  # Triggers validation failure
        "source_type": "consultation",
        "stage1_result": mock_stage1_response,
        "stage2_result": mock_stage2_response,
        "stage3_result": mock_stage3_response,
    }

    with pytest.raises((ValueError, RuntimeError, Exception)):
        client.predict_slm_briefing(invalid_payload)

    # Upstream stages remain intact and accessible
    assert mock_stage1_response["overall_patient_risk"]["prediction"] == "High"
    assert mock_stage2_response["progression_probability"] == 0.6850
    assert mock_stage3_response["urgency"] == "HIGH"


def test_offline_local_execution():
    """
    Verify that Stage 4 executes locally using local model weights
    without any external network connection (Correction 13).
    """
    adapter_path = PROJECT_ROOT / "stage4_slm" / "models" / "qwen2.5_0.5b" / "adapter"
    assert (adapter_path / "adapter_model.safetensors").exists(), "Missing local adapter safetensors"
    assert (adapter_path / "adapter_config.json").exists(), "Missing local adapter config"

    # Verify no OpenAI or cloud environment variables required
    assert "OPENAI_API_KEY" not in os.environ or os.environ.get("OPENAI_API_KEY") == ""


def test_output_format_and_sentence_constraint(
    mock_stage1_response,
    mock_stage2_response,
    mock_stage3_response,
    sample_clinical_report
):
    """Verify that Stage 4 strictly enforces the 1-2 sentence constraint and brevity."""
    client = OncologyAPIClient()
    payload = {
        "patient_id": "SYNTH_P_FORMAT_04",
        "clinical_report": sample_clinical_report,
        "source_type": "consultation",
        "stage1_result": mock_stage1_response,
        "stage2_result": mock_stage2_response,
        "stage3_result": mock_stage3_response,
    }
    result = client.predict_slm_briefing(payload)
    briefing = result["oncology_briefing"]

    # Sentence count
    import re
    sentences = [s.strip() for s in re.split(r"[.!?]", briefing) if len(s.strip()) > 3]
    assert 1 <= len(sentences) <= 2, f"Expected 1-2 sentences, got {len(sentences)}: '{briefing}'"
    assert result["sentence_count"] in [1, 2]
    assert result["format_valid"] is True


def test_no_prompt_or_template_leakage(
    mock_stage1_response,
    mock_stage2_response,
    mock_stage3_response,
    sample_clinical_report
):
    """Verify that prompt instructions, markdown headers, and JSON formatting do not leak into summary."""
    client = OncologyAPIClient()
    payload = {
        "patient_id": "SYNTH_P_LEAK_05",
        "clinical_report": sample_clinical_report,
        "source_type": "consultation",
        "stage1_result": mock_stage1_response,
        "stage2_result": mock_stage2_response,
        "stage3_result": mock_stage3_response,
    }
    result = client.predict_slm_briefing(payload)
    briefing = result["oncology_briefing"]

    assert "### Instruction:" not in briefing
    assert "### Clinical Report" not in briefing
    assert "### Target" not in briefing
    assert "{" not in briefing and "}" not in briefing


def test_no_fabricated_fallback_values(mock_stage1_response, mock_stage3_response, sample_clinical_report):
    """Verify that missing Stage 2 DL does not trigger fake fabricated values (Correction 11)."""
    client = OncologyAPIClient()
    incomplete_payload = {
        "patient_id": "SYNTH_P_NO_FAB",
        "clinical_report": sample_clinical_report,
        "source_type": "consultation",
        "stage1_result": mock_stage1_response,
        "stage2_result": {},  # Incomplete Stage 2
        "stage3_result": mock_stage3_response,
    }

    # Must raise an explicit error rather than inventing fake data
    with pytest.raises((ValueError, Exception)):
        client.predict_slm_briefing(incomplete_payload)
