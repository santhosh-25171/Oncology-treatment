import io
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import pytest

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage3_nlp.audio.transcriber import ClinicalAudioTranscriber, RESEARCH_ASR_DISCLAIMER
from integration.client.api_client import OncologyAPIClient, DL_API_URL


def create_test_wav(duration_sec: float = 1.0, freq_hz: float = 440.0, sample_rate: int = 16000, silence: bool = False) -> bytes:
    """Helper to synthesize test WAV byte streams in memory without writing to disk."""
    buf = io.BytesIO()
    total_samples = int(sample_rate * duration_sec)
    if silence:
        data = np.zeros(total_samples, dtype=np.int16)
    else:
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)
        data = (np.sin(2 * np.pi * freq_hz * t) * 12000).astype(np.int16)
    wavfile.write(buf, sample_rate, data)
    return buf.getvalue()


# =========================================================================
# 1. Existing Text Workflow Preservation Tests
# =========================================================================

def test_existing_text_pipeline_intact():
    """Verify that existing Stage 3 text NLP pipeline remains 100% functional."""
    client = OncologyAPIClient(base_url=DL_API_URL)
    clinical_note = (
        "EMERGENCY: Patient admitted with severe febrile neutropenia and acute cardiotoxicity "
        "following administration of 100 mg doxorubicin."
    )
    result = client.predict_nlp(clinical_note)

    assert result is not None
    assert "urgency" in result
    assert result["urgency"] in ["LOW", "MODERATE", "HIGH"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "entities" in result
    assert isinstance(result["entities"], list)
    assert len(result["entities"]) > 0


# =========================================================================
# 2. Audio Transcriber Unit & Edge-Case Tests
# =========================================================================

def test_transcriber_initialization():
    """Verify ClinicalAudioTranscriber initializes with correct default model."""
    transcriber = ClinicalAudioTranscriber(model_name="tiny")
    assert transcriber.model_name == "tiny"
    assert transcriber._model is None  # Lazy loading


def test_transcriber_empty_input():
    """Verify transcriber handles empty bytes safely without unhandled crashes."""
    transcriber = ClinicalAudioTranscriber(model_name="tiny")
    res = transcriber.transcribe(b"")
    assert res["success"] is False
    assert res["text"] == ""
    assert "error" in res
    assert "disclaimer" in res


def test_transcriber_corrupted_audio():
    """Verify transcriber handles corrupted/malformed WAV bytes gracefully."""
    transcriber = ClinicalAudioTranscriber(model_name="tiny")
    corrupted_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00NOT_VALID_AUDIO"
    res = transcriber.transcribe(corrupted_bytes)
    assert res["success"] is False
    assert res["text"] == ""
    assert "error" in res


def test_transcriber_silent_audio():
    """Verify transcriber detects silence and returns an informative notice."""
    transcriber = ClinicalAudioTranscriber(model_name="tiny")
    silent_wav = create_test_wav(duration_sec=1.0, silence=True)
    res = transcriber.transcribe(silent_wav)
    assert res["success"] is False
    assert res["text"] == ""
    assert "silent" in res.get("error", "").lower()


def test_transcriber_audio_decoding():
    """Verify decode_wav properly converts non-16kHz stereo WAV to 16kHz mono float32."""
    transcriber = ClinicalAudioTranscriber(model_name="tiny")
    # Generate 44.1kHz stereo audio
    sample_rate = 44100
    duration = 0.5
    total_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    stereo_data = np.column_stack([
        (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16),
        (np.cos(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    ])
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, stereo_data)

    sr, audio_float = transcriber.decode_wav(buf.getvalue())
    assert sr == 16000
    assert audio_float.dtype == np.float32
    assert audio_float.ndim == 1
    assert np.max(np.abs(audio_float)) <= 1.0


def test_transcriber_tone_speech_recognition():
    """Verify non-silent tone audio runs through Whisper and handles absence of speech."""
    transcriber = ClinicalAudioTranscriber(model_name="tiny")
    tone_wav = create_test_wav(duration_sec=1.0, freq_hz=440.0)
    res = transcriber.transcribe(tone_wav)
    # A pure sine tone has no words; Whisper correctly returns no words detected
    assert "success" in res
    assert "disclaimer" in res
    assert res["disclaimer"] == RESEARCH_ASR_DISCLAIMER


# =========================================================================
# 3. Full Audio -> Text -> NLP Pipeline Convergence Tests
# =========================================================================

def test_transcription_to_nlp_convergence():
    """Verify that transcribed clinical text routes seamlessly through the NLP triage pipeline."""
    # Simulate transcription output from dictation
    transcribed_text = (
        "Patient presents with progressive metastatic melanoma harboring BRAF V600E mutation. "
        "Recommend starting dabrafenib 150 mg and trametinib 2 mg daily."
    )
    client = OncologyAPIClient(base_url=DL_API_URL)
    nlp_out = client.predict_nlp(transcribed_text)

    assert nlp_out is not None
    assert nlp_out["urgency"] in ["LOW", "MODERATE", "HIGH"]
    assert 0.0 <= nlp_out["confidence"] <= 1.0
    assert "probabilities" in nlp_out
    assert "entities" in nlp_out

    # Check extracted oncology entities from transcribed text
    entity_texts = [e["text"].lower() for e in nlp_out.get("entities", [])]
    entity_labels = [e["label"] for e in nlp_out.get("entities", [])]
    assert any("braf" in t or "v600e" in t for t in entity_texts) or "GENE_MUTATION" in entity_labels


def test_user_edited_transcription_to_nlp():
    """Verify that a user can edit the transcribed note before NLP analysis."""
    raw_transcription = "Patient experiencing severe chest pain."
    user_edited_note = raw_transcription + " Immediate ECG and troponin ordered. Suspected cardiotoxicity from doxorubicin."

    client = OncologyAPIClient(base_url=DL_API_URL)
    nlp_out = client.predict_nlp(user_edited_note)

    assert nlp_out is not None
    assert nlp_out["urgency"] in ["MODERATE", "HIGH"]
    entity_texts = [e["text"].lower() for e in nlp_out.get("entities", [])]
    assert any("doxorubicin" in t for t in entity_texts)


# =========================================================================
# 4. UI Code Structure and Disclaimer Verification
# =========================================================================

def test_dashboard_code_contains_audio_elements():
    """Verify app.py includes dual input methods, audio uploader/recorder, and disclaimer."""
    app_path = PROJECT_ROOT / "integration" / "dashboard" / "app.py"
    assert app_path.exists()
    content = app_path.read_text(encoding="utf-8")

    # Verify input toggle
    assert "Select Input Method:" in content
    assert "📝 Text Input" in content
    assert "🎙️ Audio Input (Speech-to-Text)" in content

    # Verify audio components
    assert "Upload Clinical Audio (.wav)" in content
    assert "Record Clinical Dictation" in content
    assert "⚡ Transcribe Audio" in content
    assert "Review & Edit Transcribed Clinical Note:" in content
    assert "🚀 Analyze Transcribed Note (NLP)" in content

    # Verify disclaimer
    assert "Research Prototype: Speech-to-text transcription is an assistive input interface" in content
