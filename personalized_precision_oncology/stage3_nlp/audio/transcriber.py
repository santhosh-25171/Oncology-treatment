import os
import io
import logging
from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import scipy.io.wavfile
import scipy.signal

logger = logging.getLogger("clinical_audio_transcriber")

RESEARCH_ASR_DISCLAIMER = (
    "Research/educational prototype only. Whisper speech-to-text transcription "
    "is not certified for clinical diagnostic use."
)


class ClinicalAudioTranscriber:
    """
    Modular Speech-to-Text (ASR) Transcriber for clinical consultation notes.
    Uses local OpenAI Whisper models with scipy-based WAV decoding.
    """

    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self._model = None

    def load_model(self):
        """Lazy-loads the local Whisper model in memory on CPU."""
        if self._model is None:
            try:
                import ssl
                # Allow fallback for Windows systems with custom enterprise/local CA stores
                try:
                    ssl._create_default_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass

                import whisper
                logger.info(f"Loading local Whisper ASR model '{self.model_name}' on CPU...")
                self._model = whisper.load_model(self.model_name, device="cpu")
                logger.info("Whisper ASR model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Whisper ASR model: {e}")
                raise RuntimeError(
                    "Speech-to-Text engine could not be initialized. "
                    "Please ensure the local model is accessible."
                )

    def decode_wav(self, audio_input: Union[bytes, io.BytesIO, str]) -> Tuple[int, np.ndarray]:
        """
        Decodes WAV audio into a normalized 16kHz mono float32 numpy array using scipy.
        Does NOT rely on external ffmpeg for WAV format.
        """
        if audio_input is None:
            raise ValueError("No audio data was provided.")

        # Obtain bytes buffer
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            with open(audio_input, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(audio_input, io.BytesIO):
            raw_bytes = audio_input.getvalue()
        elif isinstance(audio_input, bytes):
            raw_bytes = audio_input
        elif hasattr(audio_input, "read"):
            raw_bytes = audio_input.read()
        else:
            raise ValueError("Unsupported audio input format. Expected WAV bytes or file-like object.")

        if not raw_bytes or len(raw_bytes) < 44:
            raise ValueError("Audio data is empty or too short to be a valid WAV file.")

        try:
            sample_rate, data = scipy.io.wavfile.read(io.BytesIO(raw_bytes))
        except Exception as e:
            raise ValueError(f"Invalid or corrupted WAV file: {e}")

        # Normalize data to float32 in [-1.0, 1.0] based on original WAV sample type
        if data.dtype == np.int16:
            audio_f32 = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio_f32 = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            audio_f32 = (data.astype(np.float32) - 128.0) / 128.0
        elif data.dtype in (np.float32, np.float64):
            audio_f32 = data.astype(np.float32)
        else:
            max_val = np.max(np.abs(data)) or 1.0
            audio_f32 = data.astype(np.float32) / max_val

        # Convert multi-channel to mono
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.mean(axis=1)

        # Resample to 16,000 Hz if necessary (Whisper native rate)
        target_sr = 16000
        if sample_rate != target_sr and len(audio_f32) > 0:
            num_samples = int(len(audio_f32) * target_sr / sample_rate)
            audio_f32 = scipy.signal.resample(audio_f32, num_samples).astype(np.float32)
            sample_rate = target_sr

        return sample_rate, audio_f32

    def transcribe(self, audio_input: Union[bytes, io.BytesIO, str]) -> Dict[str, Any]:
        """
        Transcribes WAV audio input into clinical note text.
        
        Returns:
            Dict with keys: 'text', 'success', 'model', 'duration_sec', 'disclaimer', 'error'
        """
        try:
            sample_rate, audio_np = self.decode_wav(audio_input)

            # Check for near-silent audio
            if len(audio_np) == 0:
                return {
                    "text": "",
                    "success": False,
                    "error": "Audio contains no audio samples.",
                    "disclaimer": RESEARCH_ASR_DISCLAIMER
                }

            rms = np.sqrt(np.mean(audio_np ** 2))
            if rms < 1e-4:
                return {
                    "text": "",
                    "success": False,
                    "error": "Audio appears to be completely silent. Please record with a clearer voice.",
                    "disclaimer": RESEARCH_ASR_DISCLAIMER
                }

            # Load model and transcribe
            self.load_model()
            duration_sec = round(len(audio_np) / float(sample_rate), 2)

            # Whisper accepts 16kHz float32 numpy array directly
            result = self._model.transcribe(audio_np, language="en", fp16=False)
            transcribed_text = result.get("text", "").strip()

            if not transcribed_text:
                return {
                    "text": "",
                    "success": False,
                    "error": "Speech recognition did not detect any transcribed words.",
                    "duration_sec": duration_sec,
                    "disclaimer": RESEARCH_ASR_DISCLAIMER
                }

            return {
                "text": transcribed_text,
                "success": True,
                "model": f"Whisper ({self.model_name})",
                "duration_sec": duration_sec,
                "disclaimer": RESEARCH_ASR_DISCLAIMER,
                "error": None
            }

        except ValueError as ve:
            logger.warning(f"Audio decoding error: {ve}")
            return {
                "text": "",
                "success": False,
                "error": str(ve),
                "disclaimer": RESEARCH_ASR_DISCLAIMER
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "success": False,
                "error": f"Transcription error: {str(e)}",
                "disclaimer": RESEARCH_ASR_DISCLAIMER
            }
