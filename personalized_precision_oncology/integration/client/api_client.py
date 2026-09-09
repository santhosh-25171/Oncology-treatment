import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

logger = logging.getLogger("oncology_api_client")

# Centralized API Base URL configuration
DL_API_URL = os.environ.get("DL_API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT = 30  # seconds

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class OncologyAPIClient:
    """
    Centralized, reusable API Client for the Precision Oncology FastAPI backend.
    Enforces unified inference across Stage 1 ML, Stage 2 DL, and Stage 3 NLP,
    with local Python engine fallback when FastAPI service is unreachable.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = (base_url or DL_API_URL).rstrip("/")
        self.timeout = timeout
        self._local_stage1 = None
        self._local_stage2 = None
        self._local_stage3 = None

    def health_check(self) -> Dict[str, Any]:
        """Queries GET /health and returns backend runtime status, falling back to local inspection."""
        url = f"{self.base_url}/health"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                data["backend_type"] = "FastAPI (HTTP)"
                return data
            return {
                "status": "degraded",
                "backend_type": "FastAPI (Degraded)",
                "stage1_ml": False,
                "stage2_dl": False,
                "stage3_nlp": False,
                "detail": f"HTTP {resp.status_code}: {resp.text}"
            }
        except requests.exceptions.ConnectionError:
            # Fallback: check local model availability
            local_status = self._check_local_health()
            local_status["backend_type"] = "Local Python Engine (Offline API)"
            local_status["detail"] = f"FastAPI server unreachable at {self.base_url}. Using local Python engine."
            return local_status
        except Exception as e:
            return {
                "status": "error",
                "backend_type": "Unavailable",
                "stage1_ml": False,
                "stage2_dl": False,
                "stage3_nlp": False,
                "detail": str(e)
            }

    def _check_local_health(self) -> Dict[str, Any]:
        """Inspects whether local Python models can be loaded directly."""
        s1_ok = False
        s2_ok = False
        s3_ok = False
        cnn_ok = False
        tf_ok = False
        fus_ok = False
        nlp_ok = False

        try:
            s1_path = PROJECT_ROOT / "data" / "stage1_ml" / "models" / "best_model.json"
            s1_ok = s1_path.exists()
        except Exception:
            pass

        try:
            s2_cnn = PROJECT_ROOT / "stage2_dl" / "artifacts" / "models" / "cnn_best.pt"
            s2_tf = PROJECT_ROOT / "stage2_dl" / "artifacts" / "models" / "transformer_best.pt"
            s2_fus = PROJECT_ROOT / "stage2_dl" / "artifacts" / "models" / "fusion" / "multimodal_fusion_best.pt"
            cnn_ok = s2_cnn.exists()
            tf_ok = s2_tf.exists()
            fus_ok = s2_fus.exists()
            s2_ok = cnn_ok and tf_ok and fus_ok
        except Exception:
            pass

        try:
            s3_clf = PROJECT_ROOT / "stage3_nlp" / "models" / "classification" / "urgency_model.pkl"
            s3_ner = PROJECT_ROOT / "stage3_nlp" / "models" / "ner" / "config.cfg"
            nlp_ok = s3_clf.exists() and s3_ner.exists()
            s3_ok = nlp_ok
        except Exception:
            pass

        return {
            "status": "ok" if (s1_ok and s2_ok and s3_ok) else "degraded",
            "service": "precision-oncology-local",
            "stage1_ml": s1_ok,
            "stage2_dl": s2_ok,
            "stage3_nlp": s3_ok,
            "cnn_loaded": cnn_ok,
            "transformer_loaded": tf_ok,
            "fusion_loaded": fus_ok,
            "nlp_loaded": nlp_ok,
            "temporal_prep_loaded": True,
            "version": "2.1.0-local"
        }

    # =========================================================================
    # STAGE 1 METHODS
    # =========================================================================

    def predict_stage1_risk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calls POST /predict for Stage 1 ML tabular patient profile."""
        url = f"{self.base_url}/predict"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                res = resp.json()
                res["backend"] = "FastAPI"
                return res
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            # Fallback to local stage1 pipeline
            return self._predict_stage1_local(payload)
        except Exception as e:
            raise RuntimeError(str(e))

    def predict_stage1(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Standard alias for predict_stage1_risk."""
        return self.predict_stage1_risk(payload)

    def _predict_stage1_local(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Local direct execution fallback for Stage 1."""
        try:
            if self._local_stage1 is None:
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                from stage1_ml.prediction.prediction import OncologyPredictionPipeline
                self._local_stage1 = OncologyPredictionPipeline(base_dir=str(PROJECT_ROOT))

            raw_result = self._local_stage1.predict(payload)
            ov = raw_result["overall_patient_risk"]
            tox = raw_result["toxicity_risk"]
            ther = raw_result["therapy_response"]

            return {
                "overall_patient_risk": {
                    "prediction": ov["prediction"],
                    "risk_probability": ov["risk_probability"],
                    "threshold": ov.get("threshold", 0.48),
                    "confidence": ov["confidence"],
                    "probabilities": ov["probabilities"],
                    "important_factors": ov["important_factors"],
                    "debug_info": ov.get("debug_info", {})
                },
                "toxicity_risk": {
                    "prediction": tox["prediction"],
                    "confidence": tox["confidence"],
                    "probabilities": tox["probabilities"]
                },
                "therapy_response": {
                    "prediction": ther["prediction"],
                    "confidence": ther["confidence"],
                    "probabilities": ther["probabilities"]
                },
                "risk_score": ov["risk_probability"],
                "risk_class": ov["prediction"],
                "threshold": ov.get("threshold", 0.48),
                "probabilities": ov["probabilities"],
                "top_contributing_biomarkers": ov["important_factors"],
                "debug_info": ov.get("debug_info", {}),
                "backend": "Local Python Engine"
            }
        except Exception as ex:
            raise ConnectionError(
                f"FastAPI is offline and local Stage 1 engine failed: {ex}"
            )

    # =========================================================================
    # STAGE 2 METHODS
    # =========================================================================

    def predict_image(self, image_bytes: bytes, filename: str = "patch.jpg") -> Dict[str, Any]:
        """
        Calls POST /predict-image for 6-class histopathology classification + Grad-CAM heatmap.
        """
        url = f"{self.base_url}/predict-image"
        files = {
            "file": (filename, image_bytes, "image/jpeg")
        }
        try:
            resp = requests.post(url, files=files, timeout=self.timeout)
            if resp.status_code == 200:
                res = resp.json()
                res["backend"] = "FastAPI"
                return res
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            # Fallback to local stage 2 manager
            return self._predict_image_local(image_bytes, filename)
        except Exception as e:
            raise RuntimeError(str(e))

    def _predict_image_local(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        try:
            if self._local_stage2 is None:
                from integration.api.stage2_dl_manager import Stage2DLManager
                self._local_stage2 = Stage2DLManager()
            res = self._local_stage2.predict_image(image_bytes)
            res["backend"] = "Local Python Engine"
            return res
        except Exception as ex:
            raise ConnectionError(f"FastAPI is offline and local DL engine failed: {ex}")

    def predict_trajectory(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calls POST /predict-trajectory for Transformer longitudinal progression prediction.
        """
        url = f"{self.base_url}/predict-trajectory"
        payload = {"records": records}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                res = resp.json()
                res["backend"] = "FastAPI"
                return res
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            return self._predict_trajectory_local(records)
        except Exception as e:
            raise RuntimeError(str(e))

    def _predict_trajectory_local(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if self._local_stage2 is None:
                from integration.api.stage2_dl_manager import Stage2DLManager
                self._local_stage2 = Stage2DLManager()
            res = self._local_stage2.predict_trajectory(records)
            res["backend"] = "Local Python Engine"
            return res
        except Exception as ex:
            raise ConnectionError(f"FastAPI is offline and local DL engine failed: {ex}")

    def predict_multimodal(
        self,
        image_bytes: bytes,
        records: List[Dict[str, Any]],
        image_filename: str = "biopsy.jpg"
    ) -> Dict[str, Any]:
        """
        Calls POST /predict-multimodal for joint spatial pathology + longitudinal biomarker fusion.
        """
        url = f"{self.base_url}/predict-multimodal"
        files = {
            "file": (image_filename, image_bytes, "image/jpeg")
        }
        data = {
            "temporal_data": json.dumps(records)
        }
        try:
            resp = requests.post(url, files=files, data=data, timeout=self.timeout)
            if resp.status_code == 200:
                res = resp.json()
                res["backend"] = "FastAPI"
                return res
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            return self._predict_multimodal_local(image_bytes, records)
        except Exception as e:
            raise RuntimeError(str(e))

    def _predict_multimodal_local(self, image_bytes: bytes, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if self._local_stage2 is None:
                from integration.api.stage2_dl_manager import Stage2DLManager
                self._local_stage2 = Stage2DLManager()
            res = self._local_stage2.predict_multimodal(image_bytes, records)
            res["backend"] = "Local Python Engine"
            return res
        except Exception as ex:
            raise ConnectionError(f"FastAPI is offline and local DL engine failed: {ex}")

    # =========================================================================
    # STAGE 3 NLP METHODS
    # =========================================================================

    def predict_nlp(self, text: str) -> Dict[str, Any]:
        """
        Calls POST /api/v1/nlp/predict for unified clinical urgency triage and entity extraction.
        """
        url = f"{self.base_url}/api/v1/nlp/predict"
        payload = {"text": text}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                res = resp.json()
                res["backend"] = "FastAPI"
                return res
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            return self._predict_nlp_local(text)
        except Exception as e:
            raise RuntimeError(str(e))

    def predict_urgency(self, text: str) -> Dict[str, Any]:
        """
        Calls POST /api/v1/nlp/urgency for clinical urgency triage and probabilities.
        """
        url = f"{self.base_url}/api/v1/nlp/urgency"
        payload = {"text": text}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                res = resp.json()
                res["backend"] = "FastAPI"
                return res
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            return self._predict_urgency_local(text)
        except Exception as e:
            raise RuntimeError(str(e))

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Calls POST /api/v1/nlp/ner for clinical oncology entity extraction.
        """
        url = f"{self.base_url}/api/v1/nlp/ner"
        payload = {"text": text}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                res = resp.json()
                res["backend"] = "FastAPI"
                return res
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            return self._extract_entities_local(text)
        except Exception as e:
            raise RuntimeError(str(e))

    def _get_local_stage3(self):
        if self._local_stage3 is None:
            from integration.api.stage3_nlp_manager import Stage3NLPManager
            self._local_stage3 = Stage3NLPManager()
        return self._local_stage3

    def _predict_nlp_local(self, text: str) -> Dict[str, Any]:
        try:
            m = self._get_local_stage3()
            res = m.predict_nlp(text)
            res["backend"] = "Local Python Engine"
            return res
        except Exception as ex:
            raise ConnectionError(f"FastAPI is offline and local NLP engine failed: {ex}")

    def _predict_urgency_local(self, text: str) -> Dict[str, Any]:
        try:
            m = self._get_local_stage3()
            res = m.predict_urgency(text)
            res["backend"] = "Local Python Engine"
            return res
        except Exception as ex:
            raise ConnectionError(f"FastAPI is offline and local NLP engine failed: {ex}")

    def _extract_entities_local(self, text: str) -> Dict[str, Any]:
        try:
            m = self._get_local_stage3()
            res = m.extract_entities(text)
            res["backend"] = "Local Python Engine"
            return res
        except Exception as ex:
            raise ConnectionError(f"FastAPI is offline and local NLP engine failed: {ex}")


# Default singleton instance
api_client = OncologyAPIClient()
