import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import requests

logger = logging.getLogger("oncology_api_client")

# Centralized API Base URL configuration
DL_API_URL = os.environ.get("DL_API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT = 30 # seconds


class OncologyAPIClient:
    """
    Centralized, reusable API Client for the Precision Oncology FastAPI backend.
    Enforces single inference path through the microservice.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = (base_url or DL_API_URL).rstrip("/")
        self.timeout = timeout

    def health_check(self) -> Dict[str, Any]:
        """Queries GET /health and returns backend runtime status."""
        url = f"{self.base_url}/health"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {
                "status": "degraded",
                "stage1_ml": False,
                "stage2_dl": False,
                "detail": f"HTTP {resp.status_code}: {resp.text}"
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "unavailable",
                "stage1_ml": False,
                "stage2_dl": False,
                "detail": "Could not connect to FastAPI server. Please verify it is running."
            }
        except Exception as e:
            return {
                "status": "error",
                "stage1_ml": False,
                "stage2_dl": False,
                "detail": str(e)
            }

    def predict_stage1_risk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calls POST /predict for Stage 1 ML tabular patient profile."""
        url = f"{self.base_url}/predict"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Deep Learning & Clinical API is unavailable. Please ensure the FastAPI backend is running on "
                f"{self.base_url}."
            )
        except Exception as e:
            raise RuntimeError(str(e))

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
                return resp.json()
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Deep Learning API is unavailable. Please ensure the FastAPI backend is running on "
                f"{self.base_url}."
            )
        except Exception as e:
            raise RuntimeError(str(e))

    def predict_trajectory(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calls POST /predict-trajectory for Transformer longitudinal progression prediction.
        """
        url = f"{self.base_url}/predict-trajectory"
        payload = {"records": records}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Deep Learning API is unavailable. Please ensure the FastAPI backend is running on "
                f"{self.base_url}."
            )
        except Exception as e:
            raise RuntimeError(str(e))

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
                return resp.json()
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {error_msg}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Deep Learning API is unavailable. Please ensure the FastAPI backend is running on "
                f"{self.base_url}."
            )
        except Exception as e:
            raise RuntimeError(str(e))


# Default singleton instance
api_client = OncologyAPIClient()
