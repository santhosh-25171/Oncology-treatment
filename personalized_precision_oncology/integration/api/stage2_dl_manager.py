import os
import io
import sys
import json
import base64
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import numpy as np

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage2_dl_manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE2_DIR = PROJECT_ROOT / "stage2_dl"

if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from src.models.cnn import BaselineCNN
from src.models.transformer import TransformerProgressionModel
from src.models.fusion import MultimodalFusionModel
from src.explainability.gradcam import GradCAM

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ["normal", "benign", "malignant", "tumor_margin", "necrotic", "inflammatory"]


class Stage2DLManager:
    """
    Centralized Model Manager for Stage 2 Deep Learning models.
    Loads models once into memory on application startup.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or STAGE2_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.cnn_model: Optional[BaselineCNN] = None
        self.transformer_model: Optional[TransformerProgressionModel] = None
        self.fusion_model: Optional[MultimodalFusionModel] = None
        self.gradcam_engine: Optional[GradCAM] = None

        # Preprocessing artifacts
        self.temporal_scaler = None
        self.imputation_dict = None
        self.feature_config = None
        self.input_features = []
        self.numeric_features = []

        # Dynamic status flags
        self.cnn_loaded: bool = False
        self.transformer_loaded: bool = False
        self.fusion_loaded: bool = False
        self.temporal_prep_loaded: bool = False

        self.cnn_error: Optional[str] = None
        self.transformer_error: Optional[str] = None
        self.fusion_error: Optional[str] = None

        self._load_all()

    def _load_all(self):
        self._load_temporal_preprocessing()
        self._load_cnn()
        self._load_transformer()
        self._load_fusion()

    def _load_temporal_preprocessing(self):
        prep_dir = self.base_dir / "artifacts" / "models" / "temporal_preprocessing"
        try:
            scaler_path = prep_dir / "temporal_scaler.pkl"
            impute_path = prep_dir / "imputation_dict.pkl"
            config_path = prep_dir / "feature_config.json"

            if not scaler_path.exists() or not impute_path.exists() or not config_path.exists():
                raise FileNotFoundError(f"Missing temporal preprocessing artifacts in {prep_dir}")

            with open(scaler_path, "rb") as f:
                self.temporal_scaler = pickle.load(f)
            with open(impute_path, "rb") as f:
                self.imputation_dict = pickle.load(f)
            with open(config_path, "r") as f:
                self.feature_config = json.load(f)

            self.input_features = self.feature_config["input_features"]
            self.numeric_features = self.feature_config["numeric"]
            self.temporal_prep_loaded = True
            logger.info("Stage 2 temporal preprocessing artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load temporal preprocessing artifacts: {e}")
            self.temporal_prep_loaded = False

    def _load_cnn(self):
        ckpt_path = self.base_dir / "artifacts" / "models" / "cnn_best.pt"
        try:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"CNN checkpoint not found at {ckpt_path}")

            model = BaselineCNN(num_classes=6).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()

            self.cnn_model = model
            self.gradcam_engine = GradCAM(self.cnn_model, self.cnn_model.block3[0])
            self.cnn_loaded = True
            logger.info(f"Stage 2 CNN loaded successfully from {ckpt_path.name}.")
        except Exception as e:
            self.cnn_loaded = False
            self.cnn_error = str(e)
            logger.error(f"Failed to load Stage 2 CNN: {e}")

    def _load_transformer(self):
        ckpt_path = self.base_dir / "artifacts" / "models" / "transformer_best.pt"
        try:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Transformer checkpoint not found at {ckpt_path}")

            input_size = len(self.input_features) if self.input_features else 30
            model = TransformerProgressionModel(
                input_size=input_size,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.2,
                num_classes=2
            ).to(self.device)

            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()

            self.transformer_model = model
            self.transformer_loaded = True
            logger.info(f"Stage 2 Transformer loaded successfully from {ckpt_path.name}.")
        except Exception as e:
            self.transformer_loaded = False
            self.transformer_error = str(e)
            logger.error(f"Failed to load Stage 2 Transformer: {e}")

    def _load_fusion(self):
        ckpt_path = self.base_dir / "artifacts" / "models" / "fusion" / "multimodal_fusion_best.pt"
        try:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Fusion checkpoint not found at {ckpt_path}")
            if not self.cnn_loaded or not self.transformer_loaded:
                raise RuntimeError("Cannot load MultimodalFusionModel: CNN or Transformer is not loaded.")

            model = MultimodalFusionModel(
                cnn_model=self.cnn_model,
                transformer_model=self.transformer_model,
                cnn_embed_dim=128,
                temporal_embed_dim=64,
                num_classes=2
            ).to(self.device)

            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()

            self.fusion_model = model
            self.fusion_loaded = True
            logger.info(f"Stage 2 Multimodal Fusion model loaded successfully from {ckpt_path.name}.")
        except Exception as e:
            self.fusion_loaded = False
            self.fusion_error = str(e)
            logger.error(f"Failed to load Stage 2 Multimodal Fusion: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        all_ready = self.cnn_loaded and self.transformer_loaded and self.fusion_loaded
        return {
            "status": "ok" if all_ready else "degraded",
            "stage2_dl": True,
            "cnn_loaded": self.cnn_loaded,
            "transformer_loaded": self.transformer_loaded,
            "fusion_loaded": self.fusion_loaded,
            "temporal_prep_loaded": self.temporal_prep_loaded,
            "errors": {
                k: v for k, v in {
                    "cnn": self.cnn_error,
                    "transformer": self.transformer_error,
                    "fusion": self.fusion_error
                }.items() if v is not None
            }
        }

    def preprocess_image(self, image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupt image format: {e}")

        pil_resized = pil_img.resize((224, 224))
        img_np = np.array(pil_resized)
        tensor = IMAGE_TRANSFORM(pil_resized).unsqueeze(0).to(self.device)
        return tensor, img_np

    def predict_image(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self.cnn_loaded or self.cnn_model is None:
            raise RuntimeError(f"CNN model unavailable: {self.cnn_error or 'Not loaded'}")

        tensor, img_np = self.preprocess_image(image_bytes)

        with torch.no_grad():
            logits = self.cnn_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = CLASS_NAMES[pred_idx]

        class_probabilities = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }

        gradcam_available = False
        gradcam_overlay_b64 = None
        gradcam_message = None

        if self.gradcam_engine is not None:
            try:
                heatmap, _ = self.gradcam_engine.generate_heatmap(tensor, class_idx=pred_idx)
                overlay = GradCAM.overlay_heatmap(img_np, heatmap, alpha=0.5, colormap="jet")
                overlay_pil = Image.fromarray(overlay)
                buffer = io.BytesIO()
                overlay_pil.save(buffer, format="PNG")
                gradcam_overlay_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                gradcam_available = True
            except Exception as e:
                logger.warning(f"Grad-CAM generation failed: {e}")
                gradcam_message = f"Grad-CAM could not be generated: {str(e)}"
        else:
            gradcam_message = "Grad-CAM engine not initialized."

        response = {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "class_probabilities": {k: round(v, 4) for k, v in class_probabilities.items()},
            "gradcam_available": gradcam_available,
            "gradcam_overlay": gradcam_overlay_b64
        }
        if gradcam_message:
            response["message"] = gradcam_message

        return response

    def preprocess_temporal_sequence(self, records: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not records:
            raise ValueError("Longitudinal records list cannot be empty.")

        df = pd.DataFrame(records)

        if "study_day" in df.columns:
            df = df.sort_values(by="study_day").reset_index(drop=True)
        elif "timestep" in df.columns:
            df = df.sort_values(by="timestep").reset_index(drop=True)

        if self.imputation_dict:
            for col, val in self.imputation_dict.items():
                if col in df.columns:
                    df[col] = df[col].fillna(val)

        cat_cols = ["treatment_status", "treatment_cycle", "treatment_type", "response_status"]
        present_cats = [c for c in cat_cols if c in df.columns]
        if present_cats:
            df = pd.get_dummies(df, columns=present_cats, dummy_na=False)

        for col in self.input_features:
            if col not in df.columns:
                df[col] = 0.0

        if self.temporal_scaler and self.numeric_features:
            numeric_present = [col for col in self.numeric_features if col in df.columns]
            df[numeric_present] = self.temporal_scaler.transform(df[numeric_present])

        feature_matrix = df[self.input_features].values.astype(np.float32)
        seq_len = len(feature_matrix)

        x_tensor = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        length_tensor = torch.tensor([seq_len], dtype=torch.long).to(self.device)

        return x_tensor, length_tensor

    def predict_trajectory(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.transformer_loaded or self.transformer_model is None:
            raise RuntimeError(f"Transformer model unavailable: {self.transformer_error or 'Not loaded'}")

        x_tensor, length_tensor = self.preprocess_temporal_sequence(records)

        with torch.no_grad():
            logits = self.transformer_model(x_tensor, length_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_no_prog = float(probs[0])
        prob_prog = float(probs[1])

        prediction_label = "Progression" if prob_prog >= 0.5 else "No Progression (Stable)"
        confidence = prob_prog if prob_prog >= 0.5 else prob_no_prog

        return {
            "prediction": prediction_label,
            "progression_probability": round(prob_prog, 4),
            "confidence": round(confidence, 4),
            "probabilities": {
                "No Progression (Stable)": round(prob_no_prog, 4),
                "Progression": round(prob_prog, 4)
            },
            "sequence_length": int(length_tensor.item())
        }

    def predict_multimodal(self, image_bytes: bytes, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.fusion_loaded or self.fusion_model is None:
            raise RuntimeError(f"Multimodal Fusion model unavailable: {self.fusion_error or 'Not loaded'}")

        img_tensor, _ = self.preprocess_image(image_bytes)
        x_tensor, length_tensor = self.preprocess_temporal_sequence(records)

        images_list = [img_tensor.squeeze(0).unsqueeze(0)]

        with torch.no_grad():
            fusion_logits = self.fusion_model(images_list, x_tensor, length_tensor)
            fusion_probs = torch.softmax(fusion_logits, dim=1).squeeze(0).cpu().numpy()

            cnn_logits = self.cnn_model(img_tensor)
            cnn_probs = torch.softmax(cnn_logits, dim=1).squeeze(0).cpu().numpy()
            img_pred = CLASS_NAMES[int(np.argmax(cnn_probs))]

            tx_logits = self.transformer_model(x_tensor, length_tensor)
            tx_probs = torch.softmax(tx_logits, dim=1).squeeze(0).cpu().numpy()
            temp_pred = "Progression" if tx_probs[1] >= 0.5 else "No Progression (Stable)"

        prob_no_prog = float(fusion_probs[0])
        prob_prog = float(fusion_probs[1])

        prediction_label = "Progression" if prob_prog >= 0.5 else "No Progression (Stable)"
        confidence = prob_prog if prob_prog >= 0.5 else prob_no_prog

        return {
            "prediction": prediction_label,
            "progression_probability": round(prob_prog, 4),
            "confidence": round(confidence, 4),
            "modality": "multimodal",
            "image_prediction": img_pred,
            "image_confidence": round(float(np.max(cnn_probs)), 4),
            "temporal_prediction": temp_pred,
            "temporal_confidence": round(float(np.max(tx_probs)), 4),
            "probabilities": {
                "No Progression (Stable)": round(prob_no_prog, 4),
                "Progression": round(prob_prog, 4)
            }
        }
