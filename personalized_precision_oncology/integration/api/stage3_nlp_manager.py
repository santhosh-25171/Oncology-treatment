import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("stage3_nlp_manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE3_DIR = PROJECT_ROOT / "stage3_nlp"

if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from stage3_nlp.models.predict import NLPPipeline


class Stage3NLPManager:
    """
    Centralized Model Manager for Stage 3 Clinical NLP models.
    Pre-loads TF-IDF + Logistic Regression classifier and spaCy NER model into memory.
    Provides inference for Urgency Triage and Clinical Oncology Named Entity Recognition.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or STAGE3_DIR
        self.pipeline: Optional[NLPPipeline] = None

        self.nlp_loaded: bool = False
        self.classifier_loaded: bool = False
        self.ner_loaded: bool = False
        self.error: Optional[str] = None

        self._load_models()

    def _load_models(self):
        """Loads classifier and spaCy NER pipeline."""
        try:
            self.pipeline = NLPPipeline()
            self.pipeline.load_models()

            if self.pipeline.clf_pipeline is not None:
                self.classifier_loaded = True
            if self.pipeline.ner_model is not None:
                self.ner_loaded = True

            self.nlp_loaded = self.classifier_loaded and self.ner_loaded
            logger.info("Stage 3 NLP models successfully loaded into memory.")
        except Exception as e:
            self.error = str(e)
            logger.error(f"Failed to load Stage 3 NLP models: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Returns component health status dictionary."""
        return {
            "status": "ok" if self.nlp_loaded else "degraded",
            "stage3_nlp": self.nlp_loaded,
            "classifier_loaded": self.classifier_loaded,
            "ner_loaded": self.ner_loaded,
            "error": self.error
        }

    def predict_urgency(self, text: str) -> Dict[str, Any]:
        """
        Classifies clinical note urgency (LOW, MODERATE, HIGH)
        with real model probability distribution and latency.
        """
        if not self.nlp_loaded or self.pipeline is None or self.pipeline.clf_pipeline is None:
            raise RuntimeError(f"Urgency classification model is not available: {self.error}")

        if not text or not text.strip():
            return {
                "urgency": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {"LOW": 0.0, "MODERATE": 0.0, "HIGH": 0.0},
                "execution_time_ms": 0.0
            }

        start_time = time.perf_counter()
        clf = self.pipeline.clf_pipeline

        # Predict class
        pred_class = clf.predict([text])[0]

        # Calculate exact probabilities if supported
        probabilities = {}
        confidence = 1.0
        if hasattr(clf, "predict_proba") and hasattr(clf, "classes_"):
            probs = clf.predict_proba([text])[0]
            classes = clf.classes_
            for cls_name, p in zip(classes, probs):
                probabilities[str(cls_name)] = float(p)
            confidence = float(max(probs))
        else:
            probabilities = {str(pred_class): 1.0}

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "urgency": str(pred_class),
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "execution_time_ms": round(elapsed_ms, 2)
        }

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extracts named entities (GENE_MUTATION, DRUG_NAME, DOSAGE_LEVEL, ADVERSE_EVENT)
        with exact character spans and execution latency.
        """
        if not self.nlp_loaded or self.pipeline is None or self.pipeline.ner_model is None:
            raise RuntimeError(f"Clinical NER model is not available: {self.error}")

        if not text or not text.strip():
            return {
                "entities": [],
                "entity_counts": {
                    "GENE_MUTATION": 0,
                    "DRUG_NAME": 0,
                    "DOSAGE_LEVEL": 0,
                    "ADVERSE_EVENT": 0
                },
                "total_entities": 0,
                "execution_time_ms": 0.0
            }

        start_time = time.perf_counter()
        doc = self.pipeline.ner_model(text)

        entities = []
        entity_counts = {
            "GENE_MUTATION": 0,
            "DRUG_NAME": 0,
            "DOSAGE_LEVEL": 0,
            "ADVERSE_EVENT": 0
        }

        for ent in doc.ents:
            label = ent.label_
            entities.append({
                "text": ent.text,
                "label": label,
                "start": int(ent.start_char),
                "end": int(ent.end_char)
            })
            if label in entity_counts:
                entity_counts[label] += 1
            else:
                entity_counts[label] = 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "entities": entities,
            "entity_counts": entity_counts,
            "total_entities": len(entities),
            "execution_time_ms": round(elapsed_ms, 2)
        }

    def predict_nlp(self, text: str) -> Dict[str, Any]:
        """
        Unified NLP analysis combining Urgency classification and Clinical NER.
        """
        if not text or not text.strip():
            return {
                "urgency": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {"LOW": 0.0, "MODERATE": 0.0, "HIGH": 0.0},
                "entities": [],
                "entity_counts": {
                    "GENE_MUTATION": 0,
                    "DRUG_NAME": 0,
                    "DOSAGE_LEVEL": 0,
                    "ADVERSE_EVENT": 0
                },
                "total_entities": 0,
                "execution_time_ms": 0.0
            }

        start_time = time.perf_counter()
        urgency_result = self.predict_urgency(text)
        ner_result = self.extract_entities(text)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "urgency": urgency_result["urgency"],
            "confidence": urgency_result["confidence"],
            "probabilities": urgency_result["probabilities"],
            "entities": ner_result["entities"],
            "entity_counts": ner_result["entity_counts"],
            "total_entities": ner_result["total_entities"],
            "execution_time_ms": round(elapsed_ms, 2)
        }
