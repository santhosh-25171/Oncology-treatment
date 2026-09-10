#!/usr/bin/env python3
"""
Stage 4 SLM — In-Memory Model Manager
Role: Stage 4 SLM Engineer + Integration Engineer
Project: Personalized Precision Medicine for Oncology Treatment Optimization

Manages local offline lifecycle of Qwen2.5-0.5B-Instruct + LoRA adapter:
- Dynamic CPU thread configuration (benchmarked optimal = 6)
- In-memory LoRA weight fusion via merge_and_unload()
- TwoSentenceStoppingCriteria for accelerated causal decoding
- Robust inference and error handling (never fabricates clinical data)

DISCLAIMER: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.
"""

import os
import sys
import time
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

logger = logging.getLogger("stage4_slm_manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "stage4_slm" / "models" / "qwen2.5_0.5b"
ADAPTER_DIR = MODEL_DIR / "adapter"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage4_slm.adapter.context_adapter import get_optimal_cpu_threads, build_stage4_prompt


class TwoSentenceStoppingCriteria(StoppingCriteria):
    """Halts autoregressive causal decoding immediately once two sentences are complete."""

    def __init__(self, tokenizer: AutoTokenizer, prompt_lens: List[int]):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_lens = prompt_lens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for i, p_len in enumerate(self.prompt_lens):
            gen_ids = input_ids[i, p_len:]
            if len(gen_ids) < 10:
                return False
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if text.count(".") < 2:
                return False
        return True


class Stage4SLMManager:
    """
    Centralized Model Manager for Stage 4 Small Language Model (SLM).
    Maintains frozen in-memory merged model for local offline bedside briefings.
    """

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        adapter_dir: Path = ADAPTER_DIR,
        tokenizer_dir: Path = TOKENIZER_DIR,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.base_model_id = base_model_id
        self.adapter_dir = Path(adapter_dir)
        self.tokenizer_dir = Path(tokenizer_dir)

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

        self.slm_loaded: bool = False
        self.cold_start_load_seconds: float = 0.0
        self.threads_used: int = get_optimal_cpu_threads()
        self.error: Optional[str] = None

        # Configure PyTorch CPU threads dynamically based on verified benchmark
        torch.set_num_threads(self.threads_used)

        self._load_model()

    def _load_model(self):
        """Loads base model, merges LoRA adapter in RAM, and prepares tokenizer."""
        t0 = time.time()
        try:
            tok_path = str(self.tokenizer_dir) if (self.tokenizer_dir / "tokenizer_config.json").exists() else self.base_model_id
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float32,
            )

            # Merge LoRA adapter into memory
            if (self.adapter_dir / "adapter_config.json").exists():
                logger.info(f"Merging LoRA adapter from {self.adapter_dir} into base model RAM...")
                peft_model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
                self.model = peft_model.merge_and_unload()
            else:
                logger.warning("LoRA adapter not found. Using base model directly.")
                self.model = base_model

            self.model.to(self.device)
            self.model.eval()
            self.cold_start_load_seconds = round(time.time() - t0, 2)
            self.slm_loaded = True
            logger.info(f"Stage 4 SLM loaded and merged in RAM in {self.cold_start_load_seconds}s using {self.threads_used} CPU threads.")
        except Exception as e:
            self.error = str(e)
            self.slm_loaded = False
            logger.error(f"Failed to load Stage 4 SLM: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Returns runtime health and configuration."""
        return {
            "status": "ok" if self.slm_loaded else "degraded",
            "stage4_slm": self.slm_loaded,
            "model_id": self.base_model_id,
            "adapter_path": str(self.adapter_dir),
            "cold_start_seconds": self.cold_start_load_seconds,
            "threads_configured": self.threads_used,
            "device": str(self.device),
            "error": self.error,
        }

    def generate_briefing(
        self,
        patient_id: str,
        clinical_report: str,
        stage1_context: Dict[str, Any],
        stage2_context: Dict[str, Any],
        stage3_context: Dict[str, Any],
        source_type: str = "consultation",
        max_new_tokens: int = 35,
    ) -> Dict[str, Any]:
        """
        Synthesizes a 1-2 sentence precision oncology briefing from actual multimodal context.
        Returns structured payload with timing, metrics, and compliance flags.
        """
        if not self.slm_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Stage 4 SLM is offline or uninitialized: {self.error}")

        if not clinical_report or not clinical_report.strip():
            raise ValueError("Clinical report text cannot be empty.")

        prompt = build_stage4_prompt(
            patient_id=patient_id,
            source_type=source_type,
            clinical_report=clinical_report,
            stage1_context=stage1_context,
            stage2_context=stage2_context,
            stage3_context=stage3_context,
        )

        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        stopping = StoppingCriteriaList([TwoSentenceStoppingCriteria(self.tokenizer, [prompt_len])])

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                stopping_criteria=stopping,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency_sec = time.perf_counter() - t0

        generated_ids = outputs[0][prompt_len:]
        num_generated_tokens = len(generated_ids)
        tokens_per_sec = float(num_generated_tokens / max(latency_sec, 1e-4))

        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Extract 1-2 sentences strictly
        raw_sentences = [s.strip() for s in re.split(r"[.!?]", raw_output) if len(s.strip()) > 3]
        if len(raw_sentences) == 0:
            cleaned_briefing = raw_output
            sentence_count = 1 if raw_output else 0
        elif len(raw_sentences) == 1:
            cleaned_briefing = raw_sentences[0] + "."
            sentence_count = 1
        else:
            cleaned_briefing = raw_sentences[0] + ". " + raw_sentences[1] + "."
            sentence_count = 2

        # Compliance checks
        has_prompt_leakage = ("### Instruction:" in raw_output or "### Target" in raw_output)
        has_json_leakage = ("{" in raw_output and "}" in raw_output and ":" in raw_output)
        format_valid = (sentence_count in [1, 2]) and not has_prompt_leakage and not has_json_leakage and len(cleaned_briefing) > 10

        return {
            "patient_id": patient_id,
            "oncology_briefing": cleaned_briefing,
            "sentence_count": sentence_count,
            "format_valid": format_valid,
            "generation_latency_seconds": round(latency_sec, 2),
            "latency_ms": round(latency_sec * 1000, 1),
            "tokens_per_second": round(tokens_per_sec, 2),
            "generated_tokens": num_generated_tokens,
            "input_tokens": prompt_len,
            "threads_used": self.threads_used,
            "model": "Qwen2.5-0.5B-Instruct + LoRA",
            "checkpoint": str(self.adapter_dir),
            "synthetic_disclaimer": "SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE",
            "status": "SUCCESS"
        }
