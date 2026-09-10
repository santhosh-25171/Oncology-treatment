#!/usr/bin/env python3
"""
Stage 4 SLM — Local Offline Inference & Initial Evaluation Engine
Role: Stage 4 SLM Engineer
Project: Personalized Precision Medicine for Oncology Treatment Optimization

Loads the locally trained Qwen2.5-0.5B LoRA adapter, synthesizes clinical briefs
from multimodal diagnostic context, measures inference latency, and computes
initial evaluation metrics (ROUGE-1, ROUGE-2, ROUGE-L, Entity Preservation).

DISCLAIMER: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "stage4_slm" / "models" / "qwen2.5_0.5b"
ADAPTER_DIR = MODEL_DIR / "adapter"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
TEST_CSV_PATH = PROJECT_ROOT / "stage4_slm" / "data" / "splits" / "test.csv"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class Stage4InferenceEngine:
    """Production-grade local offline inference engine for Stage 4 SLM."""

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        adapter_dir: Path = ADAPTER_DIR,
        tokenizer_dir: Path = TOKENIZER_DIR,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.adapter_dir = Path(adapter_dir)
        self.tokenizer_dir = Path(tokenizer_dir)

        print(f"[Inference] Initializing local SLM on {self.device}...")
        t0 = time.time()

        # Load tokenizer locally (fallback to base model if directory not yet populated)
        tok_path = str(self.tokenizer_dir) if (self.tokenizer_dir / "tokenizer_config.json").exists() else base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
        )

        # Attach LoRA adapter if present
        if (self.adapter_dir / "adapter_config.json").exists():
            print(f"[Inference] Attaching fine-tuned LoRA adapter from: {self.adapter_dir}")
            self.model = PeftModel.from_pretrained(self.base_model, str(self.adapter_dir))
        else:
            print("[Inference] Notice: Adapter not found yet, using base model for verification.")
            self.model = self.base_model

        self.model.to(self.device)
        self.model.eval()
        self.load_time = time.time() - t0
        print(f"[Inference] Model loaded in {self.load_time:.2f} seconds.")

        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    @staticmethod
    def format_prompt(
        patient_id: str,
        source_type: str,
        clinical_report: str,
        stage1_context: Any,
        stage2_context: Any,
        stage3_context: Any,
    ) -> str:
        """Constructs standard Stage 4 SLM prompt from multimodal context."""
        s1_str = json.dumps(stage1_context, separators=(",", ":"), sort_keys=True) if isinstance(stage1_context, dict) else str(stage1_context)
        s2_str = json.dumps(stage2_context, separators=(",", ":"), sort_keys=True) if isinstance(stage2_context, dict) else str(stage2_context)
        s3_str = json.dumps(stage3_context, separators=(",", ":"), sort_keys=True) if isinstance(stage3_context, dict) else str(stage3_context)

        return (
            f"### Instruction:\n"
            f"Synthesize a concise, clinically accurate precision oncology summary for patient {patient_id} "
            f"integrating the consultation report, Stage 1 risk assessments, Stage 2 multimodal findings, and Stage 3 triage urgency.\n\n"
            f"### Clinical Report ({source_type}):\n{clinical_report}\n\n"
            f"### Multimodal Context (Stages 1-3):\n"
            f"Stage 1 ML: {s1_str}\n"
            f"Stage 2 DL: {s2_str}\n"
            f"Stage 3 NLP: {s3_str}\n\n"
            f"### Target Oncology Summary:\n"
        )

    def generate_summary(
        self,
        prompt: str,
        max_new_tokens: int = 45,
        temperature: float = 0.1,
        repetition_penalty: float = 1.15,
    ) -> Tuple[str, float, float]:
        """
        Generates a concise 1-2 sentence briefing locally.
        Returns: (generated_text, latency_seconds, tokens_per_second)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.1),
                temperature=temperature if temperature > 0.1 else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.time() - t0

        generated_ids = outputs[0][input_len:]
        num_generated_tokens = len(generated_ids)
        tokens_per_sec = float(num_generated_tokens / max(latency, 1e-4))

        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # Clean up output: take only the first 1-2 sentences
        sentences = [s.strip() for s in raw_output.split(".") if len(s.strip()) > 5]
        cleaned_summary = ". ".join(sentences[:2]) + ("." if sentences else "")

        return cleaned_summary, latency, tokens_per_sec

    def evaluate_test_sample(self, num_samples: int = 10) -> Dict[str, Any]:
        """Runs offline evaluation on unseen test partition samples."""
        print(f"\n[Inference] Evaluating {num_samples} unseen test records from {TEST_CSV_PATH}...")
        df_test = pd.read_csv(TEST_CSV_PATH)
        sample_df = df_test.head(num_samples)

        results = []
        rouge1_list, rouge2_list, rougeL_list = [], [], []
        entity_recall_list = []
        latencies, speeds = [], []

        for idx, row in sample_df.iterrows():
            prompt = row["slm_prompt"]
            gold_ref = str(row["target_summary"]).strip()

            gen_text, latency, tok_speed = self.generate_summary(prompt)
            latencies.append(latency)
            speeds.append(tok_speed)

            # ROUGE
            scores = self.scorer.score(gold_ref, gen_text)
            r1 = scores["rouge1"].fmeasure
            r2 = scores["rouge2"].fmeasure
            rL = scores["rougeL"].fmeasure
            rouge1_list.append(r1)
            rouge2_list.append(r2)
            rougeL_list.append(rL)

            # Entity preservation
            try:
                s3 = json.loads(row["stage3_context"])
                ents = [e.lower() for e in s3.get("extracted_entities", [])]
            except Exception:
                ents = []

            found_ents = [e for e in ents if e in gen_text.lower() or any(part in gen_text.lower() for part in e.split())]
            ent_preservation = len(found_ents) / max(len(ents), 1)
            entity_recall_list.append(ent_preservation)

            results.append({
                "patient_id": row["patient_id"],
                "source_type": row["source_type"],
                "gold_summary": gold_ref,
                "generated_summary": gen_text,
                "rouge1": r1,
                "rouge2": r2,
                "rougeL": rL,
                "latency_sec": latency,
                "tokens_per_sec": tok_speed,
                "entity_preservation": ent_preservation,
            })

        metrics = {
            "num_evaluated": num_samples,
            "mean_rouge1": float(np.mean(rouge1_list)),
            "mean_rouge2": float(np.mean(rouge2_list)),
            "mean_rougeL": float(np.mean(rougeL_list)),
            "mean_entity_preservation": float(np.mean(entity_recall_list)),
            "mean_latency_sec": float(np.mean(latencies)),
            "mean_tokens_per_sec": float(np.mean(speeds)),
            "sample_results": results,
        }

        print(f"      Mean ROUGE-1: {metrics['mean_rouge1']:.4f} | ROUGE-2: {metrics['mean_rouge2']:.4f} | ROUGE-L: {metrics['mean_rougeL']:.4f}")
        print(f"      Entity Preservation: {metrics['mean_entity_preservation']*100:.1f}% | Avg Speed: {metrics['mean_tokens_per_sec']:.1f} tokens/s")
        return metrics


def main():
    print("=" * 65)
    print("STAGE 4 SLM — LOCAL INFERENCE & INITIAL EVALUATION")
    print("=" * 65)
    engine = Stage4InferenceEngine()
    metrics = engine.evaluate_test_sample(num_samples=10)

    # Display sample comparison table
    print("\n" + "=" * 90)
    print(f"{'PATIENT':<12} | {'GOLD REFERENCE SUMMARY':<35} | {'GENERATED SLM SUMMARY':<35}")
    print("-" * 90)
    for res in metrics["sample_results"][:5]:
        p = res["patient_id"]
        g = (res["gold_summary"][:32] + "...") if len(res["gold_summary"]) > 32 else res["gold_summary"]
        s = (res["generated_summary"][:32] + "...") if len(res["generated_summary"]) > 32 else res["generated_summary"]
        print(f"{p:<12} | {g:<35} | {s:<35}")
    print("=" * 90)


if __name__ == "__main__":
    main()
