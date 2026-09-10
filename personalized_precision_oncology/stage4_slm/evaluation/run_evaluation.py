#!/usr/bin/env python3
"""
Stage 4 SLM — Full Test-Set Evaluation Engine (950 Records)
Role: Stage 4 SLM Engineer + Evaluation Engineer
Project: Personalized Precision Medicine for Oncology Treatment Optimization

Evaluates the newly trained Small Language Model (Qwen2.5-0.5B-Instruct + LoRA)
across ALL 950 unseen test records with batching and two-sentence stopping criteria:
1. ROUGE-1, ROUGE-2, ROUGE-L, and sentence-level BLEU
2. Clinical Oncology Entity Precision, Recall, F1, and Preservation Rate
3. Clinical Faithfulness & Safety Audit (Fully Supported, Partially Supported, Contradictory, Unsupported)
4. Multimodal Stage 1, Stage 2, and Stage 3 Signal Preservation
5. Output Formatting & Length Compliance (1-2 sentences)
6. 7-Stage CPU Latency Benchmarking (mean, median, min, max, P95, P99, tokens/sec)

DISCLAIMER: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.
"""

import os
import sys
import time
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / "stage4_slm" / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
MODEL_DIR = PROJECT_ROOT / "stage4_slm" / "models" / "qwen2.5_0.5b"
ADAPTER_DIR = MODEL_DIR / "adapter"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
TEST_CSV_PATH = PROJECT_ROOT / "stage4_slm" / "data" / "splits" / "test.csv"
DICT_PATH = PROJECT_ROOT / "stage4_slm" / "domain" / "oncology_dictionary.json"

# Optimal CPU thread configuration discovered during latency benchmarking
torch.set_num_threads(6)


class TwoSentenceStoppingCriteria(StoppingCriteria):
    """Halts autoregressive decoding immediately once two complete sentences are formed."""

    def __init__(self, tokenizer: AutoTokenizer, prompt_lens: List[int]):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_lens = prompt_lens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if all items in batch have generated at least 2 periods
        batch_done = True
        for i, p_len in enumerate(self.prompt_lens):
            gen_ids = input_ids[i, p_len:]
            if len(gen_ids) < 10:
                return False
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if text.count(".") < 2:
                batch_done = False
                break
        return batch_done


class FullTestEvaluator:
    """Rigorous evaluation engine for retrained Stage 4 SLM across all 950 test records."""

    def __init__(self, test_csv: Path = TEST_CSV_PATH):
        self.test_csv = test_csv
        self.df_test = pd.read_csv(self.test_csv)
        print(f"[Init] Loaded {len(self.df_test)} test records from {self.test_csv}")

        # Load ontology dictionary
        self.oncology_terms = set()
        if DICT_PATH.exists():
            with open(DICT_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
                for cat, items in d.get("categories", {}).items():
                    for item in items:
                        self.oncology_terms.add(item["term"].lower())
                        if "abbreviation" in item and item["abbreviation"]:
                            self.oncology_terms.add(item["abbreviation"].lower())
                        for var in item.get("common_variants", []):
                            self.oncology_terms.add(var.lower())

        # Load tokenizer with left-padding for batched causal generation
        print(f"[Init] Loading tokenizer from: {TOKENIZER_DIR}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load base model and merge LoRA adapter into memory
        print(f"[Init] Loading base model and merging retrained adapter from: {ADAPTER_DIR}")
        t0 = time.time()
        base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
        self.peft_model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        self.model = self.peft_model.merge_and_unload()
        self.model.eval()
        self.load_latency = time.time() - t0
        print(f"[Init] Model loaded and merged in RAM in {self.load_latency:.2f}s")

        # Metric scorers
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

    def evaluate_batch(self, batch_rows: List[Tuple[int, pd.Series]], max_new_tokens: int = 25) -> List[Dict[str, Any]]:
        """Processes a mini-batch of test records concurrently, returning individual metrics."""
        prompts = [str(r["slm_prompt"]) for _, r in batch_rows]
        
        # Tokenize with left-padding
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        prompt_lens = [enc.input_ids.shape[1]] * len(prompts)

        # Generate with stopping criteria
        stopping = StoppingCriteriaList([TwoSentenceStoppingCriteria(self.tokenizer, prompt_lens)])
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
        batch_duration = time.perf_counter() - t0
        per_sample_latency = batch_duration / max(len(batch_rows), 1)

        batch_results = []
        for idx_in_batch, (global_idx, row) in enumerate(batch_rows):
            prompt_len = enc.input_ids.shape[1]
            generated_ids = outputs[idx_in_batch][prompt_len:]
            raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            patient_id = row["patient_id"]
            source_type = row.get("source_type", "consultation")
            gold_summary = str(row["target_summary"]).strip()
            clinical_report = str(row["clinical_report"])
            s1_raw = row["stage1_context"]
            s2_raw = row["stage2_context"]
            s3_raw = row["stage3_context"]

            # Sentence extraction & formatting
            raw_sentences = [s.strip() for s in re.split(r"[.!?]", raw_text) if len(s.strip()) > 3]
            if len(raw_sentences) == 0:
                cleaned_summary = raw_text
                sentence_count = 1 if raw_text else 0
            elif len(raw_sentences) == 1:
                cleaned_summary = raw_sentences[0] + "."
                sentence_count = 1
            else:
                cleaned_summary = raw_sentences[0] + ". " + raw_sentences[1] + "."
                sentence_count = 2

            # Format compliance checks
            has_prompt_leakage = ("### Instruction:" in raw_text or "### Target" in raw_text)
            has_json_leakage = ("{" in raw_text and "}" in raw_text and ":" in raw_text)
            format_valid = (sentence_count in [1, 2]) and not has_prompt_leakage and not has_json_leakage and len(cleaned_summary) > 10

            # ROUGE scores
            r_scores = self.rouge.score(gold_summary, cleaned_summary)
            r1 = float(r_scores["rouge1"].fmeasure)
            r2 = float(r_scores["rouge2"].fmeasure)
            rL = float(r_scores["rougeL"].fmeasure)

            # BLEU score
            ref_tokens = gold_summary.lower().split()
            cand_tokens = cleaned_summary.lower().split()
            bleu = float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smooth))

            # Entity Evaluation
            try:
                s3_dict = json.loads(s3_raw) if isinstance(s3_raw, str) else s3_raw
                expected_entities = [e.lower().strip() for e in s3_dict.get("extracted_entities", [])]
            except Exception:
                expected_entities = []

            summary_lower = cleaned_summary.lower()
            found_expected = [e for e in expected_entities if e in summary_lower or any(part in summary_lower for part in e.split())]
            entity_recall = float(len(found_expected) / max(len(expected_entities), 1))
            entity_preservation = entity_recall

            gen_terms = [t for t in self.oncology_terms if t in summary_lower and len(t) > 3]
            correct_terms = [t for t in gen_terms if t in clinical_report.lower() or t in str(s1_raw).lower() or t in str(s2_raw).lower() or t in str(s3_raw).lower()]
            entity_precision = float(len(correct_terms) / max(len(gen_terms), 1))
            entity_f1 = float(2 * (entity_precision * entity_recall) / (entity_precision + entity_recall)) if (entity_precision + entity_recall) > 0 else 0.0

            # Clinical Faithfulness
            context_str = f"{clinical_report} {s1_raw} {s2_raw} {s3_raw}".lower()
            unsupported_terms = [t for t in gen_terms if t not in context_str]
            has_contradiction = False
            for stg in ["stage i", "stage ii", "stage iii", "stage iv", "stage 1", "stage 2", "stage 3", "stage 4"]:
                if stg in summary_lower and stg not in context_str:
                    has_contradiction = True

            if has_contradiction:
                faithfulness_status = "CONTRADICTORY"
            elif len(unsupported_terms) > 1:
                faithfulness_status = "UNSUPPORTED_CLAIM"
            elif entity_preservation >= 0.5 and len(unsupported_terms) == 0:
                faithfulness_status = "FULLY_SUPPORTED"
            else:
                faithfulness_status = "PARTIALLY_SUPPORTED"

            # Signal Preservation
            s1_dict = json.loads(s1_raw) if isinstance(s1_raw, str) else s1_raw
            s1_risk = str(s1_dict.get("risk_category", "")).lower()
            s1_preserved = bool(s1_risk in summary_lower or "risk" in summary_lower or "mortality" in summary_lower or "prognos" in summary_lower)

            s2_dict = json.loads(s2_raw) if isinstance(s2_raw, str) else s2_raw
            s2_pred = str(s2_dict.get("prediction", "")).lower()
            s2_preserved = bool(s2_pred in summary_lower or "progress" in summary_lower or "stable" in summary_lower or "response" in summary_lower)

            s3_dict = json.loads(s3_raw) if isinstance(s3_raw, str) else s3_raw
            s3_urgency = str(s3_dict.get("urgency_level", "")).lower()
            s3_preserved = bool(s3_urgency in summary_lower or "evaluat" in summary_lower or "urgent" in summary_lower or "routine" in summary_lower)

            out_tokens = len(generated_ids)
            tps = float(out_tokens / max(per_sample_latency, 1e-4))

            batch_results.append({
                "row_index": global_idx,
                "patient_id": patient_id,
                "source_type": source_type,
                "clinical_report": clinical_report,
                "target_summary": gold_summary,
                "reference_summary": gold_summary,
                "generated_summary": cleaned_summary,
                "stage1_context": s1_raw,
                "stage2_context": s2_raw,
                "stage3_context": s3_raw,
                "input_tokens": prompt_len,
                "output_tokens": out_tokens,
                "generated_tokens": out_tokens,
                "generation_time_seconds": round(per_sample_latency, 4),
                "latency_ms": round(per_sample_latency * 1000, 2),
                "tokens_per_second": round(tps, 2),
                "format_valid": format_valid,
                "sentence_count": sentence_count,
                "rouge1": round(r1, 4),
                "rouge2": round(r2, 4),
                "rougeL": round(rL, 4),
                "bleu": round(bleu, 4),
                "entity_precision": round(entity_precision, 4),
                "entity_recall": round(entity_recall, 4),
                "entity_f1": round(entity_f1, 4),
                "entity_preservation": round(entity_preservation, 4),
                "faithfulness_status": faithfulness_status,
                "stage1_preserved": s1_preserved,
                "stage2_preserved": s2_preserved,
                "stage3_preserved": s3_preserved,
            })

        return batch_results

    def run_full_evaluation(
        self,
        max_samples: Optional[int] = None,
        batch_size: int = 2,
        output_csv: Path = EVAL_DIR / "full_test_predictions.csv",
    ) -> pd.DataFrame:
        """Executes full evaluation across all 950 test records with streaming checkpointing."""
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        total_rows = len(self.df_test)
        num_eval = min(max_samples, total_rows) if max_samples else total_rows
        print(f"\n[Evaluation] Starting full test evaluation on {num_eval} records (Batch size: {batch_size})...")

        jsonl_path = RESULTS_DIR / "full_test_results.jsonl"
        evaluated_records = []
        evaluated_row_indices = set()

        # Resumption support
        if output_csv.exists():
            try:
                existing_df = pd.read_csv(output_csv)
                if "row_index" in existing_df.columns:
                    evaluated_row_indices = set(existing_df["row_index"])
                    evaluated_records = existing_df.to_dict(orient="records")
                    print(f"[Resumption] Found {len(evaluated_records)} existing records in {output_csv}. Resuming from index...")
            except Exception as e:
                print(f"[Resumption] Note: Starting fresh due to: {e}")

        batch_queue = []
        start_time = time.time()

        for idx in range(num_eval):
            if idx in evaluated_row_indices:
                continue

            row = self.df_test.iloc[idx]
            batch_queue.append((idx, row))

            if len(batch_queue) == batch_size or idx == num_eval - 1:
                batch_res = self.evaluate_batch(batch_queue)
                evaluated_records.extend(batch_res)
                for r in batch_res:
                    evaluated_row_indices.add(r["row_index"])
                batch_queue = []

                # Periodic checkpointing every 10 records or at end
                if len(evaluated_records) % 10 == 0 or len(evaluated_row_indices) == num_eval:
                    cur_df = pd.DataFrame(evaluated_records)
                    cur_df.to_csv(output_csv, index=False)
                    
                    # Update jsonl
                    with open(jsonl_path, "w", encoding="utf-8") as f:
                        for rec in evaluated_records:
                            f.write(json.dumps(rec) + "\n")

                    elapsed = time.time() - start_time
                    avg_spd = elapsed / max(len(evaluated_records), 1)
                    print(f"      Evaluated {len(evaluated_records):03d}/{num_eval} records | Avg: {avg_spd:.2f}s/rec | Progress: {len(evaluated_records)/num_eval*100:.1f}%", flush=True)

        final_df = pd.DataFrame(evaluated_records)
        final_df.to_csv(output_csv, index=False)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in evaluated_records:
                f.write(json.dumps(rec) + "\n")

        print(f"[Done] Evaluated {len(final_df)} records saved to: {output_csv} and {jsonl_path}")
        return final_df

    def compute_and_save_summary_metrics(self, pred_df: pd.DataFrame):
        """Computes statistical summaries, entity metrics, faithfulness, latency, and saves all required artifacts."""
        print("\n[Metrics] Computing full statistical distribution...")

        # 1. Text Generation & Numerical Metrics
        metric_cols = [
            "rouge1", "rouge2", "rougeL", "bleu",
            "entity_precision", "entity_recall", "entity_f1", "entity_preservation",
            "generation_time_seconds", "latency_ms", "tokens_per_second"
        ]
        stats_data = []

        for col in metric_cols:
            if col in pred_df.columns:
                series = pred_df[col].dropna()
                stats_data.append({
                    "metric": col,
                    "mean": round(float(series.mean()), 4),
                    "median": round(float(series.median()), 4),
                    "std": round(float(series.std()), 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                    "p25": round(float(series.quantile(0.25)), 4),
                    "p75": round(float(series.quantile(0.75)), 4),
                    "p95": round(float(series.quantile(0.95)), 4),
                    "p99": round(float(series.quantile(0.99)), 4),
                })

        metrics_summary_df = pd.DataFrame(stats_data)
        metrics_summary_df.to_csv(EVAL_DIR / "evaluation_metrics.csv", index=False)

        # Full Test Metrics JSON
        full_metrics_payload = {
            "evaluation_scope": {
                "dataset": "stage4_slm/data/splits/test.csv",
                "total_records_evaluated": len(pred_df),
                "total_test_partition_records": len(self.df_test),
                "coverage_percentage": round(len(pred_df) / len(self.df_test) * 100, 2),
                "model": "Qwen/Qwen2.5-0.5B-Instruct + LoRA",
                "checkpoint": "stage4_slm/models/qwen2.5_0.5b/adapter",
                "device": "cpu",
                "cpu_threads": 6,
                "cold_start_load_seconds": self.load_latency,
            },
            "metrics": {row["metric"]: {k: v for k, v in row.items() if k != "metric"} for row in stats_data},
        }
        with open(EVAL_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(full_metrics_payload, f, indent=2)
        with open(RESULTS_DIR / "full_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(full_metrics_payload, f, indent=2)

        # 2. Faithfulness Distribution
        faith_counts = pred_df["faithfulness_status"].value_counts(normalize=True).to_dict()
        faith_summary = []
        for status in ["FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED_CLAIM", "CONTRADICTORY"]:
            pct = float(faith_counts.get(status, 0.0) * 100)
            cnt = int((pred_df["faithfulness_status"] == status).sum())
            faith_summary.append({"faithfulness_status": status, "count": cnt, "percentage": round(pct, 2)})
        pd.DataFrame(faith_summary).to_csv(EVAL_DIR / "faithfulness_audit.csv", index=False)

        faith_payload = {
            "total_evaluated": len(pred_df),
            "fully_supported_percentage": round(float(faith_counts.get("FULLY_SUPPORTED", 0.0) * 100), 2),
            "partially_supported_percentage": round(float(faith_counts.get("PARTIALLY_SUPPORTED", 0.0) * 100), 2),
            "contradictory_percentage": round(float(faith_counts.get("CONTRADICTORY", 0.0) * 100), 2),
            "unsupported_claim_percentage": round(float(faith_counts.get("UNSUPPORTED_CLAIM", 0.0) * 100), 2),
            "summary_table": faith_summary,
        }
        with open(RESULTS_DIR / "faithfulness_results.json", "w", encoding="utf-8") as f:
            json.dump(faith_payload, f, indent=2)

        # 3. Stage Signal Preservation
        s1_rate = float(pred_df["stage1_preserved"].mean() * 100)
        s2_rate = float(pred_df["stage2_preserved"].mean() * 100)
        s3_rate = float(pred_df["stage3_preserved"].mean() * 100)
        signal_summary = [
            {"signal": "Stage 1 ML (Mortality/Recurrence/Risk Category)", "expected": "Reflect risk or mortality", "preservation_percentage": round(s1_rate, 2), "status": "PASS" if s1_rate >= 70 else "PARTIAL"},
            {"signal": "Stage 2 DL (Progression Trajectory/Efficacy)", "expected": "Reflect progression or stable disease", "preservation_percentage": round(s2_rate, 2), "status": "PASS" if s2_rate >= 70 else "PARTIAL"},
            {"signal": "Stage 3 NLP (Urgency Priority & Entities)", "expected": "Reflect triage urgency", "preservation_percentage": round(s3_rate, 2), "status": "PASS" if s3_rate >= 70 else "PARTIAL"},
        ]
        pd.DataFrame(signal_summary).to_csv(EVAL_DIR / "stage_signal_preservation.csv", index=False)

        # 4. Entity Metrics
        entity_payload = {
            "mean_entity_precision": round(float(pred_df["entity_precision"].mean()), 4),
            "mean_entity_recall": round(float(pred_df["entity_recall"].mean()), 4),
            "mean_entity_f1": round(float(pred_df["entity_f1"].mean()), 4),
            "mean_entity_preservation_rate": round(float(pred_df["entity_preservation"].mean()), 4),
        }
        entity_summary = [{"entity_metric": k, "value": v} for k, v in entity_payload.items()]
        pd.DataFrame(entity_summary).to_csv(EVAL_DIR / "entity_evaluation.csv", index=False)
        with open(RESULTS_DIR / "entity_metrics.json", "w", encoding="utf-8") as f:
            json.dump(entity_payload, f, indent=2)

        # 5. Latency Metrics
        lat_series = pred_df["generation_time_seconds"]
        spd_series = pred_df["tokens_per_second"]
        lat_payload = {
            "cold_start_seconds": round(self.load_latency, 2),
            "warm_mean_seconds": round(float(lat_series.mean()), 2),
            "warm_median_seconds": round(float(lat_series.median()), 2),
            "p95_seconds": round(float(lat_series.quantile(0.95)), 2),
            "p99_seconds": round(float(lat_series.quantile(0.99)), 2),
            "min_seconds": round(float(lat_series.min()), 2),
            "max_seconds": round(float(lat_series.max()), 2),
            "mean_tokens_per_second": round(float(spd_series.mean()), 2),
            "target_5s_compliance": "PASS" if lat_series.mean() <= 5.0 else "FAIL (5-second target NOT YET ACHIEVED on current CPU hardware)",
            "Mean Generation Latency (s)": round(float(lat_series.mean()), 2),
            "Cold Start Latency (s)": round(self.load_latency, 2),
        }
        latency_summary = [{"benchmark_metric": k, "value": v} for k, v in lat_payload.items()]
        pd.DataFrame(latency_summary).to_csv(EVAL_DIR / "latency_benchmark.csv", index=False)
        with open(RESULTS_DIR / "latency_metrics.json", "w", encoding="utf-8") as f:
            json.dump(lat_payload, f, indent=2)

        # Format failure rate
        format_fail_rate = round(float((~pred_df["format_valid"]).mean() * 100), 2)
        print(f"[Metrics] Format compliance: {100.0 - format_fail_rate:.2f}% | Format failure rate: {format_fail_rate:.2f}%")
        print("[Metrics] All evaluation artifacts successfully generated.")


def main():
    parser = argparse.ArgumentParser(description="Stage 4 SLM Full Test Evaluation")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all 950)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for parallel decoding (default: 2)")
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 4 SLM — FULL TEST-SET EVALUATION PIPELINE")
    print("=" * 70)
    evaluator = FullTestEvaluator()
    pred_df = evaluator.run_full_evaluation(max_samples=args.num_samples, batch_size=args.batch_size)
    evaluator.compute_and_save_summary_metrics(pred_df)
    print("=" * 70)


if __name__ == "__main__":
    main()
