#!/usr/bin/env python3
"""
Stage 4 SLM — Production-Quality Parameter-Efficient Fine-Tuning (PEFT/LoRA)
Role: Stage 4 SLM Engineer
Project: Personalized Precision Medicine for Oncology Treatment Optimization

Trains a local Small Language Model (Qwen2.5-0.5B-Instruct) using completion-only loss masking,
strict patient-isolated train/validation splits, and local checkpointing.

DISCLAIMER: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.
"""

import os
import sys
import time
import math
import json
import random
import argparse
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure Windows Certificate Store is injected for HuggingFace downloads
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "stage4_slm" / "training" / "training_config.yaml"


torch.set_num_threads(8)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OncologySynthesisDataset(Dataset):
    """
    Dataset implementing completion-only loss masking.
    Prompt tokens are masked with -100 so that cross-entropy loss is computed
    strictly on the target oncology summary tokens.
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples > 0:
            self.df = self.df.head(max_samples)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.samples: List[Dict[str, torch.Tensor]] = []
        self._prepare_data()

    def _prepare_data(self):
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        for _, row in self.df.iterrows():
            prompt_text = str(row["slm_prompt"])
            target_text = str(row["target_summary"]).strip()

            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            # Truncate if total sequence exceeds max_seq_length
            if len(prompt_ids) + len(target_ids) > self.max_seq_length:
                prompt_ids = prompt_ids[-(self.max_seq_length - len(target_ids)):]

            input_ids = prompt_ids + target_ids
            # Mask prompt tokens with -100 for completion-only loss
            labels = [-100] * len(prompt_ids) + target_ids
            attention_mask = [1] * len(input_ids)

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "prompt_len": len(prompt_ids),
                "target_len": len(target_ids),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Dynamically pads input_ids, labels, and attention_mask to batch max length."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []

    for item in batch:
        cur_len = item["input_ids"].shape[0]
        pad_len = max_len - cur_len

        # Pad input_ids with pad_token_id
        padded_inputs = torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        # Pad labels with -100
        padded_labels = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        # Pad attention_mask with 0
        padded_mask = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])

        batch_input_ids.append(padded_inputs)
        batch_labels.append(padded_labels)
        batch_attention_mask.append(padded_mask)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attention_mask),
    }


class Stage4SLMTrainer:
    """Orchestrates model fine-tuning with PEFT LoRA, completion-only loss masking, and checkpointing."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        set_seed(self.cfg["training"]["seed"])

        self.model_id = self.cfg["model"]["model_id"]
        self.device = torch.device(self.cfg["model"]["device"])
        self.max_seq_length = self.cfg["data"]["max_seq_length"]

        # Directories
        self.adapter_dir = PROJECT_ROOT / self.cfg["output"]["adapter_dir"]
        self.tokenizer_dir = PROJECT_ROOT / self.cfg["output"]["tokenizer_dir"]
        self.logs_dir = PROJECT_ROOT / self.cfg["output"]["logs_dir"]
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"[1/6] Loading tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        print(f"[2/6] Ingesting verified dataset splits...")
        train_csv = PROJECT_ROOT / self.cfg["data"]["train_path"]
        val_csv = PROJECT_ROOT / self.cfg["data"]["validation_path"]

        max_train_samples = self.cfg["training"].get("max_train_samples", None)
        max_val_samples = self.cfg["training"].get("max_val_samples", None)

        self.train_dataset = OncologySynthesisDataset(
            train_csv, self.tokenizer, self.max_seq_length, max_samples=max_train_samples
        )
        self.val_dataset = OncologySynthesisDataset(
            val_csv, self.tokenizer, self.max_seq_length, max_samples=max_val_samples
        )
        print(f"      Train samples: {len(self.train_dataset):,} | Validation samples: {len(self.val_dataset):,}")

        # Verification of completion-only loss masking
        sample = self.train_dataset[0]
        prompt_mask = sample["labels"][:sample["prompt_len"]]
        target_labels = sample["labels"][sample["prompt_len"]:]
        assert (prompt_mask == -100).all(), "Prompt tokens not masked with -100!"
        assert (target_labels != -100).all(), "Target tokens incorrectly masked!"
        print(f"      Completion-only loss masking verified: {sample['prompt_len']} prompt tokens masked (-100), {sample['target_len']} target tokens active.")

        print(f"[3/6] Loading base model: {self.model_id} on {self.device}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        )

        peft_cfg = self.cfg["peft_lora"]
        print(f"[4/6] Initializing PEFT LoRA (r={peft_cfg['r']}, alpha={peft_cfg['lora_alpha']})...")
        self.lora_config = LoraConfig(
            r=peft_cfg["r"],
            lora_alpha=peft_cfg["lora_alpha"],
            target_modules=peft_cfg["target_modules"],
            lora_dropout=peft_cfg["lora_dropout"],
            bias=peft_cfg["bias"],
            task_type=peft_cfg["task_type"],
        )
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.to(self.device)

        # Print parameter statistics
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        self.param_stats = {
            "trainable_parameters": trainable_params,
            "all_parameters": all_params,
            "trainable_percentage": float(trainable_params / all_params * 100),
        }
        print(f"      Total params: {all_params:,} | Trainable params: {trainable_params:,} ({self.param_stats['trainable_percentage']:.4f}%)")
        print(f"      Target modules: {peft_cfg['target_modules']}")

    def train(self) -> Dict[str, Any]:
        """Executes the fine-tuning loop with gradient accumulation, evaluation, and checkpointing."""
        print("[5/6] Starting SLM training loop...")
        train_cfg = self.cfg["training"]
        batch_size = train_cfg["per_device_train_batch_size"]
        grad_accum = train_cfg["gradient_accumulation_steps"]
        max_steps = train_cfg["max_steps"]
        eval_every = train_cfg["eval_every_steps"]
        save_every = train_cfg["save_every_steps"]

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.pad_token_id),
        )

        # Validation loader (evaluated periodically for loss tracking)
        val_subset = torch.utils.data.Subset(self.val_dataset, list(range(min(20, len(self.val_dataset)))))
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.pad_token_id),
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_cfg["warmup_steps"],
            num_training_steps=max_steps,
        )

        self.model.train()
        train_history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        step = 0
        optimizer.zero_grad()

        tracemalloc.start()
        start_time = time.time()
        print(f"      Configured: max_steps={max_steps}, batch_size={batch_size}, grad_accum={grad_accum}, effective_batch={batch_size * grad_accum}")
        print(f"      Available Training Partition: {len(self.train_dataset):,} records | Available Validation Partition: {len(self.val_dataset):,} records")

        while step < max_steps:
            for batch_idx, batch in enumerate(train_loader):
                if step >= max_steps:
                    break

                step_start = time.time()
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                loss = outputs.loss / grad_accum
                loss.backward()

                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    step += 1
                    raw_loss = float(loss.item() * grad_accum)
                    step_time = time.time() - step_start
                    lr = scheduler.get_last_lr()[0]

                    # Periodic validation evaluation
                    val_loss = None
                    if step % eval_every == 0 or step == max_steps:
                        val_loss = self.evaluate(val_loader)
                        print(f"      Step {step:02d}/{max_steps} | Train Loss: {raw_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e} | Step Time: {step_time:.2f}s", flush=True)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_adapter(is_best=True)
                    else:
                        print(f"      Step {step:02d}/{max_steps} | Train Loss: {raw_loss:.4f} | LR: {lr:.2e} | Step Time: {step_time:.2f}s", flush=True)

                    train_history.append({
                        "step": step,
                        "train_loss": raw_loss,
                        "val_loss": val_loss,
                        "lr": lr,
                        "step_time": step_time,
                    })

        total_duration = time.time() - start_time
        peak_mem_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        print(f"      Training completed in {total_duration:.2f}s! Best Val Loss: {best_val_loss:.4f} | Peak Memory: {peak_mem_mb:.1f} MB")

        # Final save
        self.save_adapter(is_best=False)
        self.save_tokenizer()

        # Save training logs
        log_payload = {
            "model_id": self.model_id,
            "training_duration_seconds": total_duration,
            "total_steps": step,
            "records_available": len(self.train_dataset),
            "validation_records_available": len(self.val_dataset),
            "effective_batch_size": batch_size * grad_accum,
            "peak_memory_mb": round(peak_mem_mb, 2),
            "hardware": "CPU (8 threads, Intel Core i5-11320H)",
            "best_val_loss": best_val_loss,
            "final_train_loss": train_history[-1]["train_loss"] if train_history else None,
            "parameter_statistics": self.param_stats,
            "history": train_history,
        }
        with open(self.logs_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2)

        return log_payload

    def evaluate(self, val_loader: DataLoader) -> float:
        """Computes average loss on validation subset."""
        self.model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                total_loss += float(out.loss.item())
                batches += 1
        self.model.train()
        return float(total_loss / max(batches, 1))

    def save_adapter(self, is_best: bool = False):
        """Saves PEFT LoRA adapter weights and config."""
        save_path = self.adapter_dir
        self.model.save_pretrained(str(save_path))
        if is_best:
            print(f"      [Checkpoint] Saved best adapter to: {save_path}")

    def save_tokenizer(self):
        """Saves tokenizer files locally for offline inference."""
        self.tokenizer.save_pretrained(str(self.tokenizer_dir))
        print(f"      [Artifact] Saved tokenizer to: {self.tokenizer_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 4 SLM Production Fine-Tuning")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max optimization steps")
    parser.add_argument("--grad_accum", type=int, default=None, help="Override gradient accumulation steps")
    args = parser.parse_args()

    print("=" * 65)
    print("STAGE 4 SLM — PRODUCTION FINE-TUNING PIPELINE")
    print("=" * 65)
    trainer = Stage4SLMTrainer()
    if args.max_steps is not None:
        trainer.cfg["training"]["max_steps"] = args.max_steps
    if args.grad_accum is not None:
        trainer.cfg["training"]["gradient_accumulation_steps"] = args.grad_accum
    trainer.train()
    print("[6/6] Fine-tuning and checkpoint saving completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
