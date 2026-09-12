#!/usr/bin/env python3
"""
Stage 4 SLM — Full-Scale GPU Training Pipeline (7,896 Records)
Role: Stage 4 SLM Engineer
Project: Personalized Precision Medicine for Oncology Treatment Optimization

Trains Qwen/Qwen2.5-0.5B-Instruct + LoRA on an external NVIDIA GPU (T4, A100, H100, RTX 3090/4090).
Uses completion-only loss masking, bfloat16/float16 mixed precision, and gradient accumulation.

Usage:
  python train_slm_gpu.py --config training_config_gpu.yaml
"""

import os
import sys
import time
import math
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


class OncologyCompletionDataset(Dataset):
    """Completion-only masked dataset for oncology bedside briefing synthesis."""

    def __init__(self, csv_path: str, tokenizer: AutoTokenizer, max_seq_length: int = 512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        self._prepare()

    def _prepare(self):
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        for _, row in self.df.iterrows():
            prompt_str = str(row["slm_prompt"])
            target_str = str(row["target_summary"]).strip()

            p_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            t_ids = self.tokenizer.encode(target_str, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            if len(p_ids) + len(t_ids) > self.max_seq_length:
                p_ids = p_ids[-(self.max_seq_length - len(t_ids)):]

            input_ids = p_ids + t_ids
            # Mask prompt tokens with -100 so loss is computed exclusively on target summary
            labels = [-100] * len(p_ids) + t_ids
            attention_mask = [1] * len(input_ids)

            # Pad to max_seq_length
            pad_len = self.max_seq_length - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def run_gpu_training(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    if device.type != "cuda":
        print("[WARNING] CUDA is NOT available! GPU training will fall back to CPU, which will be slow.")
        print("[WARNING] Recommended: Run on an NVIDIA GPU instance (Colab, RunPod, AWS EC2).")
    else:
        print(f"[GPU] {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM")

    model_name = cfg["model"]["base_model"]
    max_len = cfg["model"].get("max_seq_length", 512)
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Tokenizer] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[Data] Loading datasets...")
    train_dataset = OncologyCompletionDataset(cfg["training"]["train_dataset"], tokenizer, max_seq_length=max_len)
    val_dataset = OncologyCompletionDataset(cfg["training"]["val_dataset"], tokenizer, max_seq_length=max_len)
    print(f"       Train records: {len(train_dataset)} | Val records: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["per_device_train_batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["per_device_eval_batch_size"], shuffle=False)

    dtype = torch.bfloat16 if cfg["training"].get("bf16", False) and torch.cuda.is_bf16_supported() else (torch.float16 if cfg["training"].get("fp16", False) else torch.float32)
    print(f"[Model] Loading {model_name} in {dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    epochs = cfg["training"]["num_train_epochs"]
    grad_accum = cfg["training"]["gradient_accumulation_steps"]
    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\n[Training] Starting full training: {epochs} epochs | {total_steps} optimizer steps | Grad Accum: {grad_accum}")
    t_start = time.time()
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        step_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()

            step_loss += loss.item() * grad_accum
            running_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (cfg["training"]["logging_steps"] * grad_accum) == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_step_loss = step_loss / (cfg["training"]["logging_steps"] * grad_accum)
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_loader)}] Train Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e}")
                step_loss = 0.0

        # Epoch Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_train_loss = running_loss / max(len(train_loader), 1)
        elapsed = time.time() - t_start
        print(f"\n[Epoch {epoch+1} Complete] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Elapsed: {elapsed/3600:.2f} hrs\n")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "elapsed_seconds": round(elapsed, 2)
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  Saving best checkpoint to {output_dir / 'best_checkpoint'}...")
            model.save_pretrained(str(output_dir / "best_checkpoint"))
            tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    # Final Save
    print(f"[Done] Saving final model to {output_dir / 'adapter'}...")
    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": cfg,
            "total_duration_hours": round((time.time() - t_start) / 3600, 3),
            "final_train_loss": round(avg_train_loss, 4),
            "final_val_loss": round(avg_val_loss, 4),
            "history": history,
            "device": str(device),
        }, f, indent=2)

    print(f"[Complete] Full GPU training finished in {(time.time() - t_start) / 3600:.2f} hours. Output saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training_config_gpu.yaml")
    args = parser.parse_args()
    run_gpu_training(args.config)
