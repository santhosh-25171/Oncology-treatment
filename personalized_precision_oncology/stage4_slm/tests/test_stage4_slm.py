#!/usr/bin/env python3
"""
Unit tests for Stage 4 SLM (Small Language Model) Fine-Tuning & Inference Pipeline.
Role: Stage 4 SLM Engineer
Project: Personalized Precision Medicine for Oncology Treatment Optimization

DISCLAIMER: SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.
"""

import sys
import json
from pathlib import Path
import pytest
import torch
import yaml

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

CONFIG_PATH = PROJECT_ROOT / "stage4_slm" / "training" / "training_config.yaml"
MODEL_DIR = PROJECT_ROOT / "stage4_slm" / "models" / "qwen2.5_0.5b"
ADAPTER_DIR = MODEL_DIR / "adapter"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"


def test_training_config_structure():
    """Verify training_config.yaml exists and contains all required hyperparameter sections."""
    assert CONFIG_PATH.exists(), f"Configuration not found at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "model" in cfg, "Missing 'model' section in config"
    assert "peft_lora" in cfg, "Missing 'peft_lora' section in config"
    assert "data" in cfg, "Missing 'data' section in config"
    assert "training" in cfg, "Missing 'training' section in config"
    assert "output" in cfg, "Missing 'output' section in config"

    assert cfg["model"]["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg["data"]["max_seq_length"] == 512
    assert cfg["data"]["completion_only_loss_masking"] is True
    assert cfg["peft_lora"]["r"] == 16
    assert cfg["peft_lora"]["lora_alpha"] == 32


def test_peft_lora_target_modules():
    """Verify all 7 linear projection layers are targeted for LoRA parameter-efficiency."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert cfg["peft_lora"]["target_modules"] == expected_modules


def test_prompt_formatting():
    """Verify Stage4InferenceEngine.format_prompt properly injects all clinical and multimodal signals."""
    from stage4_slm.inference.run_inference import Stage4InferenceEngine

    p_id = "SYNTH_P001"
    src = "Pathology"
    clin_rep = "High-grade invasive ductal carcinoma. ER-positive, PR-negative, HER2-equivocal."
    s1 = {"high_risk_flag": 1, "predicted_mortality_risk": 0.35}
    s2 = {"predicted_efficacy_score": 0.72}
    s3 = {"extracted_entities": ["invasive ductal carcinoma", "ER-positive"], "triage_urgency": "Urgent"}

    prompt = Stage4InferenceEngine.format_prompt(p_id, src, clin_rep, s1, s2, s3)

    assert p_id in prompt
    assert src in prompt
    assert clin_rep in prompt
    assert "Stage 1 ML:" in prompt
    assert "Stage 2 DL:" in prompt
    assert "Stage 3 NLP:" in prompt
    assert "### Target Oncology Summary:" in prompt


def test_completion_loss_masking_contract():
    """Verify that prompt tokens receive label -100 and target summary tokens have active labels."""
    from stage4_slm.training.train_slm import OncologySynthesisDataset

    tok_path = str(TOKENIZER_DIR) if (TOKENIZER_DIR / "tokenizer_config.json").exists() else "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_csv = PROJECT_ROOT / "stage4_slm" / "data" / "splits" / "train.csv"
    dataset = OncologySynthesisDataset(train_csv, tok, max_seq_length=512, max_samples=3)

    assert len(dataset) == 3
    sample = dataset[0]

    # Prompt portion must be all -100
    prompt_labels = sample["labels"][:sample["prompt_len"]]
    assert (prompt_labels == -100).all(), "Prompt tokens were not masked with -100"

    # Completion target portion must NOT be -100
    target_labels = sample["labels"][sample["prompt_len"]:]
    assert (target_labels != -100).all(), "Target summary tokens were incorrectly masked with -100"

    # Length matching
    assert len(sample["input_ids"]) == len(sample["labels"]) == len(sample["attention_mask"])


def test_collate_fn_padding():
    """Verify that batch collation correctly pads input_ids, labels (-100), and attention_mask (0)."""
    from stage4_slm.training.train_slm import collate_fn

    item1 = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
    }
    item2 = {
        "input_ids": torch.tensor([4, 5], dtype=torch.long),
        "labels": torch.tensor([-100, 5], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
    }

    batch = collate_fn([item1, item2], pad_token_id=0)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)

    # Item 2 last token must be padded
    assert batch["input_ids"][1, 2].item() == 0
    assert batch["labels"][1, 2].item() == -100
    assert batch["attention_mask"][1, 2].item() == 0
