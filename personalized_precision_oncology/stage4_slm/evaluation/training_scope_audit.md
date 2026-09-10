# STAGE 4 — SLM EVALUATION ENGINEER
## TRAINING SCOPE & SCALE AUDIT REPORT

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Role**: Stage 4 SLM Evaluation Engineer (Role 4)  
**Date**: September 2026  
**Status**: AUDITED — CRITICAL SCOPE FINDINGS DOCUMENTED  
**Classification**: `SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE`

---

> [!CAUTION]
> ### CLINICAL DISCLAIMER
> **SYNTHETIC RESEARCH DATA — NOT FOR CLINICAL USE.**  
> All records, clinical notes, laboratory values, genomic markers, imaging features, and generated oncology summaries are synthetically generated for machine learning research, software engineering validation, and multi-stage pipeline development. This system has not been evaluated in human clinical trials and must not be used for medical diagnostic or therapeutic decisions.

---

## 1. Executive Summary: The Training Scale Reality

A central duty of the **Stage 4 Evaluation Engineer** is to resolve ambiguity regarding what was actually trained versus what was available in the repository.

In the previous SLM Engineer handover, documentation frequently referenced the full dataset of 7,896 training records and 974/1,010 validation records. However, a rigorous audit of the execution code ([`stage4_slm/training/train_slm.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/training/train_slm.py)), the configuration ([`stage4_slm/training/training_config.yaml`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/training/training_config.yaml)), and the output logs ([`stage4_slm/training/training_logs/training_history.json`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/training/training_logs/training_history.json)) reveals the following definitive engineering truth:

> [!IMPORTANT]
> ### CRITICAL AUDIT FINDING: SUBSET TRAINING ONLY
> The frozen SLM model was **NOT** trained on the full training dataset of 7,896 records.  
> Due to CPU compute constraints and latency bottlenecks on commodity hardware without GPU acceleration, the training pipeline was calibrated to train for **exactly 10 optimization steps** on a sliced training partition of **exactly 100 training examples**, with validation loss evaluated on **20 validation examples**.

---

## 2. Comparison: Full Dataset Available vs. Actually Used

```
+---------------------------------------------------------------------------------------------------------+
| TRAINING SCALE COMPARISON TABLE                                                                         |
+-------------------------------------+-----------------------+---------------------+---------------------+
| Dimension                           | Full Dataset Available| Actually Used in Run| Coverage Percentage |
+-------------------------------------+-----------------------+---------------------+---------------------+
| Training Records (`train.csv`)      | 7,896 records         | 100 records         | 1.266%              |
| Unique Training Patients            | 2,560 patients        | ~35 patients        | 1.367%              |
| Validation Records (`val.csv`)      | 1,010 records         | 20 records          | 1.980%              |
| Total Optimization Steps Completed  | ~7,896 steps (1 epoch)| 10 steps            | 0.127% of 1 epoch   |
| Optimization Epochs Completed       | 1.0 - 3.0 epochs      | 0.100 epoch (subset)| N/A (partial run)   |
| Checkpoint Evaluated                | Final convergence     | Step 10 checkpoint  | Intermediate Step   |
+-------------------------------------+-----------------------+---------------------+---------------------+
```

---

## 3. Detailed Parameter & Hyperparameter Verification

The following values were extracted directly from `training_config.yaml`, `train_slm.py`, and `training_history.json`:

| Training Hyperparameter | Configured & Executed Value | Verification Source | Notes |
| :--- | :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-0.5B-Instruct` | `training_config.yaml:8` | 494M base parameters |
| **Precision / Dtype** | `torch.float32` (FP32) | `train_slm.py:196` | CPU execution standard |
| **Training Examples Loaded** | **100** | `training_config.yaml:51` | `df.head(100)` in dataset loader |
| **Validation Examples Loaded** | **20** | `training_config.yaml:52` | `df.head(20)` in dataset loader |
| **Optimization Steps (`max_steps`)**| **10** | `training_config.yaml:42` | Terminated at step 10 |
| **Per-Device Batch Size** | **1** | `training_config.yaml:45` | Sequential single-sample training |
| **Gradient Accumulation Steps** | **1** | `training_config.yaml:46` | Backprop on every batch |
| **Effective Batch Size** | **1** | `training_config.yaml:47` | Batch size $\times$ Grad accum = 1 |
| **Learning Rate** | **$0.0002$ ($2.0 \times 10^{-4}$)** | `training_config.yaml:40` | Initial AdamW peak rate |
| **Learning Rate Schedule** | Cosine with Warmup | `training_config.yaml:50` | `get_cosine_schedule_with_warmup` |
| **Warmup Steps** | **2** | `training_config.yaml:49` | Step 1: $1.0 \times 10^{-4}$, Step 2: $2.0 \times 10^{-4}$ |
| **Weight Decay** | **0.01** | `training_config.yaml:41` | L2 regularization |
| **Gradient Clipping Norm** | **1.0** | `training_config.yaml:48` | `clip_grad_norm_` |
| **Evaluation Frequency** | Every 5 steps | `training_config.yaml:43` | Evaluated at step 5 and step 10 |
| **Checkpoint Frequency** | Every 5 steps | `training_config.yaml:44` | Saved best adapter on lower val loss|
| **Checkpoint Evaluated** | Step 10 Adapter | `adapter_model.safetensors` | Saved at: `stage4_slm/models/qwen2.5_0.5b/adapter` |

---

## 4. Empirical Step-by-Step Training Progression

From `training_history.json`:

```json
{
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "training_duration_seconds": 512.9989104270935,
  "total_steps": 10,
  "best_val_loss": 1.6474339604377746,
  "final_train_loss": 1.3169339895248413,
  "parameter_statistics": {
    "trainable_parameters": 8798208,
    "all_parameters": 502830976,
    "trainable_percentage": 1.7497346861940342
  }
}
```

```
Step Loss Trajectory:
Step 01: Train Loss = 2.4920 | LR = 1.00e-04 (Warmup 1)
Step 02: Train Loss = 2.3946 | LR = 2.00e-04 (Warmup 2 / Peak)
Step 03: Train Loss = 2.3693 | LR = 1.92e-04
Step 04: Train Loss = 2.0674 | LR = 1.71e-04
Step 05: Train Loss = 1.6720 | Val Loss = 1.9968 | Checkpoint saved
Step 06: Train Loss = 1.6113 | LR = 1.00e-04
Step 07: Train Loss = 2.1590 | LR = 6.17e-05
Step 08: Train Loss = 1.5732 | LR = 2.93e-05
Step 09: Train Loss = 2.5334 | LR = 7.61e-06
Step 10: Train Loss = 1.3169 | Val Loss = 1.6474 | Checkpoint saved (BEST)
```

---

## 5. Architectural & Methodological Evaluation

### Why Did Role 3 Train on 100 Samples for 10 Steps?
1. **Hardware Reality**: The host system has 12 CPU cores and NO CUDA GPU. A single training step with completion-only loss masking over a 336-token sequence takes ~30 to 45 seconds on CPU.
2. **Time to Full Epoch on CPU**:
   $$\text{Time per Epoch} = 7,896 \text{ samples} \times 35 \text{ seconds/sample} = 276,360 \text{ seconds} \approx \mathbf{76.8\text{ hours (3.2 days)}}$$
3. **Engineering Trade-off**: Role 3 prioritized establishing an end-to-end working pipeline (validating tokenizer, completion-only loss masking contract, LoRA adapter injection, optimizer stability, and checkpoint saving) within an ~8.5 minute development window.

### Limitations of the 10-Step / 100-Sample Model:
- **Low Exposure to Long-Tail Oncology Entities**: With only 100 training samples, the model was exposed to ~35 unique synthetic patients and perhaps 40 out of the 100+ entities in `oncology_dictionary.json`.
- **Under-fitted on Edge histologies**: Rare histological subtypes (e.g., adenoid cystic carcinoma, Merkel cell carcinoma, cholangiocarcinoma) likely did not appear in the 100-sample training batch.
- **Reliance on Base Model Knowledge**: For entities unseen in the 100 training samples, the model must rely entirely on its frozen base pre-training representations rather than domain-adapted weights.

---

## 6. Recommendations for Future Production Training (NOT Executed Now)

In strict adherence to Role 4 guidelines: **WE DO NOT SILENTLY RETRAIN THE MODEL**. The model evaluated throughout this evaluation remains the exact frozen Step 10 checkpoint produced by Role 3.

For the future **Stage 5 / Production Integration Phase**, we recommend:
1. **GPU Allocation**: Provision an NVIDIA T4, A10G, or RTX 4090 GPU (8–16 GB VRAM).
2. **Mixed Precision (BF16 / FP16)**: Enable BF16 mixed-precision training via HuggingFace `Accelerate`.
3. **Full Split Training**:
   - Training Records: 7,896
   - Batch Size: 8 with Gradient Accumulation 2 (Effective batch size = 16)
   - Steps per Epoch: ~493 steps
   - Epochs: 3 (total ~1,480 steps)
   - Estimated Training Time on GPU: **~18–25 minutes** (vs 76+ hours on CPU).
4. **Learning Rate Optimization**: Cosine decay with 50 warmup steps, initial LR $2.0 \times 10^{-4}$.
