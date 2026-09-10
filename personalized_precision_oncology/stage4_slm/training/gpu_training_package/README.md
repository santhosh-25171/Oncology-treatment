# Stage 4 SLM — Full-Scale GPU Training Package (7,896 Records)

**Role:** Stage 4 SLM Engineer  
**Project:** Personalized Precision Medicine for Oncology Treatment Optimization  
**Architecture:** `Qwen/Qwen2.5-0.5B-Instruct` + LoRA ($r=8, \alpha=16$)  
**Dataset:** 7,896 Train records / 1,010 Validation records (zero leakage, patient-isolated)

---

## 1. Hardware Status & Justification
- **Host System:** Intel Core i7-11800H @ 2.30GHz (Integrated Intel Iris Xe Graphics, **No discrete CUDA GPU**).
- **Physical CPU Benchmark:** 20.24 seconds / training step $\implies$ **44.39 hours / epoch** (~133.2 hours for 3 epochs).
- **Status:** **FULL TRAINING BLOCKED BY HARDWARE ON LOCAL CPU**.
- **Solution:** This standalone, turnkey GPU package is ready for immediate execution on Google Colab (free T4 GPU), RunPod, AWS EC2 (g4dn/g5), or any CUDA-enabled workstation.

---

## 2. Package Contents
| File | Purpose |
| :--- | :--- |
| `train_slm_gpu.py` | Production PyTorch training script with completion-only loss masking, AMP (FP16/BF16), and LoRA. |
| `training_config_gpu.yaml` | Training hyperparameters (batch size 4, grad accum 4 = effective batch 16, lr 2e-4, 3 epochs). |
| `requirements_gpu.txt` | Python package dependencies for CUDA environment. |
| `Stage4_SLM_Full_Training_Colab.ipynb` | Ready-to-run Google Colab Jupyter notebook. |
| `run_gpu_training.sh` | One-click execution shell script for Linux / Cloud GPU. |
| `run_gpu_training.bat` | One-click execution batch script for Windows with NVIDIA GPU. |

---

## 3. Execution Instructions

### Option A: Google Colab (Recommended, Free T4 GPU)
1. Open Google Colab: https://colab.research.google.com
2. Upload `Stage4_SLM_Full_Training_Colab.ipynb`.
3. Set Runtime Type to **GPU (T4)**.
4. Upload `train.csv` and `val.csv` from `stage4_slm/data/splits/`.
5. Run all cells. Expected training duration: **~4.2 hours**.
6. Download the resulting `final_slm_adapter_7896.zip` and extract to `stage4_slm/models/qwen2.5_0.5b/adapter/`.

### Option B: Linux / Cloud GPU Instance (RunPod, Lambda Labs, AWS)
```bash
cd stage4_slm/training/gpu_training_package
chmod +x run_gpu_training.sh
./run_gpu_training.sh
```

### Option C: Windows Workstation with NVIDIA GPU
```cmd
cd stage4_slm\training\gpu_training_package
run_gpu_training.bat
```

---

## 4. Expected Training Metrics on GPU
- **Trainable Parameters:** 2,162,688 (0.44% of base model)
- **Effective Batch Size:** 16 (per-device 4 $\times$ grad accum 4)
- **Total Steps:** 1,480 optimizer steps across 3 epochs
- **Target Final Train Loss:** $< 0.65$
- **Target Final Validation Loss:** $< 0.70$
- **Checkpoint Output:** `stage4_slm/models/qwen2.5_0.5b/final_full_train/adapter/`
