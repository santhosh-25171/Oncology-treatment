#!/bin/bash
# Shell script to execute full Stage 4 SLM GPU training on Linux / Cloud GPU instance
set -e

echo "=========================================================="
echo "Stage 4 SLM — Full-Scale GPU Training Launch (7,896 records)"
echo "=========================================================="

# 1. Install GPU requirements
pip install -r requirements_gpu.txt

# 2. Check GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 3. Launch training
python train_slm_gpu.py --config training_config_gpu.yaml

echo "=========================================================="
echo "GPU Training Completed! Checkpoint in stage4_slm/models/qwen2.5_0.5b/final_full_train"
echo "=========================================================="
