@echo off
REM Batch script to execute full Stage 4 SLM GPU training on Windows with NVIDIA GPU
echo ==========================================================
echo Stage 4 SLM -- Full-Scale GPU Training Launch (7,896 records)
echo ==========================================================

pip install -r requirements_gpu.txt
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python train_slm_gpu.py --config training_config_gpu.yaml

echo ==========================================================
echo GPU Training Completed!
echo ==========================================================
pause
