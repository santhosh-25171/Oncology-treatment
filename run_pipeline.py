import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"RUNNING: {script_path}")
    print(f"{'='*50}")
    
    try:
        # Run the script and stream the output to console
        process = subprocess.run([sys.executable, script_path], check=True)
        print(f"\n[SUCCESS] {script_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {script_path} failed with exit code {e.returncode}.")
        sys.exit(1)

def main():
    print("Starting Stage 1 ML Pipeline Execution Controller...\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    pipeline_steps = [
        "stage1_ml/data/clean_data.py",
        "stage1_ml/eda/eda.py",
        "stage1_ml/features/feature_engineering.py",
        "stage1_ml/training/train.py",
        "stage1_ml/training/tune.py",
        "stage1_ml/evaluation/model_validation.py",
        "stage1_ml/explainability/explain.py",
        "stage1_ml/prediction/test_prediction.py"
    ]
    
    for step in pipeline_steps:
        script_path = os.path.join(base_dir, step)
        run_script(script_path)
        
    print("\n" + "="*50)
    print("STAGE 1 ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    
if __name__ == "__main__":
    main()
