"""
Stage 5 Master Verification Script
Executes all pipelines and runs test suites across the Stage 5 modules.
Run this directly in VS Code: python run_stage5_verification.py
"""

import os
import sys
import subprocess

def print_banner(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def run_cmd(description, cmd):
    print(f"\n[RUNNING] {description}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print(f"[SUCCESS] {description} passed.")
    else:
        print(f"[FAILED] {description} exited with code {result.returncode}.")
    return result.returncode

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print_banner("PERSONALIZED PRECISION ONCOLOGY — STAGE 5 VERIFICATION")
    print(f"Python interpreter: {sys.executable}")
    print(f"Working directory:  {os.getcwd()}")

    # 1. Run Data Engineering Pipeline
    print_banner("1. Running Stage 5 Data Engineering Pipeline")
    rc1 = run_cmd(
        "Data Engineering Pipeline",
        f'"{sys.executable}" -m personalized_precision_oncology.stage5_genai.data_engineering.src.run_pipeline'
    )

    # 2. Run Data Engineering Tests
    print_banner("2. Running Stage 5 Data Engineering Tests (30 tests)")
    rc2 = run_cmd(
        "Data Engineering Tests",
        f'"{sys.executable}" -m pytest personalized_precision_oncology/stage5_genai/data_engineering/tests --import-mode=importlib -v'
    )

    # 3. Run EDA & Prompt Engineering Tests
    print_banner("3. Running Stage 5 EDA & Prompt Engineering Tests (16 tests)")
    rc3 = run_cmd(
        "EDA & Prompt Engineering Tests",
        f'"{sys.executable}" -m pytest personalized_precision_oncology/stage5_genai/eda_prompteng/tests --import-mode=importlib -v'
    )

    # 4. Run GenAI Scenario Generator Tests
    print_banner("4. Running Stage 5 GenAI Scenario Generator Tests (23 tests)")
    rc4 = run_cmd(
        "GenAI Scenario Generator Tests",
        f'"{sys.executable}" -m pytest personalized_precision_oncology/stage5_genai/genai/tests --import-mode=importlib -v'
    )

    # 5. Run Evaluation Engineer Tests
    print_banner("5. Running Stage 5 Evaluation Tests (31 tests)")
    rc5 = run_cmd(
        "Evaluation Tests",
        f'"{sys.executable}" -m pytest personalized_precision_oncology/stage5_genai/evaluation/tests --import-mode=importlib -v'
    )

    # 6. Run Integration Engineer Tests
    print_banner("6. Running Stage 5 Integration Tests (29 tests)")
    rc6 = run_cmd(
        "Integration Tests",
        f'"{sys.executable}" -m pytest personalized_precision_oncology/stage5_genai/integration/tests --import-mode=importlib -v'
    )

    # Summary
    print_banner("STAGE 5 VERIFICATION SUMMARY")
    modules = [
        ("Data Engineering Pipeline", rc1),
        ("Data Engineering Tests (30)", rc2),
        ("EDA & Prompt Engineering Tests (16)", rc3),
        ("GenAI Scenario Generator Tests (23)", rc4),
        ("Evaluation Tests (31)", rc5),
        ("Integration Tests (29)", rc6)
    ]

    all_passed = True
    for name, code in modules:
        status = "PASSED" if code == 0 else "FAILED"
        if code != 0:
            all_passed = False
        print(f" - {name:<42}: {status}")

    print("-" * 70)
    if all_passed:
        print("ALL 129 TESTS AND PIPELINES ARE FULLY FUNCTIONAL AND PASSING (100% SUCCESS)!")
    else:
        print("One or more modules returned errors. Please check the logs above.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
