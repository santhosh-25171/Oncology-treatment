"""
End-to-End Orchestration Pipeline for Stage 5 Data Engineering.
Executes the full lineage:
Public Historical Oncology Sources
        ↓
Data Ingestion
        ↓
Raw Reference Data
        ↓
Schema Validation
        ↓
Data Cleaning
        ↓
Data Validation
        ↓
Distribution Extraction
        ↓
Historical Reference Baseline
        ↓
GenAI Reference JSONL
        ↓
HANDOFF TO SYNTHETICCASEGENERATOR
"""

import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DataEngineering.Pipeline")

from .config import (
    CLEANED_COHORT_CSV,
    GENAI_REFERENCE_BASELINE_JSONL,
    DATA_QUALITY_REPORT_JSON
)
from .ingestion import OncologyDataIngestion
from .cleaning import OncologyDataCleaner
from .validation import OncologyDataValidator
from .distributions import OncologyDistributionExtractor
from .genai_baseline import GenAIReferenceBaselineCompiler


def run_pipeline() -> int:
    """Execute the complete reproducible Data Engineering pipeline."""
    logger.info("==================================================")
    logger.info("STARTING STAGE 5 DATA ENGINEERING PIPELINE")
    logger.info("Personalized Precision Oncology Reference Baseline")
    logger.info("==================================================")

    # 1. Ingestion
    logger.info("[Step 1/5] Ingesting raw historical oncology sources...")
    ingestion = OncologyDataIngestion()
    raw_datasets, raw_counts = ingestion.load_all()
    for name, count in raw_counts.items():
        logger.info(f"  - Source '{name}': {count} raw records")

    # 2. Cleaning & Standardization
    logger.info("[Step 2/5] Cleaning and standardizing cohort records...")
    cleaner = OncologyDataCleaner()
    cleaned_cohort = cleaner.run(raw_datasets)
    
    # Export cleaned cohort table
    CLEANED_COHORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned_cohort.to_csv(CLEANED_COHORT_CSV, index=False)
    logger.info(f"Cleaned cohort exported to {CLEANED_COHORT_CSV} ({len(cleaned_cohort)} records)")

    # 3. Data Quality & Validation
    logger.info("[Step 3/5] Performing automated data quality validation...")
    validator = OncologyDataValidator()
    validation_report = validator.run_all(
        cohort_df=cleaned_cohort,
        raw_counts=raw_counts,
        cleaning_metrics=cleaner.metrics
    )
    status = validation_report["validation_status"]["overall_status"]
    logger.info(f"Validation Status: {status} (Passed checks: {validation_report['validation_status']['total_checks_passed']}, Critical errors: {validation_report['validation_status']['critical_errors']})")
    if status == "FAILED":
        logger.error(f"Validation failed: {validation_report['validation_status']['error_details']}")
        return 1

    # 4. Distribution Extraction
    logger.info("[Step 4/5] Extracting reference baseline distributions...")
    dist_extractor = OncologyDistributionExtractor(cleaned_cohort, raw_datasets)
    extracted_distributions = dist_extractor.run_all()

    # 5. GenAI-Ready Reference Baseline Compilation
    logger.info("[Step 5/5] Compiling GenAI-ready reference baseline JSONL...")
    genai_compiler = GenAIReferenceBaselineCompiler(cleaned_cohort, extracted_distributions, raw_datasets)
    total_jsonl = genai_compiler.compile_and_export()

    logger.info("==================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Cleaned Records: {len(cleaned_cohort)}")
    logger.info(f"GenAI JSONL Baseline: {GENAI_REFERENCE_BASELINE_JSONL} ({total_jsonl} lines)")
    logger.info(f"Quality Report: {DATA_QUALITY_REPORT_JSON}")
    logger.info("READY FOR HANDOFF TO STAGE 5 SYNTHETICCASEGENERATOR")
    logger.info("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
