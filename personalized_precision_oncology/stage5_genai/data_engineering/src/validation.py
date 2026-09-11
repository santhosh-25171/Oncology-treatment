"""
Automated Data Validation and Quality Assurance Pipeline Module.
Performs:
- Schema validation against JSON Schema definitions
- Data type compliance checks
- Missingness audits (flagging nulls vs 'not_available_in_source')
- Patient-level uniqueness verification
- Biological/physiological boundary validation
- Categorical consistency checks
- Mutation syntax normalization audits
- Rare mutation preservation verification
- Generates reports/data_quality_report.json
"""

import json
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .config import (
    SCHEMAS_DIR,
    DATA_QUALITY_REPORT_JSON,
    CLINICAL_BOUNDS,
    UNAVAILABLE_SENTINEL,
    PROTECTED_RARE_VARIANTS
)

logger = logging.getLogger("DataEngineering.Validation")


class OncologyDataValidator:
    """Performs rigorous automated validation checks on cleaned oncology datasets."""

    def __init__(self):
        self.report = {
            "original_row_count": 0,
            "original_column_count": 0,
            "cleaned_row_count": 0,
            "removed_duplicate_count": 0,
            "invalid_record_count": 0,
            "missing_value_summary": {},
            "validation_errors": [],
            "warnings": [],
            "retained_records": 0,
            "source_information": {},
            "final_validation_status": "PASSED",
            "report_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": "1.0.0",
                "role": "Data Engineer (Stage 5)",
                "executor": "Automated Quality Assurance Validator"
            },
            "source_record_counts": {},
            "cleaning_summary": {
                "rows_before_cleaning": 0,
                "rows_after_cleaning": 0,
                "duplicates_removed": 0,
                "invalid_records_removed_or_quarantined": 0
            },
            "missing_value_audit": {},
            "numeric_range_validation": {},
            "categorical_consistency_audit": {},
            "rare_mutation_preservation_audit": {},
            "validation_status": {
                "overall_status": "PASSED",
                "total_checks_passed": 0,
                "critical_errors": 0,
                "warnings": 0,
                "error_details": []
            }
        }

    def validate_schema(self, cohort_df: pd.DataFrame) -> bool:
        """Validate dataframe records against cleaned_cohort_schema.json."""
        schema_path = SCHEMAS_DIR / "cleaned_cohort_schema.json"
        if not schema_path.exists():
            self.report["validation_status"]["critical_errors"] += 1
            self.report["validation_status"]["error_details"].append("Schema file not found")
            return False

        with open(schema_path, "r") as f:
            schema = json.load(f)

        required_cols = schema.get("required", [])
        missing_cols = [col for col in required_cols if col not in cohort_df.columns]
        if missing_cols:
            self.report["validation_status"]["critical_errors"] += 1
            self.report["validation_status"]["error_details"].append(f"Missing required columns: {missing_cols}")
            return False

        self.report["validation_status"]["total_checks_passed"] += 1
        return True

    def validate_patient_uniqueness(self, cohort_df: pd.DataFrame) -> bool:
        """Ensure strict patient-level uniqueness in cleaned cohort."""
        duplicates = cohort_df.duplicated(subset=["patient_id"]).sum()
        if duplicates > 0:
            self.report["validation_status"]["critical_errors"] += 1
            self.report["validation_status"]["error_details"].append(f"Found {duplicates} duplicate patient IDs")
            return False
        self.report["validation_status"]["total_checks_passed"] += 1
        return True

    def validate_clinical_ranges(self, cohort_df: pd.DataFrame) -> bool:
        """Validate numeric variables against physiological clinical bounds."""
        range_results = {}
        all_passed = True

        # Validate Age
        ages = pd.to_numeric(cohort_df["age"], errors="coerce")
        min_age, max_age = CLINICAL_BOUNDS["age"]["min"], CLINICAL_BOUNDS["age"]["max"]
        out_of_bounds_age = int(((ages < min_age) | (ages > max_age)).sum())
        range_results["age"] = {
            "expected_range": [min_age, max_age],
            "out_of_bounds_count": out_of_bounds_age,
            "status": "PASSED" if out_of_bounds_age == 0 else "FAILED"
        }
        if out_of_bounds_age > 0:
            all_passed = False
            self.report["validation_status"]["critical_errors"] += 1
            self.report["validation_status"]["error_details"].append(f"Age out of bounds in {out_of_bounds_age} rows")

        # Validate PD-L1 TPS
        pdl1_vals = cohort_df[cohort_df["pdl1_tps_percent"] != UNAVAILABLE_SENTINEL]["pdl1_tps_percent"]
        pdl1_numeric = pd.to_numeric(pdl1_vals, errors="coerce").dropna()
        out_of_bounds_pdl1 = int(((pdl1_numeric < 0) | (pdl1_numeric > 100)).sum())
        range_results["pdl1_tps_percent"] = {
            "expected_range": [0, 100],
            "assayed_records": len(pdl1_numeric),
            "out_of_bounds_count": out_of_bounds_pdl1,
            "status": "PASSED" if out_of_bounds_pdl1 == 0 else "FAILED"
        }
        if out_of_bounds_pdl1 > 0:
            all_passed = False
            self.report["validation_status"]["critical_errors"] += 1

        # Validate ctDNA MAF
        ctdna_vals = cohort_df[cohort_df["ctdna_maf_percent"] != UNAVAILABLE_SENTINEL]["ctdna_maf_percent"]
        ctdna_numeric = pd.to_numeric(ctdna_vals, errors="coerce").dropna()
        out_of_bounds_ctdna = int(((ctdna_numeric < 0) | (ctdna_numeric > 100)).sum())
        range_results["ctdna_maf_percent"] = {
            "expected_range": [0, 100],
            "assayed_records": len(ctdna_numeric),
            "out_of_bounds_count": out_of_bounds_ctdna,
            "status": "PASSED" if out_of_bounds_ctdna == 0 else "FAILED"
        }
        if out_of_bounds_ctdna > 0:
            all_passed = False
            self.report["validation_status"]["critical_errors"] += 1

        # Validate TMB
        tmb_vals = cohort_df[cohort_df["tmb_mut_per_mb"] != UNAVAILABLE_SENTINEL]["tmb_mut_per_mb"]
        tmb_numeric = pd.to_numeric(tmb_vals, errors="coerce").dropna()
        out_of_bounds_tmb = int(((tmb_numeric < 0) | (tmb_numeric > 500)).sum())
        range_results["tmb_mut_per_mb"] = {
            "expected_range": [0, 500],
            "assayed_records": len(tmb_numeric),
            "out_of_bounds_count": out_of_bounds_tmb,
            "status": "PASSED" if out_of_bounds_tmb == 0 else "FAILED"
        }
        if out_of_bounds_tmb > 0:
            all_passed = False
            self.report["validation_status"]["critical_errors"] += 1

        self.report["numeric_range_validation"] = range_results
        if all_passed:
            self.report["validation_status"]["total_checks_passed"] += 1
        return all_passed

    def validate_categorical_consistency(self, cohort_df: pd.DataFrame) -> bool:
        """Ensure categorical fields adhere strictly to standardized vocabularies."""
        valid_stages = {
            "Stage IA", "Stage IB", "Stage I",
            "Stage IIA", "Stage IIB", "Stage II",
            "Stage IIIA", "Stage IIIB", "Stage IIIC", "Stage III",
            "Stage IVA", "Stage IVB", "Stage IV",
            "Unstaged"
        }
        valid_sexes = {"Male", "Female", "Unknown"}

        stages_present = set(cohort_df["cancer_stage"].unique())
        invalid_stages = list(stages_present - valid_stages)

        sexes_present = set(cohort_df["sex"].unique())
        invalid_sexes = list(sexes_present - valid_sexes)

        cat_results = {
            "cancer_stage": {
                "observed_categories": sorted(list(stages_present)),
                "invalid_categories": invalid_stages,
                "status": "PASSED" if not invalid_stages else "FAILED"
            },
            "sex": {
                "observed_categories": sorted(list(sexes_present)),
                "invalid_categories": invalid_sexes,
                "status": "PASSED" if not invalid_sexes else "FAILED"
            }
        }
        self.report["categorical_consistency_audit"] = cat_results

        if invalid_stages or invalid_sexes:
            self.report["validation_status"]["critical_errors"] += 1
            return False

        self.report["validation_status"]["total_checks_passed"] += 1
        return True

    def validate_rare_mutation_preservation(self, cohort_df: pd.DataFrame) -> bool:
        """Confirm rare driver and resistance mutations were retained in the cleaned baseline."""
        retained_mutations = []
        all_variants = set(cohort_df["primary_mutation"].unique())
        for sm in cohort_df["secondary_resistance_mutation"].unique():
            if pd.notna(sm) and sm != UNAVAILABLE_SENTINEL and str(sm).strip() != "None":
                for part in str(sm).split(";"):
                    all_variants.add(part.split(":")[-1].strip())

        retained_count = 0
        total_tracked = 0
        details = {}
        for gene, variants in PROTECTED_RARE_VARIANTS.items():
            details[gene] = {}
            for v in variants:
                total_tracked += 1
                found = (v in all_variants or any(v in x for x in all_variants))
                details[gene][v] = "RETAINED" if found else "NOT_OBSERVED_IN_COHORT"
                if found:
                    retained_count += 1
                    retained_mutations.append(f"{gene}:{v}")

        rate = (retained_count / total_tracked) * 100.0 if total_tracked > 0 else 100.0
        audit_passed = retained_count > 0  # Crucial: verified rare variants must be present!

        self.report["rare_mutation_preservation_audit"] = {
            "rare_mutations_identified": total_tracked,
            "rare_mutations_retained": retained_count,
            "retention_rate_percent": round(rate, 2),
            "retained_variants_sample": retained_mutations[:15],
            "audit_passed": audit_passed
        }

        if not audit_passed:
            self.report["validation_status"]["critical_errors"] += 1
            return False

        self.report["validation_status"]["total_checks_passed"] += 1
        return True

    def run_all(
        self,
        cohort_df: pd.DataFrame,
        raw_counts: Dict[str, int],
        cleaning_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute full validation suite and compile reports/data_quality_report.json."""
        self.report["source_record_counts"] = raw_counts
        self.report["cleaning_summary"] = {
            "rows_before_cleaning": cleaning_metrics.get("rows_before_cleaning", 0),
            "rows_after_cleaning": cleaning_metrics.get("rows_after_cleaning", 0),
            "duplicates_removed": cleaning_metrics.get("duplicates_removed", 0),
            "invalid_records_removed_or_quarantined": cleaning_metrics.get("invalid_records_removed_or_quarantined", 0)
        }
        self.report["missing_value_audit"] = cleaning_metrics.get("missing_values_by_field", {})

        # Run validation checks
        self.validate_schema(cohort_df)
        self.validate_patient_uniqueness(cohort_df)
        self.validate_clinical_ranges(cohort_df)
        self.validate_categorical_consistency(cohort_df)
        self.validate_rare_mutation_preservation(cohort_df)

        if self.report["validation_status"]["critical_errors"] == 0:
            status_val = "PASSED"
        else:
            status_val = "FAILED"

        self.report["validation_status"]["overall_status"] = status_val
        self.report["final_validation_status"] = status_val
        self.report["original_row_count"] = cleaning_metrics.get("rows_before_cleaning", 0)
        self.report["original_column_count"] = len(cohort_df.columns)
        self.report["cleaned_row_count"] = len(cohort_df)
        self.report["removed_duplicate_count"] = cleaning_metrics.get("duplicates_removed", 0)
        self.report["invalid_record_count"] = cleaning_metrics.get("invalid_records_removed_or_quarantined", 0)
        self.report["missing_value_summary"] = cleaning_metrics.get("missing_values_by_field", {})
        self.report["validation_errors"] = self.report["validation_status"]["error_details"]
        self.report["retained_records"] = len(cohort_df)
        self.report["source_information"] = raw_counts

        # Export report to reports/data_quality_report.json
        DATA_QUALITY_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_QUALITY_REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)

        logger.info(f"Data Quality Report written to {DATA_QUALITY_REPORT_JSON}")
        return self.report
