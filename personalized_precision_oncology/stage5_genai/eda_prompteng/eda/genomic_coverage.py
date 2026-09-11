"""
Genomic and Clinical Coverage Analysis Module.
Analyzes cohort-level distributions across cancer types, AJCC stages, age, sex,
and sequencing panels, strictly maintaining source separation and auditing feature availability.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Path definitions
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ENG_PROCESSED = BASE_DIR / "data_engineering" / "processed"
DATA_ENG_REPORTS = BASE_DIR / "data_engineering" / "reports"

CLEANED_COHORT_CSV = DATA_ENG_PROCESSED / "cleaned_cohort.csv"
CANCER_TYPE_DIST_JSON = DATA_ENG_PROCESSED / "cancer_type_distribution.json"
STAGE_DIST_JSON = DATA_ENG_PROCESSED / "stage_distribution.json"
AGE_DIST_JSON = DATA_ENG_PROCESSED / "age_distribution.json"
SEX_DIST_JSON = DATA_ENG_PROCESSED / "sex_distribution.json"
GENAI_JSONL = DATA_ENG_PROCESSED / "genai_reference_baseline.jsonl"


class GenomicCoverageAnalyzer:
    """Performs coverage analysis over historical oncology reference baselines."""

    def __init__(self):
        self.cohort_df = self._load_cleaned_cohort()
        self.cancer_dist = self._load_json(CANCER_TYPE_DIST_JSON)
        self.stage_dist = self._load_json(STAGE_DIST_JSON)
        self.age_dist = self._load_json(AGE_DIST_JSON)
        self.sex_dist = self._load_json(SEX_DIST_JSON)

    def _load_cleaned_cohort(self) -> pd.DataFrame:
        if not CLEANED_COHORT_CSV.exists():
            raise FileNotFoundError(f"Missing cleaned cohort: {CLEANED_COHORT_CSV}")
        return pd.read_csv(CLEANED_COHORT_CSV)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"status": "source_limitation", "reason": f"File not found: {path.name}"}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_feature_availability(self, feature_name: str) -> Dict[str, Any]:
        """Verify whether a variable exists in the baseline and quantify missingness."""
        if feature_name not in self.cohort_df.columns:
            return {
                "feature": feature_name,
                "status": "source_limitation",
                "reason": "Variable not available in supplied reference baseline"
            }
        col = self.cohort_df[feature_name]
        missing_count = int((col == "not_available_in_source").sum() + col.isna().sum())
        total = len(col)
        available_count = total - missing_count
        return {
            "feature": feature_name,
            "status": "available" if available_count > 0 else "source_limitation",
            "available_records": available_count,
            "missing_or_unassayed_records": missing_count,
            "coverage_percentage": round((available_count / total) * 100.0, 2) if total > 0 else 0.0
        }

    def analyze_cancer_type_coverage(self) -> Dict[str, Any]:
        """Analyze coverage across NSCLC subtypes with source separation."""
        tcga_luad = len(self.cohort_df[self.cohort_df["data_source"].str.contains("TCGA") & self.cohort_df["cancer_type"].str.contains("LUAD")])
        tcga_lusc = len(self.cohort_df[self.cohort_df["data_source"].str.contains("TCGA") & self.cohort_df["cancer_type"].str.contains("LUSC")])
        msk_advanced = len(self.cohort_df[self.cohort_df["data_source"].str.contains("MSK-IMPACT")])

        total = len(self.cohort_df)
        return {
            "total_cohort_patients": total,
            "source_stratified_counts": {
                "TCGA-LUAD (Primary Resection WES)": tcga_luad,
                "TCGA-LUSC (Primary Resection WES)": tcga_lusc,
                "MSK-IMPACT (Advanced/Metastatic Targeted Panel)": msk_advanced
            },
            "subtype_summary": {
                "adenocarcinoma_count": int(self.cohort_df["histology"].str.lower().str.contains("adeno").sum()),
                "squamous_count": int(self.cohort_df["histology"].str.lower().str.contains("squam").sum()),
                "large_cell_neuroendocrine": int(self.cohort_df["histology"].str.lower().str.contains("neuro|large").sum())
            }
        }

    def analyze_stage_genomic_coverage(self) -> Dict[str, Any]:
        """Analyze stage distribution and cross-tabulate with genomic profiling panels."""
        stages = self.cohort_df["cancer_stage"].value_counts().to_dict()
        stage_by_source = pd.crosstab(self.cohort_df["cancer_stage"], self.cohort_df["data_source"]).to_dict()

        # Check early vs metastatic representation
        early_stages = ["Stage I", "Stage IA", "Stage IB", "Stage II", "Stage IIA", "Stage IIB"]
        locally_advanced = ["Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC"]
        metastatic = ["Stage IV", "Stage IVA", "Stage IVB"]

        early_count = int(self.cohort_df["cancer_stage"].isin(early_stages).sum())
        loc_adv_count = int(self.cohort_df["cancer_stage"].isin(locally_advanced).sum())
        met_count = int(self.cohort_df["cancer_stage"].isin(metastatic).sum())

        return {
            "overall_stage_counts": stages,
            "stage_by_source_crosstab": stage_by_source,
            "stage_group_breakdown": {
                "early_stage_resectable": {"count": early_count, "percent": round((early_count / len(self.cohort_df)) * 100.0, 2)},
                "locally_advanced": {"count": loc_adv_count, "percent": round((loc_adv_count / len(self.cohort_df)) * 100.0, 2)},
                "metastatic_distant": {"count": met_count, "percent": round((met_count / len(self.cohort_df)) * 100.0, 2)}
            }
        }

    def run(self) -> Dict[str, Any]:
        """Execute full genomic coverage analysis."""
        key_features = [
            "age", "sex", "cancer_stage", "histology", "pack_years",
            "vital_status", "overall_survival_months", "primary_driver_gene",
            "tmb_mut_per_mb", "pdl1_tps_percent", "ctdna_maf_percent"
        ]
        feature_audit = {feat: self.check_feature_availability(feat) for feat in key_features}

        return {
            "feature_availability_audit": feature_audit,
            "cancer_type_coverage": self.analyze_cancer_type_coverage(),
            "stage_genomic_coverage": self.analyze_stage_genomic_coverage()
        }
