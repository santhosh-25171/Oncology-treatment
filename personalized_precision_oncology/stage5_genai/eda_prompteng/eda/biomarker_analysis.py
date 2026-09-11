"""
Biomarker Coverage and Distribution Analysis Module.
Audits measured biomarkers (TMB, PD-L1 TPS, ctDNA MAF) vs unassayed clinical variables.
Checks feature availability strictly before calculation and detects biomarker blind spots.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ENG_PROCESSED = BASE_DIR / "data_engineering" / "processed"
BIOMARKER_DIST_JSON = DATA_ENG_PROCESSED / "biomarker_distributions.json"
CLEANED_COHORT_CSV = DATA_ENG_PROCESSED / "cleaned_cohort.csv"


class BiomarkerAnalyzer:
    """Analyzes quantitative biomarker coverage, distributions, and missingness."""

    def __init__(self):
        self.cohort_df = pd.read_csv(CLEANED_COHORT_CSV) if CLEANED_COHORT_CSV.exists() else pd.DataFrame()
        self.bio_data = self._load_json(BIOMARKER_DIST_JSON)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"status": "source_limitation", "reason": f"File not found: {path.name}"}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def analyze_tmb_coverage(self) -> Dict[str, Any]:
        """Audit TMB measurement availability and distribution."""
        if "tmb_mut_per_mb" not in self.cohort_df.columns:
            return {
                "status": "source_limitation",
                "reason": "Variable 'tmb_mut_per_mb' not available in supplied reference baseline"
            }

        valid_tmb = pd.to_numeric(self.cohort_df[self.cohort_df["tmb_mut_per_mb"] != "not_available_in_source"]["tmb_mut_per_mb"], errors="coerce").dropna()
        total = len(self.cohort_df)
        assayed = len(valid_tmb)

        if assayed == 0:
            return {
                "status": "source_limitation",
                "reason": "Variable not available in supplied reference baseline"
            }

        # Check high vs low TMB (cutoff 10 mut/Mb based on KEYNOTE-158)
        tmb_high_count = int((valid_tmb >= 10.0).sum())
        tmb_low_count = assayed - tmb_high_count

        return {
            "status": "available",
            "assayed_patients": assayed,
            "unassayed_or_missing_patients": total - assayed,
            "coverage_percent": round((assayed / total) * 100.0, 2),
            "median": round(float(valid_tmb.median()), 2),
            "mean": round(float(valid_tmb.mean()), 2),
            "min": round(float(valid_tmb.min()), 2),
            "max": round(float(valid_tmb.max()), 2),
            "tmb_high_count_gte_10": tmb_high_count,
            "tmb_high_percent": round((tmb_high_count / assayed) * 100.0, 2),
            "tmb_low_count_lt_10": tmb_low_count
        }

    def analyze_pdl1_coverage(self) -> Dict[str, Any]:
        """Audit PD-L1 Tumor Proportion Score (TPS %) availability."""
        if "pdl1_tps_percent" not in self.cohort_df.columns:
            return {
                "status": "source_limitation",
                "reason": "Variable 'pdl1_tps_percent' not available in supplied reference baseline"
            }

        valid_pdl1 = pd.to_numeric(self.cohort_df[self.cohort_df["pdl1_tps_percent"] != "not_available_in_source"]["pdl1_tps_percent"], errors="coerce").dropna()
        total = len(self.cohort_df)
        assayed = len(valid_pdl1)

        if assayed == 0:
            return {
                "status": "source_limitation",
                "reason": "Variable not available in supplied reference baseline"
            }

        neg_count = int((valid_pdl1 < 1.0).sum())
        low_count = int(((valid_pdl1 >= 1.0) & (valid_pdl1 < 50.0)).sum())
        high_count = int((valid_pdl1 >= 50.0).sum())

        return {
            "status": "available",
            "assayed_patients": assayed,
            "unassayed_or_missing_patients": total - assayed,
            "coverage_percent": round((assayed / total) * 100.0, 2),
            "median": round(float(valid_pdl1.median()), 2),
            "mean": round(float(valid_pdl1.mean()), 2),
            "categories": {
                "negative_lt_1pct": {"count": neg_count, "percent": round((neg_count / assayed) * 100.0, 2)},
                "low_1_to_49pct": {"count": low_count, "percent": round((low_count / assayed) * 100.0, 2)},
                "high_gte_50pct": {"count": high_count, "percent": round((high_count / assayed) * 100.0, 2)}
            }
        }

    def analyze_ctdna_coverage(self) -> Dict[str, Any]:
        """Audit ctDNA Mutant Allele Fraction (MAF %) availability."""
        if "ctdna_maf_percent" not in self.cohort_df.columns:
            return {
                "status": "source_limitation",
                "reason": "Variable 'ctdna_maf_percent' not available in supplied reference baseline"
            }

        valid_ctdna = pd.to_numeric(self.cohort_df[self.cohort_df["ctdna_maf_percent"] != "not_available_in_source"]["ctdna_maf_percent"], errors="coerce").dropna()
        total = len(self.cohort_df)
        assayed = len(valid_ctdna)

        if assayed == 0:
            return {
                "status": "source_limitation",
                "reason": "Variable not available in supplied reference baseline"
            }

        return {
            "status": "available",
            "assayed_patients": assayed,
            "unassayed_or_missing_patients": total - assayed,
            "coverage_percent": round((assayed / total) * 100.0, 2),
            "median_maf": round(float(valid_ctdna.median()), 2),
            "mean_maf": round(float(valid_ctdna.mean()), 2),
            "min_maf": round(float(valid_ctdna.min()), 2),
            "max_maf": round(float(valid_ctdna.max()), 2)
        }

    def audit_inflammatory_markers(self) -> Dict[str, Any]:
        """Audit availability of peripheral inflammatory markers (NLR, CRP, LDH)."""
        # Inflammatory markers are available as clinical trial benchmarks in raw evidence,
        # but are not recorded at the patient level in TCGA / MSK-IMPACT reference freezes.
        return {
            "status": "source_limitation",
            "reason": "Patient-level inflammatory markers (NLR, CRP, LDH) not available in supplied reference baseline cohort",
            "published_literature_benchmark_status": "Available in oncology_biomarkers_evidence_raw.csv (KEYNOTE-024, JTO 2018)"
        }

    def run(self) -> Dict[str, Any]:
        """Execute full biomarker coverage analysis."""
        return {
            "tmb_coverage": self.analyze_tmb_coverage(),
            "pdl1_coverage": self.analyze_pdl1_coverage(),
            "ctdna_coverage": self.analyze_ctdna_coverage(),
            "inflammatory_biomarkers_coverage": self.audit_inflammatory_markers()
        }
