"""
Raw Data Ingestion Module.
Responsible for reading immutable historical oncology source files and tracking ingestion metrics.
Never modifies original source files.
"""

import pandas as pd
import logging
from typing import Dict, Any, Tuple
from .config import (
    TCGA_CLINICAL_RAW,
    TCGA_MUTATIONS_RAW,
    MSK_IMPACT_RAW,
    SEER_EPIDEMIOLOGY_RAW,
    CLINVAR_CIVIC_RAW,
    BIOMARKERS_EVIDENCE_RAW
)

logger = logging.getLogger("DataEngineering.Ingestion")


class OncologyDataIngestion:
    """Handles raw oncology reference data ingestion."""

    def __init__(self):
        self.raw_counts: Dict[str, int] = {}

    def load_tcga_clinical(self) -> pd.DataFrame:
        """Load raw TCGA clinical records."""
        if not TCGA_CLINICAL_RAW.exists():
            raise FileNotFoundError(f"Missing raw TCGA clinical file: {TCGA_CLINICAL_RAW}")
        df = pd.read_csv(TCGA_CLINICAL_RAW, dtype=str)
        self.raw_counts["tcga_clinical"] = len(df)
        logger.info(f"Loaded {len(df)} raw TCGA clinical records.")
        return df

    def load_tcga_mutations(self) -> pd.DataFrame:
        """Load raw TCGA somatic mutations."""
        if not TCGA_MUTATIONS_RAW.exists():
            raise FileNotFoundError(f"Missing raw TCGA mutations file: {TCGA_MUTATIONS_RAW}")
        df = pd.read_csv(TCGA_MUTATIONS_RAW, dtype=str)
        self.raw_counts["tcga_mutations"] = len(df)
        logger.info(f"Loaded {len(df)} raw TCGA somatic mutation records.")
        return df

    def load_msk_impact(self) -> pd.DataFrame:
        """Load raw MSK-IMPACT NSCLC clinical and targeted panel records."""
        if not MSK_IMPACT_RAW.exists():
            raise FileNotFoundError(f"Missing raw MSK-IMPACT file: {MSK_IMPACT_RAW}")
        df = pd.read_csv(MSK_IMPACT_RAW, dtype=str)
        self.raw_counts["msk_impact"] = len(df)
        logger.info(f"Loaded {len(df)} raw MSK-IMPACT records.")
        return df

    def load_seer_epidemiology(self) -> pd.DataFrame:
        """Load raw SEER epidemiology distributions."""
        if not SEER_EPIDEMIOLOGY_RAW.exists():
            raise FileNotFoundError(f"Missing raw SEER epidemiology file: {SEER_EPIDEMIOLOGY_RAW}")
        df = pd.read_csv(SEER_EPIDEMIOLOGY_RAW, dtype=str)
        self.raw_counts["seer_epidemiology"] = len(df)
        logger.info(f"Loaded {len(df)} raw SEER epidemiology records.")
        return df

    def load_clinvar_civic(self) -> pd.DataFrame:
        """Load raw ClinVar / CIViC somatic resistance evidence."""
        if not CLINVAR_CIVIC_RAW.exists():
            raise FileNotFoundError(f"Missing raw ClinVar/CIViC file: {CLINVAR_CIVIC_RAW}")
        df = pd.read_csv(CLINVAR_CIVIC_RAW, dtype=str)
        self.raw_counts["clinvar_civic"] = len(df)
        logger.info(f"Loaded {len(df)} raw ClinVar/CIViC evidence records.")
        return df

    def load_biomarkers_evidence(self) -> pd.DataFrame:
        """Load raw oncology biomarker reference intervals."""
        if not BIOMARKERS_EVIDENCE_RAW.exists():
            raise FileNotFoundError(f"Missing raw biomarkers evidence file: {BIOMARKERS_EVIDENCE_RAW}")
        df = pd.read_csv(BIOMARKERS_EVIDENCE_RAW, dtype=str)
        self.raw_counts["biomarkers_evidence"] = len(df)
        logger.info(f"Loaded {len(df)} raw biomarker evidence records.")
        return df

    def load_all(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
        """Load all raw sources and return dataframes with record counts."""
        datasets = {
            "tcga_clinical": self.load_tcga_clinical(),
            "tcga_mutations": self.load_tcga_mutations(),
            "msk_impact": self.load_msk_impact(),
            "seer_epidemiology": self.load_seer_epidemiology(),
            "clinvar_civic": self.load_clinvar_civic(),
            "biomarkers_evidence": self.load_biomarkers_evidence()
        }
        return datasets, self.raw_counts
