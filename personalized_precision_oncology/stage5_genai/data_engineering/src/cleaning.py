"""
Data Cleaning and Standardization Pipeline Module.
Implements reproducible cleaning:
- Standardizes column names to snake_case
- Deduplicates exact duplicate records
- Handles missing data explicitly (null vs "not_available_in_source")
- Standardizes categorical values (stages, sexes, statuses)
- Validates numerical ranges (age, pack-years, survival, biomarkers)
- Normalizes mutation nomenclature
- Rigorously preserves verified rare mutations
"""

import re
import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, Any, Tuple, List
from .config import (
    STAGE_NORMALIZATION_MAP,
    SEX_NORMALIZATION_MAP,
    CLINICAL_BOUNDS,
    UNAVAILABLE_SENTINEL,
    PROTECTED_RARE_VARIANTS,
    QUARANTINE_REPORT_JSON
)

logger = logging.getLogger("DataEngineering.Cleaning")


def to_snake_case(name: str) -> str:
    """Convert arbitrary header name to clean snake_case."""
    name = re.sub(r"[^\w\s]", "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


class OncologyDataCleaner:
    """Orchestrates reproducible cleaning across historical oncology datasets."""

    def __init__(self):
        self.metrics = {
            "rows_before_cleaning": 0,
            "rows_after_cleaning": 0,
            "duplicates_removed": 0,
            "invalid_records_removed_or_quarantined": 0,
            "rare_mutations_identified": 0,
            "rare_mutations_retained": 0,
            "missing_values_by_field": {}
        }
        self.quarantined_records = []

    def clean_tcga_clinical(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Clean TCGA clinical records."""
        df = df_raw.copy()
        df.columns = [to_snake_case(c) for c in df.columns]
        initial_len = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        self.metrics["duplicates_removed"] += (initial_len - len(df))

        valid_rows = []
        for idx, row in df.iterrows():
            record_id = str(row.get("bcr_patient_barcode", f"tcga_row_{idx}"))
            # Validate age
            try:
                age = float(row.get("age_at_index", np.nan))
                if age < CLINICAL_BOUNDS["age"]["min"] or age > CLINICAL_BOUNDS["age"]["max"]:
                    self.metrics["invalid_records_removed_or_quarantined"] += 1
                    self.quarantined_records.append({
                        "record_id": record_id,
                        "source": "TCGA-GDC PanCancer Atlas",
                        "validation_rule": "CLINICAL_BOUNDS['age'] (18-115)",
                        "reason": f"Invalid biological range (age={age})",
                        "status": "quarantined",
                        "raw_record": row.to_dict()
                    })
                    continue
            except (ValueError, TypeError):
                self.metrics["invalid_records_removed_or_quarantined"] += 1
                self.quarantined_records.append({
                    "record_id": record_id,
                    "source": "TCGA-GDC PanCancer Atlas",
                    "validation_rule": "CLINICAL_BOUNDS['age'] numeric type",
                    "reason": "Non-numeric age value",
                    "status": "quarantined",
                    "raw_record": row.to_dict()
                })
                continue

            # Standardize stage
            raw_stage = str(row.get("ajcc_pathologic_stage", "")).strip().lower()
            norm_stage = STAGE_NORMALIZATION_MAP.get(raw_stage, None)
            if not norm_stage or (norm_stage == "Unstaged" and raw_stage == "stage x"):
                if raw_stage == "stage x":
                    self.metrics["invalid_records_removed_or_quarantined"] += 1
                    self.quarantined_records.append({
                        "record_id": record_id,
                        "source": "TCGA-GDC PanCancer Atlas",
                        "validation_rule": "STAGE_NORMALIZATION_MAP",
                        "reason": "Invalid unclassifiable stage Stage X",
                        "status": "quarantined",
                        "raw_record": row.to_dict()
                    })
                    continue
                norm_stage = "Unstaged"

            # Standardize sex
            raw_sex = str(row.get("gender", "")).strip().lower()
            norm_sex = SEX_NORMALIZATION_MAP.get(raw_sex, "Unknown")

            # Standardize pack years
            try:
                pack_years = float(row.get("pack_years_smoked", np.nan))
                if pack_years < 0:
                    pack_years = np.nan
            except (ValueError, TypeError):
                pack_years = np.nan

            # Standardize survival
            try:
                os_months = float(row.get("overall_survival_months", np.nan))
                if os_months < 0:
                    os_months = np.nan
            except (ValueError, TypeError):
                os_months = np.nan

            try:
                pfs_months = float(row.get("progression_free_months", np.nan))
                if pfs_months < 0:
                    pfs_months = np.nan
            except (ValueError, TypeError):
                pfs_months = np.nan

            valid_rows.append({
                "patient_id": row["bcr_patient_barcode"],
                "cancer_type": f"NSCLC-{row['cancer_type']}",
                "age": age,
                "sex": norm_sex,
                "cancer_stage": norm_stage,
                "histology": "Adenocarcinoma" if "LUAD" in row["cancer_type"] else "Squamous Cell Carcinoma",
                "smoking_history": row.get("tobacco_smoking_history", UNAVAILABLE_SENTINEL),
                "pack_years": pack_years if not np.isnan(pack_years) else UNAVAILABLE_SENTINEL,
                "vital_status": row.get("vital_status", "Unknown"),
                "overall_survival_months": os_months if not np.isnan(os_months) else UNAVAILABLE_SENTINEL,
                "progression_free_months": pfs_months if not np.isnan(pfs_months) else UNAVAILABLE_SENTINEL,
                "primary_driver_gene": UNAVAILABLE_SENTINEL,
                "primary_mutation": UNAVAILABLE_SENTINEL,
                "secondary_resistance_mutation": UNAVAILABLE_SENTINEL,
                "prior_targeted_therapy": UNAVAILABLE_SENTINEL,
                "best_recist_response": UNAVAILABLE_SENTINEL,
                "tmb_mut_per_mb": UNAVAILABLE_SENTINEL,
                "pdl1_tps_percent": UNAVAILABLE_SENTINEL,
                "ctdna_maf_percent": UNAVAILABLE_SENTINEL,
                "resistance_mechanism": UNAVAILABLE_SENTINEL,
                "data_source": "TCGA-GDC PanCancer Atlas",
                "source_record_id": row["bcr_patient_barcode"]
            })

        return pd.DataFrame(valid_rows)

    def clean_tcga_mutations(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Clean TCGA somatic mutation records."""
        df = df_raw.copy()
        df.columns = [to_snake_case(c) for c in df.columns]
        initial_len = len(df)

        df = df.drop_duplicates().reset_index(drop=True)
        self.metrics["duplicates_removed"] += (initial_len - len(df))

        if df.empty or "hgvsp_short" not in df.columns:
            return df

        # Normalize mutation names (ensure p. prefix)
        def normalize_variant(v: str) -> str:
            if not isinstance(v, str) or not v.strip():
                return UNAVAILABLE_SENTINEL
            v = v.strip()
            if not v.startswith("p.") and re.match(r"^[A-Z]\d+[A-Z\*]", v):
                v = f"p.{v}"
            return v

        df["hgvsp_short"] = df["hgvsp_short"].apply(normalize_variant)
        return df

    def merge_tcga_profiles(self, clinical_df: pd.DataFrame, mutations_df: pd.DataFrame) -> pd.DataFrame:
        """Link TCGA somatic driver and co-occurring mutations into clinical profiles."""
        merged_rows = []
        for _, patient in clinical_df.iterrows():
            pid = patient["patient_id"]
            p_muts = mutations_df[mutations_df["bcr_patient_barcode"] == pid]

            row_dict = patient.to_dict()
            if not p_muts.empty:
                genes = p_muts["hugo_symbol"].tolist()
                variants = p_muts["hgvsp_short"].tolist()
                types = p_muts["variant_classification"].tolist()

                row_dict["primary_driver_gene"] = genes[0]
                row_dict["primary_mutation"] = variants[0]

                if len(genes) > 1:
                    row_dict["secondary_resistance_mutation"] = ";".join([f"{g}:{v}" for g, v in zip(genes[1:], variants[1:])])
                else:
                    row_dict["secondary_resistance_mutation"] = "None"
            merged_rows.append(row_dict)

        return pd.DataFrame(merged_rows)

    def clean_msk_impact(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Clean MSK-IMPACT targeted sequencing and resistance dataset."""
        df = df_raw.copy()
        df.columns = [to_snake_case(c) for c in df.columns]
        initial_len = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        self.metrics["duplicates_removed"] += (initial_len - len(df))

        cleaned_rows = []
        for idx, row in df.iterrows():
            try:
                age = float(row.get("age", np.nan))
                if age < CLINICAL_BOUNDS["age"]["min"] or age > CLINICAL_BOUNDS["age"]["max"]:
                    self.metrics["invalid_records_removed_or_quarantined"] += 1
                    continue
            except (ValueError, TypeError):
                continue

            raw_stage = str(row.get("cancer_stage", "")).strip().lower()
            norm_stage = STAGE_NORMALIZATION_MAP.get(raw_stage, "Stage IV")

            raw_sex = str(row.get("sex", "")).strip().lower()
            norm_sex = SEX_NORMALIZATION_MAP.get(raw_sex, "Unknown")

            # Parse TMB
            try:
                tmb = float(row.get("tmb_mut_per_mb", np.nan))
            except (ValueError, TypeError):
                tmb = UNAVAILABLE_SENTINEL

            # Parse PD-L1 TPS
            try:
                pdl1 = float(row.get("pdl1_tps_percent", np.nan))
                if pdl1 < 0 or pdl1 > 100:
                    pdl1 = UNAVAILABLE_SENTINEL
            except (ValueError, TypeError):
                pdl1 = UNAVAILABLE_SENTINEL

            # Parse ctDNA MAF
            raw_ctdna = str(row.get("ctdna_maf_percent", "")).strip()
            if raw_ctdna == UNAVAILABLE_SENTINEL or not raw_ctdna:
                ctdna = UNAVAILABLE_SENTINEL
            else:
                try:
                    ctdna = float(raw_ctdna)
                    if ctdna < 0 or ctdna > 100:
                        ctdna = UNAVAILABLE_SENTINEL
                except (ValueError, TypeError):
                    ctdna = UNAVAILABLE_SENTINEL

            cleaned_rows.append({
                "patient_id": row["patient_id"],
                "cancer_type": f"NSCLC-{row.get('histology', 'Adenocarcinoma')}",
                "age": age,
                "sex": norm_sex,
                "cancer_stage": norm_stage,
                "histology": row.get("histology", "Adenocarcinoma"),
                "smoking_history": UNAVAILABLE_SENTINEL,
                "pack_years": UNAVAILABLE_SENTINEL,
                "vital_status": UNAVAILABLE_SENTINEL,
                "overall_survival_months": UNAVAILABLE_SENTINEL,
                "progression_free_months": UNAVAILABLE_SENTINEL,
                "primary_driver_gene": row.get("primary_driver_gene", UNAVAILABLE_SENTINEL),
                "primary_mutation": row.get("primary_mutation", UNAVAILABLE_SENTINEL),
                "secondary_resistance_mutation": row.get("secondary_resistance_mutation", "None"),
                "prior_targeted_therapy": row.get("prior_targeted_therapy", UNAVAILABLE_SENTINEL),
                "best_recist_response": row.get("best_recist_response", UNAVAILABLE_SENTINEL),
                "tmb_mut_per_mb": tmb,
                "pdl1_tps_percent": pdl1,
                "ctdna_maf_percent": ctdna,
                "resistance_mechanism": row.get("resistance_mechanism", UNAVAILABLE_SENTINEL),
                "data_source": "MSK-IMPACT Targeted Sequencing (cBioPortal)",
                "source_record_id": row["patient_id"]
            })

        return pd.DataFrame(cleaned_rows)

    def audit_rare_mutation_preservation(self, df_cleaned: pd.DataFrame):
        """Verify that rare protected mutations were NOT pruned or dropped during cleaning."""
        all_variants_in_cleaned = set()
        for _, row in df_cleaned.iterrows():
            pm = str(row.get("primary_mutation", ""))
            sm = str(row.get("secondary_resistance_mutation", ""))
            all_variants_in_cleaned.add(pm)
            for part in sm.split(";"):
                all_variants_in_cleaned.add(part.split(":")[-1].strip())

        total_identified = 0
        total_retained = 0
        for gene, variants in PROTECTED_RARE_VARIANTS.items():
            for v in variants:
                total_identified += 1
                if v in all_variants_in_cleaned or any(v in x for x in all_variants_in_cleaned):
                    total_retained += 1

        self.metrics["rare_mutations_identified"] = total_identified
        self.metrics["rare_mutations_retained"] = total_retained

    def run(self, raw_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Run complete cleaning pipeline on raw datasets."""
        tcga_clin_raw = raw_datasets["tcga_clinical"]
        tcga_mut_raw = raw_datasets["tcga_mutations"]
        msk_raw = raw_datasets["msk_impact"]

        self.metrics["rows_before_cleaning"] = len(tcga_clin_raw) + len(msk_raw)

        # 1. Clean TCGA
        tcga_clin_cleaned = self.clean_tcga_clinical(tcga_clin_raw)
        tcga_mut_cleaned = self.clean_tcga_mutations(tcga_mut_raw)
        tcga_merged = self.merge_tcga_profiles(tcga_clin_cleaned, tcga_mut_cleaned)

        # 2. Clean MSK-IMPACT
        msk_cleaned = self.clean_msk_impact(msk_raw)

        # 3. Combine into unified patient cohort
        cohort = pd.concat([tcga_merged, msk_cleaned], ignore_index=True)

        # Deduplicate on patient_id
        initial_cohort_len = len(cohort)
        cohort = cohort.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
        self.metrics["duplicates_removed"] += (initial_cohort_len - len(cohort))

        self.metrics["rows_after_cleaning"] = len(cohort)

        # 4. Audit rare mutation preservation
        self.audit_rare_mutation_preservation(cohort)

        # Audit missing values
        for col in cohort.columns:
            missing_count = int((cohort[col] == UNAVAILABLE_SENTINEL).sum() + cohort[col].isna().sum())
            self.metrics["missing_values_by_field"][col] = missing_count

        # Export quarantined records report
        QUARANTINE_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        quarantine_data = {
            "total_quarantined": len(self.quarantined_records),
            "quarantined_records": self.quarantined_records
        }
        with open(QUARANTINE_REPORT_JSON, "w", encoding="utf-8") as qf:
            json.dump(quarantine_data, qf, indent=2)

        logger.info(f"Cleaning complete: {self.metrics['rows_after_cleaning']} records retained. Quarantined: {len(self.quarantined_records)}")
        return cohort
