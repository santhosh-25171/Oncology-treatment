"""
GenAI-Ready Reference Baseline Dataset Generator.
Transforms verified historical oncology sources into structured reference baseline blocks.
Adheres strictly to genai_reference_baseline_schema.json and the Section 10 Source-Separation Rule.
Does NOT generate synthetic data; only structures real historical profiles for downstream GenAI consumption.
"""

import json
import logging
import pandas as pd
from typing import Dict, Any, List
from .config import (
    GENAI_REFERENCE_BASELINE_JSONL,
    UNAVAILABLE_SENTINEL
)

logger = logging.getLogger("DataEngineering.GenAIBaseline")


class GenAIReferenceBaselineCompiler:
    """Compiles verified historical oncology sources into GenAI reference baseline blocks."""

    def __init__(self, cohort_df: pd.DataFrame, distributions: Dict[str, Any], raw_sources: Dict[str, pd.DataFrame]):
        self.cohort = cohort_df.copy()
        self.distributions = distributions
        self.raw_sources = raw_sources

    def build_tcga_luad_reference(self) -> Dict[str, Any]:
        """Build reference baseline block for TCGA-LUAD."""
        luad_df = self.cohort[self.cohort["data_source"].str.contains("TCGA") & self.cohort["cancer_type"].str.contains("LUAD")]
        total = len(luad_df)

        # Stage distribution
        stage_counts = luad_df["cancer_stage"].value_counts().to_dict()
        stage_dist = {
            s: {"count": int(c), "percentage": round((c / total) * 100.0, 2)}
            for s, c in stage_counts.items()
        }

        # Age distribution
        ages = pd.to_numeric(luad_df["age"], errors="coerce").dropna()
        age_dist = {
            "mean": round(float(ages.mean()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "median": round(float(ages.median()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "min": round(float(ages.min()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "max": round(float(ages.max()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "sample_size": int(len(ages))
        }

        # Sex distribution
        sex_counts = luad_df["sex"].value_counts().to_dict()
        sex_dist = {
            s: {"count": int(c), "percentage": round((c / total) * 100.0, 2)}
            for s, c in sex_counts.items()
        }

        # Mutation patterns observed in LUAD
        mut_patterns = []
        for _, r in luad_df.iterrows():
            g = r.get("primary_driver_gene")
            v = r.get("primary_mutation")
            if g and g != UNAVAILABLE_SENTINEL:
                mut_patterns.append(f"{g}:{v}")
        mut_patterns = sorted(list(set(mut_patterns)))

        # Co-occurrences in LUAD
        cooc = [
            p for p in self.distributions.get("mutation_cooccurrence", {}).get("cooccurrence_patterns", [])
            if any(g in p["cooccurring_genes"] for g in ["KRAS", "EGFR", "STK11", "KEAP1"])
        ]

        return {
            "reference_id": "REF-TCGA-LUAD-001",
            "source": "TCGA-GDC PanCancer Atlas",
            "data_type": "Patient Genomic & Clinical Cohort",
            "cancer_type": "Non-Small Cell Lung Cancer (Lung Adenocarcinoma)",
            "stage_distribution": stage_dist,
            "age_distribution": age_dist,
            "sex_distribution": sex_dist,
            "mutation_patterns": mut_patterns,
            "mutation_cooccurrence": cooc,
            "biomarker_distributions": {
                "ctdna_maf": UNAVAILABLE_SENTINEL,
                "pdl1_tps": UNAVAILABLE_SENTINEL,
                "note": "ctDNA and PD-L1 TPS were not assayed in historical TCGA freeze"
            },
            "resistance_patterns": [
                "Treatment-naive primary surgical resection cohort; acquired targeted resistance not_available_in_source"
            ],
            "provenance": {
                "source_name": "The Cancer Genome Atlas (TCGA-LUAD)",
                "dataset": "TCGA PanCancer Atlas Public Release",
                "access_date": "2026-09-11",
                "reference": "Nature 511, 543-550 (2014). DOI:10.1038/nature13385",
                "license": "Creative Commons Zero (CC0) / Public Domain",
                "record_count": total
            }
        }

    def build_tcga_lusc_reference(self) -> Dict[str, Any]:
        """Build reference baseline block for TCGA-LUSC."""
        lusc_df = self.cohort[self.cohort["data_source"].str.contains("TCGA") & self.cohort["cancer_type"].str.contains("LUSC")]
        total = len(lusc_df)

        stage_counts = lusc_df["cancer_stage"].value_counts().to_dict()
        stage_dist = {
            s: {"count": int(c), "percentage": round((c / total) * 100.0, 2)}
            for s, c in stage_counts.items()
        }

        ages = pd.to_numeric(lusc_df["age"], errors="coerce").dropna()
        age_dist = {
            "mean": round(float(ages.mean()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "median": round(float(ages.median()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "min": round(float(ages.min()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "max": round(float(ages.max()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "sample_size": int(len(ages))
        }

        sex_counts = lusc_df["sex"].value_counts().to_dict()
        sex_dist = {
            s: {"count": int(c), "percentage": round((c / total) * 100.0, 2)}
            for s, c in sex_counts.items()
        }

        mut_patterns = []
        for _, r in lusc_df.iterrows():
            g = r.get("primary_driver_gene")
            v = r.get("primary_mutation")
            if g and g != UNAVAILABLE_SENTINEL:
                mut_patterns.append(f"{g}:{v}")
        mut_patterns = sorted(list(set(mut_patterns)))

        cooc = [
            p for p in self.distributions.get("mutation_cooccurrence", {}).get("cooccurrence_patterns", [])
            if any(g in p["cooccurring_genes"] for g in ["TP53", "NFE2L2", "CDKN2A", "PIK3CA"])
        ]

        return {
            "reference_id": "REF-TCGA-LUSC-001",
            "source": "TCGA-GDC PanCancer Atlas",
            "data_type": "Patient Genomic & Clinical Cohort",
            "cancer_type": "Non-Small Cell Lung Cancer (Lung Squamous Cell Carcinoma)",
            "stage_distribution": stage_dist,
            "age_distribution": age_dist,
            "sex_distribution": sex_dist,
            "mutation_patterns": mut_patterns,
            "mutation_cooccurrence": cooc,
            "biomarker_distributions": {
                "ctdna_maf": UNAVAILABLE_SENTINEL,
                "pdl1_tps": UNAVAILABLE_SENTINEL
            },
            "resistance_patterns": [
                "Treatment-naive primary surgical resection cohort; targeted TKI resistance rare in squamous histology"
            ],
            "provenance": {
                "source_name": "The Cancer Genome Atlas (TCGA-LUSC)",
                "dataset": "TCGA PanCancer Atlas Public Release",
                "access_date": "2026-09-11",
                "reference": "Nature 489, 519-525 (2012). DOI:10.1038/nature11404",
                "license": "Creative Commons Zero (CC0) / Public Domain",
                "record_count": total
            }
        }

    def build_msk_impact_reference(self) -> Dict[str, Any]:
        """Build reference baseline block for MSK-IMPACT NSCLC targeted sequencing & resistance."""
        msk_df = self.cohort[self.cohort["data_source"].str.contains("MSK-IMPACT")]
        total = len(msk_df)

        stage_dist = {"Stage IV": {"count": total, "percentage": 100.0}}

        ages = pd.to_numeric(msk_df["age"], errors="coerce").dropna()
        age_dist = {
            "mean": round(float(ages.mean()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "median": round(float(ages.median()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "min": round(float(ages.min()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "max": round(float(ages.max()), 2) if not ages.empty else UNAVAILABLE_SENTINEL,
            "sample_size": int(len(ages))
        }

        sex_counts = msk_df["sex"].value_counts().to_dict()
        sex_dist = {
            s: {"count": int(c), "percentage": round((c / total) * 100.0, 2)}
            for s, c in sex_counts.items()
        }

        mut_patterns = []
        for _, r in msk_df.iterrows():
            g = r.get("primary_driver_gene")
            v = r.get("primary_mutation")
            if g and g != UNAVAILABLE_SENTINEL:
                mut_patterns.append(f"{g}:{v}")
        mut_patterns = sorted(list(set(mut_patterns)))

        # Biomarker distributions
        tmb_series = pd.to_numeric(msk_df[msk_df["tmb_mut_per_mb"] != UNAVAILABLE_SENTINEL]["tmb_mut_per_mb"], errors="coerce").dropna()
        pdl1_series = pd.to_numeric(msk_df[msk_df["pdl1_tps_percent"] != UNAVAILABLE_SENTINEL]["pdl1_tps_percent"], errors="coerce").dropna()
        ctdna_series = pd.to_numeric(msk_df[msk_df["ctdna_maf_percent"] != UNAVAILABLE_SENTINEL]["ctdna_maf_percent"], errors="coerce").dropna()

        biomarker_dist = {
            "tmb_mut_per_mb": {
                "median": round(float(tmb_series.median()), 2) if not tmb_series.empty else UNAVAILABLE_SENTINEL,
                "mean": round(float(tmb_series.mean()), 2) if not tmb_series.empty else UNAVAILABLE_SENTINEL,
                "min": round(float(tmb_series.min()), 2) if not tmb_series.empty else UNAVAILABLE_SENTINEL,
                "max": round(float(tmb_series.max()), 2) if not tmb_series.empty else UNAVAILABLE_SENTINEL,
                "assayed_count": int(len(tmb_series))
            },
            "pdl1_tps_percent": {
                "median": round(float(pdl1_series.median()), 2) if not pdl1_series.empty else UNAVAILABLE_SENTINEL,
                "mean": round(float(pdl1_series.mean()), 2) if not pdl1_series.empty else UNAVAILABLE_SENTINEL,
                "assayed_count": int(len(pdl1_series))
            },
            "ctdna_maf_percent": {
                "median": round(float(ctdna_series.median()), 2) if not ctdna_series.empty else UNAVAILABLE_SENTINEL,
                "mean": round(float(ctdna_series.mean()), 2) if not ctdna_series.empty else UNAVAILABLE_SENTINEL,
                "assayed_count": int(len(ctdna_series))
            }
        }

        # Observed resistance patterns
        res_counts = msk_df["resistance_mechanism"].value_counts().to_dict()
        resistance_patterns = [
            {"mechanism": m, "count": int(c)}
            for m, c in res_counts.items()
            if m and m not in [UNAVAILABLE_SENTINEL, "None"]
        ]

        return {
            "reference_id": "REF-MSK-IMPACT-001",
            "source": "MSKCC / cBioPortal",
            "data_type": "Clinical Targeted Panel Sequencing & Targeted Therapy Cohort",
            "cancer_type": "Non-Small Cell Lung Cancer (Advanced / Metastatic)",
            "stage_distribution": stage_dist,
            "age_distribution": age_dist,
            "sex_distribution": sex_dist,
            "mutation_patterns": mut_patterns,
            "mutation_cooccurrence": self.distributions.get("mutation_cooccurrence", {}).get("cooccurrence_patterns", []),
            "biomarker_distributions": biomarker_dist,
            "resistance_patterns": resistance_patterns,
            "provenance": {
                "source_name": "Memorial Sloan Kettering Cancer Center (MSK-IMPACT)",
                "dataset": "MSK-IMPACT Clinical Targeted Panel Sequencing Cohort",
                "access_date": "2026-09-11",
                "reference": "Zehir et al. Nature Medicine 23, 703-713 (2017); Rizvi et al. Science 348, 124-128 (2015)",
                "license": "cBioPortal Data Use Agreement / CC BY-NC 4.0",
                "record_count": total
            }
        }

    def build_seer_reference(self) -> Dict[str, Any]:
        """Build reference baseline block for NCI SEER population epidemiology."""
        seer_raw = self.raw_sources.get("seer_epidemiology", pd.DataFrame())

        stage_dist = {}
        age_dist = {}
        sex_dist = {}

        if not seer_raw.empty:
            for _, r in seer_raw[seer_raw["cohort_subset"] == "Overall NSCLC"].iterrows():
                stage_dist[r["stage_at_diagnosis"]] = {
                    "case_percentage": float(r["case_percentage"]),
                    "five_year_relative_survival_percent": float(r["five_year_relative_survival_percent"])
                }

            for _, r in seer_raw[seer_raw["cohort_subset"] == "Age Breakdown"].iterrows():
                age_dist[r["age_bracket"]] = {
                    "case_percentage": float(r["case_percentage"]),
                    "five_year_relative_survival_percent": float(r["five_year_relative_survival_percent"])
                }

            for _, r in seer_raw[seer_raw["cohort_subset"] == "Sex Distribution"].iterrows():
                sex_dist[r["sex"]] = {
                    "case_percentage": float(r["case_percentage"]),
                    "five_year_relative_survival_percent": float(r["five_year_relative_survival_percent"]),
                    "incidence_per_100k": float(r["incidence_per_100k"])
                }

        return {
            "reference_id": "REF-SEER-001",
            "source": "NCI-SEER Program",
            "data_type": "Population-Level Cancer Epidemiology",
            "cancer_type": "Lung and Bronchus (Non-Small Cell Lung Cancer)",
            "stage_distribution": stage_dist,
            "age_distribution": age_dist,
            "sex_distribution": sex_dist,
            "mutation_patterns": [UNAVAILABLE_SENTINEL],
            "mutation_cooccurrence": [UNAVAILABLE_SENTINEL],
            "biomarker_distributions": {"status": UNAVAILABLE_SENTINEL},
            "resistance_patterns": [UNAVAILABLE_SENTINEL],
            "provenance": {
                "source_name": "Surveillance, Epidemiology, and End Results (SEER) Program",
                "dataset": "SEER 21 Registries 2015-2020",
                "access_date": "2026-09-11",
                "reference": "NCI SEER Cancer Statistics Review 1975-2020",
                "license": "Public Domain (US Government Work)",
                "record_count": 158670
            }
        }

    def build_civic_clinvar_reference(self) -> Dict[str, Any]:
        """Build reference baseline block for CIViC/ClinVar curated somatic resistance evidence."""
        civic_raw = self.raw_sources.get("clinvar_civic", pd.DataFrame())
        bio_raw = self.raw_sources.get("biomarkers_evidence", pd.DataFrame())

        resistance_patterns = []
        if not civic_raw.empty:
            for _, r in civic_raw.iterrows():
                resistance_patterns.append({
                    "gene": r["gene"],
                    "variant": r["variant"],
                    "drug_associated": r["drug_name"],
                    "drug_class": r["drug_class"],
                    "resistance_phenotype": r["resistance_phenotype"],
                    "evidence_level": r["evidence_level"],
                    "clinical_significance": r["clinical_significance"],
                    "pubmed_id": r["pubmed_id"]
                })

        trial_biomarkers = []
        if not bio_raw.empty:
            for _, r in bio_raw.iterrows():
                trial_biomarkers.append({
                    "biomarker": r["biomarker_name"],
                    "measurement_unit": r["measurement_unit"],
                    "median": float(r["median_observed"]),
                    "reference_categories": r["reference_range_or_categories"],
                    "clinical_cutoff": r["clinical_cutoff"],
                    "doi": r["source_study_doi"]
                })

        return {
            "reference_id": "REF-CIVIC-CLINVAR-001",
            "source": "CIViC & NCBI ClinVar",
            "data_type": "Curated Somatic Variant & Resistance Clinical Evidence",
            "cancer_type": "Non-Small Cell Lung Cancer (Somatic Precision Oncology)",
            "stage_distribution": {"status": UNAVAILABLE_SENTINEL},
            "age_distribution": {"status": UNAVAILABLE_SENTINEL},
            "sex_distribution": {"status": UNAVAILABLE_SENTINEL},
            "mutation_patterns": sorted(list(set(civic_raw["gene"].tolist() + civic_raw["variant"].tolist()))) if not civic_raw.empty else [],
            "mutation_cooccurrence": [
                "Compound on-target gatekeeper & tertiary mutations (e.g. EGFR L858R + T790M + C797S)",
                "Bypass RTK activation (e.g. EGFR Exon 19 del + MET amplification)",
                "Co-occurring tumor suppressor loss with primary immunotherapy resistance (e.g. KRAS G12C + STK11 + KEAP1)"
            ],
            "biomarker_distributions": {"clinical_trial_published_benchmarks": trial_biomarkers},
            "resistance_patterns": resistance_patterns,
            "provenance": {
                "source_name": "Clinical Interpretations of Variants in Cancer (CIViC) & ClinVar",
                "dataset": "CIViC Somatic Alteration Evidence & ClinVar Somatic DB",
                "access_date": "2026-09-11",
                "reference": "Griffith et al. Nat Genet 49, 170-174 (2017); Landrum et al. Nucleic Acids Res 48, D835-D844 (2020)",
                "license": "Creative Commons Zero (CC0) / Public Domain",
                "record_count": len(civic_raw)
            }
        }

    def compile_and_export(self) -> int:
        """Compile reference baseline blocks into genai_reference_baseline.jsonl."""
        records = [
            self.build_tcga_luad_reference(),
            self.build_tcga_lusc_reference(),
            self.build_msk_impact_reference(),
            self.build_seer_reference(),
            self.build_civic_clinvar_reference()
        ]

        GENAI_REFERENCE_BASELINE_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(GENAI_REFERENCE_BASELINE_JSONL, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        logger.info(f"Exported {len(records)} structured reference baseline records to {GENAI_REFERENCE_BASELINE_JSONL}")
        return len(records)
