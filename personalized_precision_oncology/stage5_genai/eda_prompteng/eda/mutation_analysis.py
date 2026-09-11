"""
Mutation Landscape Analysis Module.
Quantifies driver and variant frequencies, detects rare alterations (<3% cohort prevalence),
checks VAF availability, and audits candidate check-targets against the reference baseline.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Set
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ENG_PROCESSED = BASE_DIR / "data_engineering" / "processed"
MUTATION_FREQ_JSON = DATA_ENG_PROCESSED / "mutation_frequency.json"
CLEANED_COHORT_CSV = DATA_ENG_PROCESSED / "cleaned_cohort.csv"

# Clinically relevant NSCLC check targets to audit against dataset coverage
CANDIDATE_GENOMIC_CHECK_TARGETS = [
    {"gene": "NTRK1", "alteration_type": "Fusion", "clinical_utility": "TRK inhibitor sensitivity (Larotrectinib / Entrectinib)"},
    {"gene": "NTRK2", "alteration_type": "Fusion", "clinical_utility": "TRK inhibitor sensitivity"},
    {"gene": "NTRK3", "alteration_type": "Fusion", "clinical_utility": "TRK inhibitor sensitivity"},
    {"gene": "NRG1", "alteration_type": "Fusion", "clinical_utility": "HER3-directed therapy / Zenocutuzumab target"},
    {"gene": "RET", "alteration_type": "Fusion", "clinical_utility": "RET inhibitor sensitivity (Selpercatinib / Pralsetinib)"},
    {"gene": "ERBB2", "alteration_type": "Exon 20 Insertion", "clinical_utility": "HER2 ADC / Trastuzumab deruxtecan"},
    {"gene": "BRAF", "alteration_type": "Class II / Non-V600E Mutation", "clinical_utility": "MEK inhibitor response heterogeneity"},
    {"gene": "EGFR", "alteration_type": "Exon 20 Insertion", "clinical_utility": "EGFR/MET bispecific (Amivantamab) / Mobocertinib"},
    {"gene": "MET", "alteration_type": "High-level Amplification (Gene Copy >= 10)", "clinical_utility": "MET TKI sensitivity / EGFR bypass"}
]


class MutationLandscapeAnalyzer:
    """Analyzes mutation frequency, rare variants, and evaluates candidate check targets."""

    def __init__(self):
        self.cohort_df = pd.read_csv(CLEANED_COHORT_CSV) if CLEANED_COHORT_CSV.exists() else pd.DataFrame()
        self.mut_freq_data = self._load_json(MUTATION_FREQ_JSON)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"status": "source_limitation", "reason": f"File not found: {path.name}"}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_vaf_availability(self) -> Dict[str, Any]:
        """Check whether Variant Allele Fraction (VAF) is available across patient records."""
        # VAF is present in raw TCGA mutations, but not consistently present in cross-cohort cleaned matrix
        if "variant_allele_fraction" in self.cohort_df.columns:
            vaf_series = pd.to_numeric(self.cohort_df["variant_allele_fraction"], errors="coerce").dropna()
            if not vaf_series.empty:
                return {
                    "status": "available",
                    "available_records": len(vaf_series),
                    "mean_vaf": round(float(vaf_series.mean()), 3),
                    "median_vaf": round(float(vaf_series.median()), 3)
                }
        return {
            "status": "source_limitation",
            "reason": "Variable 'variant_allele_fraction' not available in unified patient cohort reference baseline"
        }

    def identify_rare_mutations(self, threshold_percent: float = 3.0) -> List[Dict[str, Any]]:
        """Identify mutations observed with low representation (< threshold_percent)."""
        rare_variants = []
        variant_freqs = self.mut_freq_data.get("variant_level_frequencies", [])
        total_cohort = len(self.cohort_df)

        for item in variant_freqs:
            freq = item.get("frequency_percent", 0.0)
            count = item.get("count", 0)
            if freq <= threshold_percent or count <= 2:
                rare_variants.append({
                    "variant_identifier": item.get("variant_identifier"),
                    "gene": item.get("gene"),
                    "protein_change": item.get("protein_change"),
                    "observed_count": count,
                    "frequency_percent": freq,
                    "classification": "rare"
                })

        return rare_variants

    def evaluate_check_targets(self) -> List[Dict[str, Any]]:
        """
        Evaluate candidate clinical targets against actual dataset coverage.
        Data-driven: never hard-codes blind spots without verifying dataset presence.
        """
        results = []
        observed_genes = set(self.cohort_df["primary_driver_gene"].dropna().unique())
        observed_mutations = set(self.cohort_df["primary_mutation"].dropna().unique())
        for sm in self.cohort_df["secondary_resistance_mutation"].dropna().unique():
            if str(sm) not in ["None", "not_available_in_source", "nan"]:
                for part in str(sm).split(";"):
                    observed_mutations.add(part.strip())

        for target in CANDIDATE_GENOMIC_CHECK_TARGETS:
            gene = target["gene"]
            alt_type = target["alteration_type"]

            # Check if observed in reference cohort
            is_gene_present = gene in observed_genes
            matching_alts = [m for m in observed_mutations if gene in str(m) or alt_type.lower() in str(m).lower()]

            if not is_gene_present and not matching_alts:
                coverage_status = "not_observed"
                finding = "not_observed_in_reference_baseline"
                reason = f"Target {gene} ({alt_type}) was evaluated against baseline and is not observed in reference dataset (0 records)."
            elif len(matching_alts) <= 2:
                coverage_status = "rare"
                finding = f"Observed in reference baseline with rare representation ({len(matching_alts)} instances)."
                reason = f"Target {gene} has limited representation ({len(matching_alts)} records)."
            else:
                coverage_status = "well_represented"
                finding = f"Well represented in reference baseline ({len(matching_alts)} instances)."
                reason = f"Target {gene} has adequate coverage."

            results.append({
                "target_gene": gene,
                "alteration_type": alt_type,
                "clinical_utility": target["clinical_utility"],
                "coverage_status": coverage_status,
                "finding": finding,
                "reason": reason
            })

        return results

    def run(self) -> Dict[str, Any]:
        """Run mutation landscape analysis."""
        return {
            "vaf_feature_audit": self.check_vaf_availability(),
            "rare_mutations": self.identify_rare_mutations(threshold_percent=3.0),
            "candidate_targets_evaluation": self.evaluate_check_targets()
        }
