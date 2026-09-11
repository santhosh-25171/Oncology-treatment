"""
Mutation Co-occurrence Analysis Module.
Analyzes multi-gene and compound mutation combinations observed in the reference baseline.
Audits candidate multi-driver combinations and classifies co-occurrence sparsity without assuming biological impossibility.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ENG_PROCESSED = BASE_DIR / "data_engineering" / "processed"
MUTATION_COOCCURRENCE_JSON = DATA_ENG_PROCESSED / "mutation_cooccurrence.json"
CLEANED_COHORT_CSV = DATA_ENG_PROCESSED / "cleaned_cohort.csv"

# Clinically relevant candidate co-occurrence check targets
CANDIDATE_COOCCURRENCE_CHECK_TARGETS = [
    {
        "genes": "EGFR + KRAS",
        "alteration_type": "Dual canonical oncogenic driver co-mutation",
        "clinical_context": "Classically mutually exclusive; rare co-alteration (<1% NSCLC) with uncertain targeted therapy hierarchy"
    },
    {
        "genes": "ALK + EGFR",
        "alteration_type": "Simultaneous ALK rearrangement and EGFR kinase domain activating mutation",
        "clinical_context": "Rare concurrent drivers (<0.5% NSCLC); creates therapeutic dilemma between ALK vs EGFR TKI"
    },
    {
        "genes": "BRAF + EGFR",
        "alteration_type": "Concurrent BRAF V600E and EGFR Exon 19 del / L858R",
        "clinical_context": "Rare dual driver or bypass mechanism requiring combination BRAF/MEK + EGFR TKI"
    },
    {
        "genes": "KRAS + STK11 + KEAP1",
        "alteration_type": "Pan-negative immunotherapy resistant triad",
        "clinical_context": "Extreme primary immunotherapy resistance and poor survival despite high TMB"
    },
    {
        "genes": "EGFR (L858R + T790M + C797S)",
        "alteration_type": "Triple compound cis/trans EGFR TKI resistance alteration",
        "clinical_context": "Tertiary Osimertinib resistance; cis-C797S is refractory to all approved TKIs"
    },
    {
        "genes": "KRAS (G12C + Y99C)",
        "alteration_type": "Secondary switch II pocket mutation under KRAS G12C inhibition",
        "clinical_context": "Prevents covalent drug binding to cysteine 12; acquired Sotorasib resistance"
    },
    {
        "genes": "ALK (EML4-ALK + G1202R)",
        "alteration_type": "Secondary solvent-front steric hindrance mutation under Alectinib",
        "clinical_context": "Refractory to 1st/2nd gen ALK TKIs; sensitive to 3rd gen Lorlatinib"
    }
]


class CooccurrenceAnalyzer:
    """Analyzes multi-mutation patterns and evaluates candidate co-occurrence check targets."""

    def __init__(self):
        self.cohort_df = pd.read_csv(CLEANED_COHORT_CSV) if CLEANED_COHORT_CSV.exists() else pd.DataFrame()
        self.cooc_data = self._load_json(MUTATION_COOCCURRENCE_JSON)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"status": "source_limitation", "reason": f"File not found: {path.name}"}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def analyze_observed_cooccurrences(self) -> List[Dict[str, Any]]:
        """Extract and categorize observed combinations from the Data Engineering baseline."""
        patterns = self.cooc_data.get("cooccurrence_patterns", [])
        total_cohort = len(self.cohort_df)
        categorized = []

        for p in patterns:
            instances = p.get("observed_instances", 0)
            prev_pct = p.get("prevalence_in_cohort_percent", 0.0)

            # Classify representation
            if instances >= 5:
                status = "well_represented"
            elif instances >= 2:
                status = "rare"
            else:
                status = "sparse"

            categorized.append({
                "combination": p.get("cooccurring_genes"),
                "observed_instances": instances,
                "prevalence_percent": prev_pct,
                "coverage_status": status,
                "clinical_context": p.get("clinical_context", "Observed co-mutation in NSCLC")
            })

        return categorized

    def evaluate_candidate_cooccurrences(self) -> List[Dict[str, Any]]:
        """
        Evaluate candidate multi-driver combinations against actual baseline coverage.
        Uses 'not_observed_in_reference_baseline' instead of 'biologically absent'.
        """
        results = []
        observed_patterns = {
            p["cooccurring_genes"]: p.get("observed_instances", 0)
            for p in self.cooc_data.get("cooccurrence_patterns", [])
        }

        # Also inspect patient-level primary and secondary variants
        patient_pairs = set()
        for _, r in self.cohort_df.iterrows():
            pg = str(r.get("primary_driver_gene", ""))
            sec = str(r.get("secondary_resistance_mutation", ""))
            pm = str(r.get("primary_mutation", ""))
            patient_pairs.add(f"{pg} + {sec}")

        for target in CANDIDATE_COOCCURRENCE_CHECK_TARGETS:
            combo_name = target["genes"]

            # Match against observed patterns
            matched = False
            matched_count = 0
            for obs_name, count in observed_patterns.items():
                # Check normalized overlap
                genes_target = set([g.strip() for g in combo_name.replace("(", "+").replace(")", "+").split("+") if g.strip()])
                genes_obs = set([g.strip() for g in obs_name.replace("(", "+").replace(")", "+").split("+") if g.strip()])
                if genes_target.issubset(genes_obs) or genes_obs.issubset(genes_target):
                    matched = True
                    matched_count = count
                    break

            # Also check text substring in patient records
            if not matched:
                for pp in patient_pairs:
                    if combo_name.split()[0] in pp and (len(combo_name.split()) > 2 and combo_name.split()[2] in pp):
                        matched = True
                        matched_count = 1
                        break

            if matched:
                if matched_count <= 1:
                    cov_status = "sparse"
                    finding = f"Sparsely observed in baseline ({matched_count} instance)."
                elif matched_count <= 3:
                    cov_status = "rare"
                    finding = f"Rarely observed in baseline ({matched_count} instances)."
                else:
                    cov_status = "well_represented"
                    finding = f"Well represented ({matched_count} instances)."
            else:
                cov_status = "not_observed"
                finding = "not_observed_in_reference_baseline"

            results.append({
                "combination": combo_name,
                "alteration_type": target["alteration_type"],
                "clinical_context": target["clinical_context"],
                "coverage_status": cov_status,
                "finding": finding,
                "reason": (
                    f"Combination {combo_name} was evaluated against baseline and is {finding}."
                    if cov_status != "not_observed"
                    else f"Combination {combo_name} is not observed in reference baseline; unobserved in dataset does not imply biological impossibility."
                )
            })

        return results

    def run(self) -> Dict[str, Any]:
        """Execute co-occurrence analysis."""
        return {
            "observed_cooccurrences": self.analyze_observed_cooccurrences(),
            "candidate_cooccurrences_evaluation": self.evaluate_candidate_cooccurrences()
        }
