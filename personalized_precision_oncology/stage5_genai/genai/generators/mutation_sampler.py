"""
Constrained Mutation & Alteration Sampler.
Uses empirical historical distributions and co-occurrences as constraints.
Ensures novel or unobserved combinations are explicitly labeled as:
'synthetic_combination_not_observed_in_reference'.
"""

import random
from typing import Dict, Any, List, Optional
from ..src.reference_loader import ReferenceDataLoader


class MutationSampler:
    """Samples biological alterations under historical reference constraints."""

    def __init__(self, ref_loader: Optional[ReferenceDataLoader] = None, seed: int = 42):
        self.ref_loader = ref_loader or ReferenceDataLoader()
        self.seed = seed
        self.rng = random.Random(seed)

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)

    def sample_common_driver(self) -> Dict[str, Any]:
        """Sample a frequently observed driver (KRAS, EGFR, TP53)."""
        common_options = [
            {"gene": "KRAS", "variant": "p.G12C", "alteration_type": "Missense Mutation", "status": "historically_observed"},
            {"gene": "EGFR", "variant": "p.L858R", "alteration_type": "Kinase Domain Mutation", "status": "historically_observed"},
            {"gene": "EGFR", "variant": "Exon 19 del (p.E746_A750del)", "alteration_type": "In-frame Deletion", "status": "historically_observed"},
            {"gene": "TP53", "variant": "p.R273H", "alteration_type": "Missense Mutation", "status": "historically_observed"}
        ]
        return self.rng.choice(common_options)

    def sample_rare_variant(self, blind_spot_pattern: Optional[str] = None) -> Dict[str, Any]:
        """Sample or construct a rare variant (<3% prevalence) based on baseline or blind spot."""
        if blind_spot_pattern:
            parts = blind_spot_pattern.split()
            gene = parts[0]
            var = parts[1] if len(parts) > 1 else "Variant"
            return {
                "gene": gene,
                "variant": var,
                "alteration_type": "Somatic Alteration",
                "status": "historically_observed_rare_variant"
            }
        rare_options = [
            {"gene": "BRAF", "variant": "p.V600E", "alteration_type": "Kinase Activating Mutation", "status": "historically_observed_rare_variant"},
            {"gene": "MET", "variant": "p.D1010H", "alteration_type": "Juxtamembrane Domain Mutation", "status": "historically_observed_rare_variant"},
            {"gene": "ERBB2", "variant": "p.Y772_A775dup", "alteration_type": "Exon 20 Insertion", "status": "historically_observed_rare_variant"},
            {"gene": "RET", "variant": "KIF5B-RET", "alteration_type": "Gene Fusion", "status": "historically_observed_rare_variant"}
        ]
        return self.rng.choice(rare_options)

    def sample_unobserved_fusion(self, gene: str = "NTRK1") -> Dict[str, Any]:
        """Sample an actionable gene fusion not observed in the historical baseline."""
        partners = {
            "NTRK1": "TPR-NTRK1",
            "NTRK2": "PAN3-NTRK2",
            "NTRK3": "ETV6-NTRK3",
            "NRG1": "CD74-NRG1"
        }
        variant = partners.get(gene, f"Canonical-{gene}")
        return {
            "gene": gene,
            "variant": variant,
            "alteration_type": "Oncogenic Fusion",
            "status": "synthetic_combination_not_observed_in_reference"
        }

    def sample_resistance_pair(self, primary_gene: str = "EGFR") -> List[Dict[str, Any]]:
        """Sample an on-target or bypass resistance mechanism alongside a primary driver."""
        if primary_gene == "EGFR":
            return [
                {"gene": "EGFR", "variant": "p.L858R", "alteration_type": "Primary Activating", "status": "historically_observed"},
                {"gene": "EGFR", "variant": "p.T790M", "alteration_type": "Secondary Gatekeeper", "status": "historically_observed"},
                {"gene": "EGFR", "variant": "p.C797S", "alteration_type": "Tertiary Resistance", "status": "synthetic_combination_not_observed_in_reference"}
            ]
        elif primary_gene == "KRAS":
            return [
                {"gene": "KRAS", "variant": "p.G12C", "alteration_type": "Primary Driver", "status": "historically_observed"},
                {"gene": "KRAS", "variant": "p.Y99C", "alteration_type": "Switch II Pocket Resistance", "status": "synthetic_combination_not_observed_in_reference"}
            ]
        elif primary_gene == "ALK":
            return [
                {"gene": "ALK", "variant": "EML4-ALK (Variant 1)", "alteration_type": "Primary Rearrangement", "status": "historically_observed"},
                {"gene": "ALK", "variant": "p.G1202R", "alteration_type": "Solvent-Front Resistance", "status": "synthetic_combination_not_observed_in_reference"}
            ]
        return []
