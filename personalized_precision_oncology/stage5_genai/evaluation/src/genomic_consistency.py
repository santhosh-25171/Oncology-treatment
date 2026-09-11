"""
Genomic Consistency Auditor.
Compares generated genomic profiles against historical reference distributions
and classifies features as:
- OBSERVED_IN_REFERENCE
- RARE_IN_REFERENCE
- NOT_OBSERVED_IN_REFERENCE
- INSUFFICIENT_EVIDENCE
- SOURCE_LIMITATION
"""

from typing import Dict, Any, List, Tuple


class GenomicConsistencyAuditor:
    """Audits genomic alterations against historical baseline distributions."""

    def __init__(self, mutation_frequencies: Dict[str, Any], cooccurrences: Dict[str, Any], treatment_resistance: Dict[str, Any] = None):
        self.mut_freqs = mutation_frequencies
        self.coocs = cooccurrences
        self.treatment_resistance = treatment_resistance or {}
        self._build_indexes()

    def _build_indexes(self):
        self.gene_freq_map = {}
        for item in self.mut_freqs.get("gene_level_frequencies", []):
            self.gene_freq_map[item["gene"].upper()] = float(item.get("frequency_percent", 0.0))

        self.variant_set = set()
        for item in self.mut_freqs.get("variant_level_frequencies", []):
            v_id = item.get("variant_identifier", "")
            if v_id:
                self.variant_set.add(v_id.upper())

        # Include genes observed in co-occurrence baseline
        for item in self.coocs.get("cooccurrence_patterns", []):
            combo = item.get("cooccurring_genes", "")
            for g in combo.split("+"):
                g_clean = g.strip().upper()
                if g_clean and g_clean not in self.gene_freq_map:
                    self.gene_freq_map[g_clean] = float(item.get("prevalence_in_cohort_percent", 1.33))

        # Include curated resistance catalog genes/mechanisms
        self.resistance_catalog = set()
        for item in self.treatment_resistance.get("cohort_observed_resistance", []):
            mech = str(item.get("resistance_mechanism", "")).upper()
            self.resistance_catalog.add(mech)
        for item in self.treatment_resistance.get("civic_clinvar_curated_evidence", []):
            g = item.get("gene", "").upper()
            if g and g not in self.gene_freq_map:
                self.gene_freq_map[g] = 1.33
            v = item.get("variant", "").upper()
            self.resistance_catalog.add(f"{g}:{v}")
            self.resistance_catalog.add(v)

    def audit_genomics(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Classify each alteration and verify non-fabrication of evidence."""
        genomic_profile = scenario.get("genomic_profile", {})
        alterations = genomic_profile.get("alterations", [])
        cooccurrences = genomic_profile.get("cooccurring_alterations", [])
        resistance_features = genomic_profile.get("resistance_related_features", [])

        classifications = {}
        issues = []

        all_alts = []
        for a in alterations:
            all_alts.append(("primary", a))
        for a in cooccurrences:
            all_alts.append(("cooccurring", a))
        for a in resistance_features:
            all_alts.append(("resistance", a))

        for alt_type, alt in all_alts:
            gene = alt.get("gene", "").upper()
            variant = alt.get("variant", "")
            status = alt.get("status", "")
            key = f"{gene}:{variant}" if variant else gene

            # Determine baseline status
            freq = self.gene_freq_map.get(gene, 0.0)

            # Check if alteration is known in resistance catalog
            is_in_resistance_catalog = (
                gene in self.resistance_catalog or 
                variant.upper() in self.resistance_catalog or 
                key.upper() in self.resistance_catalog or
                "HISTOLOGICAL" in gene or "TRANSFORMATION" in gene or "SWITCH" in gene or
                any(gene in item for item in self.resistance_catalog)
            )

            if status == "synthetic_combination_not_observed_in_reference":
                classification = "NOT_OBSERVED_IN_REFERENCE"
            elif "not_observed" in status.lower() or "unobserved" in status.lower():
                classification = "NOT_OBSERVED_IN_REFERENCE"
            elif freq == 0.0 and not is_in_resistance_catalog:
                classification = "NOT_OBSERVED_IN_REFERENCE"
            elif freq <= 3.0 or is_in_resistance_catalog:
                classification = "RARE_IN_REFERENCE"
            else:
                classification = "OBSERVED_IN_REFERENCE"

            classifications[key] = {
                "gene": gene,
                "variant": variant,
                "type": alt_type,
                "baseline_frequency_pct": freq,
                "classification": classification,
                "declared_status": status
            }

            # Integrity check: if claimed historically_observed but freq is 0 and not in variant set or resistance catalog
            if status == "historically_observed" and freq == 0.0 and not is_in_resistance_catalog:
                issues.append(f"Alteration {key} declared as 'historically_observed' but gene is not observed in reference baseline.")

        # Check for unobserved combination labeling rule
        for k, c in classifications.items():
            if c["classification"] == "NOT_OBSERVED_IN_REFERENCE":
                # Must not be labeled as historically observed without synthetic tag
                if c["declared_status"] == "historically_observed":
                    issues.append(f"Unobserved alteration {k} was improperly labeled as 'historically_observed'.")

        is_valid = len(issues) == 0
        return {
            "valid": is_valid,
            "classifications": classifications,
            "issues": issues
        }
