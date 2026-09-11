"""
Scenario Diversity and Near-Duplicate Analyzer.
Analyzes coverage across categories, blind spots, mutations, and detects exact/near duplicates.
"""

from typing import Dict, Any, List, Set


class DiversityAnalyzer:
    """Audits scenario uniqueness, coverage diversity, and duplicate signatures."""

    def analyze_diversity(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates:
        - exact duplicates
        - near duplicates using normalized biological signatures
        - category coverage
        - blind-spot coverage
        - underrepresented stress dimensions
        """
        total = len(scenarios)
        exact_signatures = set()
        exact_duplicates = []

        normalized_signatures = {}
        near_duplicates = []

        category_counts = {}
        blind_spot_counts = {}
        gene_counts = {}
        resistance_counts = {}

        for sc in scenarios:
            sid = sc.get("scenario_id")
            cat = sc.get("scenario_category", "unknown")
            bs_id = sc.get("target_blind_spot", {}).get("blind_spot_id", "unknown")

            category_counts[cat] = category_counts.get(cat, 0) + 1
            blind_spot_counts[bs_id] = blind_spot_counts.get(bs_id, 0) + 1

            # Exact signature (all alterations + stage + prior tx)
            alts = sc.get("genomic_profile", {}).get("alterations", [])
            coocs = sc.get("genomic_profile", {}).get("cooccurring_alterations", [])
            res = sc.get("genomic_profile", {}).get("resistance_related_features", [])

            all_genes = sorted([a.get("gene", "") for a in alts + coocs + res if a.get("gene")])
            for g in all_genes:
                gene_counts[g] = gene_counts.get(g, 0) + 1

            all_res = sorted([r.get("variant", "") for r in res if r.get("variant")])
            for r in all_res:
                resistance_counts[r] = resistance_counts.get(r, 0) + 1

            exact_sig = f"{bs_id}|{cat}|{tuple(sorted([a.get('variant','') for a in alts]))}|{tuple(sorted([r.get('variant','') for r in res]))}"
            if exact_sig in exact_signatures:
                exact_duplicates.append(sid)
            exact_signatures.add(exact_sig)

            # Normalized signature (target gene + resistance gene + histology)
            norm_sig = f"{tuple(all_genes)}|{sc.get('patient_context',{}).get('histology')}"
            if norm_sig in normalized_signatures:
                near_duplicates.append({"original": normalized_signatures[norm_sig], "duplicate": sid, "signature": norm_sig})
            else:
                normalized_signatures[norm_sig] = sid

        return {
            "total_scenarios": total,
            "unique_exact_scenarios": total - len(exact_duplicates),
            "exact_duplicate_count": len(exact_duplicates),
            "exact_duplicates": exact_duplicates,
            "near_duplicate_count": len(near_duplicates),
            "near_duplicates": near_duplicates,
            "category_distribution": category_counts,
            "blind_spot_distribution": blind_spot_counts,
            "unique_blind_spots_covered": len(blind_spot_counts),
            "gene_coverage_frequency": gene_counts,
            "resistance_mechanism_frequency": resistance_counts
        }
