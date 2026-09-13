"""
Provenance Validator for Evaluation.
Verifies completeness of provenance fields, timestamps, reference sources, and seeds.
"""

from typing import Dict, Any, List, Tuple


class ProvenanceValidator:
    """Audits the provenance metadata block of each scenario."""

    def validate_provenance(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, bool]]:
        """
        Validates provenance fields:
        - reference_sources (non-empty list)
        - blind_spot_source (non-empty string)
        - generation_method ("llm" or "template")
        - generation_timestamp (ISO 8601 string)
        - random_seed (integer)
        """
        issues = []
        checks = {
            "has_reference_sources": False,
            "has_blind_spot_source": False,
            "has_timestamp": False,
            "has_seed": False
        }

        provenance = scenario.get("provenance")
        if not provenance or not isinstance(provenance, dict):
            issues.append("Missing or non-dictionary 'provenance' block.")
            return False, issues, checks

        # reference_sources
        ref_sources = provenance.get("reference_sources")
        if isinstance(ref_sources, list) and len(ref_sources) > 0:
            checks["has_reference_sources"] = True
        else:
            issues.append("Provenance 'reference_sources' must be a non-empty list of source citations.")

        # blind_spot_source
        bs_source = provenance.get("blind_spot_source")
        if isinstance(bs_source, str) and bs_source.strip():
            checks["has_blind_spot_source"] = True
        else:
            issues.append("Provenance 'blind_spot_source' must be a non-empty string.")

        # generation_timestamp
        ts = provenance.get("generation_timestamp")
        if isinstance(ts, str) and ("T" in ts or "-" in ts):
            checks["has_timestamp"] = True
        else:
            issues.append("Provenance 'generation_timestamp' must be a valid ISO timestamp string.")

        # random_seed
        seed = provenance.get("random_seed")
        if isinstance(seed, int):
            checks["has_seed"] = True
        else:
            issues.append("Provenance 'random_seed' must be an integer.")

        # generation_method
        gen_method = provenance.get("generation_method")
        if gen_method not in ["llm", "template", "deterministic_fallback"]:
            issues.append(f"Provenance 'generation_method' must be 'llm', 'template', or 'deterministic_fallback', found '{gen_method}'.")

        is_valid = len(issues) == 0
        return is_valid, issues, checks
