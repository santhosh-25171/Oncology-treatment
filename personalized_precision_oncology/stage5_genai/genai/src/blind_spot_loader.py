"""
Genomic Blind-Spot Loader & Prioritizer.
Loads and prioritizes detected genomic blind spots from Stage 5 EDA.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EDA_REPORTS_DIR = BASE_DIR / "eda_prompteng" / "reports"
BLIND_SPOT_REPORT_JSON = EDA_REPORTS_DIR / "blind_spot_report.json"
GENOMIC_EDA_REPORT_JSON = EDA_REPORTS_DIR / "genomic_eda_report.json"
PROMPT_COVERAGE_REPORT_JSON = EDA_REPORTS_DIR / "prompt_coverage_report.json"


class BlindSpotLoader:
    """Loads blind spots from Stage 5 EDA and prioritizes them for synthetic scenario generation."""

    def __init__(self):
        self.blind_spot_report = self._load_json(BLIND_SPOT_REPORT_JSON)
        self.genomic_eda_report = self._load_json(GENOMIC_EDA_REPORT_JSON)
        self.prompt_coverage_report = self._load_json(PROMPT_COVERAGE_REPORT_JSON)
        self.blind_spots = self.blind_spot_report.get("blind_spots", [])

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_blind_spots(self) -> List[Dict[str, Any]]:
        return self.blind_spots

    def get_blind_spot_by_id(self, bs_id: str) -> Dict[str, Any]:
        for bs in self.blind_spots:
            if bs.get("blind_spot_id") == bs_id:
                return bs
        raise KeyError(f"Blind spot {bs_id} not found in {BLIND_SPOT_REPORT_JSON.name}")

    def get_prioritized_blind_spots(self, count: int = 20) -> List[Dict[str, Any]]:
        """
        Prioritizes a diverse collection of blind spots across distinct categories:
        - Rare mutations
        - Missing/unobserved mutations
        - Compound mutations & co-occurrences
        - Targeted drug resistance mechanisms
        - Conflicting biomarker evidence
        - Source limitations & stage-specific gaps
        """
        # Group by category
        by_category = {}
        for bs in self.blind_spots:
            cat = bs.get("category", "other")
            by_category.setdefault(cat, []).append(bs)

        prioritized = []

        # Ensure representation across categories
        # 1. Rare mutations (BS001 to BS006)
        rare = by_category.get("rare_mutation", [])
        prioritized.extend(rare[:5])  # BS001 (BRAF), BS002 (MET), BS003 (TP53), BS004 (ERBB2), BS005 (RET)

        # 2. Missing/unobserved pan-cancer fusions (BS007 to BS010)
        missing = by_category.get("missing_mutation", [])
        prioritized.extend(missing[:3])  # BS007 (NTRK1), BS008 (NTRK2), BS010 (NRG1)

        # 3. Compound / Co-occurrence alterations (BS011 to BS013)
        cooc = by_category.get("mutation_cooccurrence", [])
        prioritized.extend(cooc)  # BS011 (KRAS+STK11+KEAP1), BS012 (KRAS Y99C), BS013 (ALK G1202R)

        # 4. Resistance mechanisms (BS014 to BS020)
        res = by_category.get("resistance_mechanism", [])
        prioritized.extend(res[:6])  # BS014 (C797S), BS015 (MET amp), BS016 (Y99C), BS017 (G1202R), BS018 (SCLC switch), BS020 (ADC payload)

        # 5. Conflicting evidence (BS023)
        conflict = by_category.get("evidence_conflict", [])
        prioritized.extend(conflict)  # BS023 (TMB-High vs PD-L1 0%)

        # 6. Source limitation (BS021, BS022)
        source_lim = by_category.get("source_limitation", [])
        prioritized.extend(source_lim[:1])  # BS021 (ctDNA MAF in early stage)

        # 7. Stage-specific gaps (BS024)
        stage_gap = by_category.get("stage_specific", [])
        prioritized.extend(stage_gap)  # BS024 (Acquired TKI resistance in Stage I/II)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for bs in prioritized:
            bs_id = bs["blind_spot_id"]
            if bs_id not in seen:
                seen.add(bs_id)
                deduped.append(bs)

        return deduped[:count]
