"""
Scenario Loader for Stage 5 Evaluation.
Reads generated synthetic scenarios from genai/scenarios/synthetic_edge_cases.jsonl
and upstream historical reference baseline and blind spot reports in a strictly read-only manner.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ENG_PROCESSED = BASE_DIR / "data_engineering" / "processed"
EDA_REPORTS_DIR = BASE_DIR / "eda_prompteng" / "reports"
GENAI_SCENARIOS_DIR = BASE_DIR / "genai" / "scenarios"

SYNTHETIC_SCENARIOS_JSONL = GENAI_SCENARIOS_DIR / "synthetic_edge_cases.jsonl"
GENAI_METADATA_JSON = GENAI_SCENARIOS_DIR / "generation_metadata.json"
GENAI_SUMMARY_JSON = GENAI_SCENARIOS_DIR / "generation_summary.json"

BLIND_SPOT_REPORT_JSON = EDA_REPORTS_DIR / "blind_spot_report.json"
GENOMIC_EDA_REPORT_JSON = EDA_REPORTS_DIR / "genomic_eda_report.json"

MUTATION_FREQ_JSON = DATA_ENG_PROCESSED / "mutation_frequency.json"
MUTATION_COOC_JSON = DATA_ENG_PROCESSED / "mutation_cooccurrence.json"
BIOMARKER_DIST_JSON = DATA_ENG_PROCESSED / "biomarker_distributions.json"
RESISTANCE_DIST_JSON = DATA_ENG_PROCESSED / "treatment_resistance_distributions.json"
BASELINE_JSONL = DATA_ENG_PROCESSED / "genai_reference_baseline.jsonl"


class ScenarioLoader:
    """Provides structured, read-only loading of synthetic edge-case scenarios and upstream reference baselines."""

    def __init__(self):
        self.scenarios = self._load_scenarios()
        self.blind_spots_report = self._load_json(BLIND_SPOT_REPORT_JSON)
        self.genomic_eda_report = self._load_json(GENOMIC_EDA_REPORT_JSON)
        self.mutation_frequencies = self._load_json(MUTATION_FREQ_JSON)
        self.mutation_cooccurrences = self._load_json(MUTATION_COOC_JSON)
        self.biomarker_distributions = self._load_json(BIOMARKER_DIST_JSON)
        self.treatment_resistance = self._load_json(RESISTANCE_DIST_JSON)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = []
        if not SYNTHETIC_SCENARIOS_JSONL.exists():
            return scenarios
        with open(SYNTHETIC_SCENARIOS_JSONL, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        scenarios.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Malformed JSON in {SYNTHETIC_SCENARIOS_JSONL.name} line {line_no}: {e}")
        return scenarios

    def get_scenarios(self) -> List[Dict[str, Any]]:
        return self.scenarios

    def get_blind_spots(self) -> List[Dict[str, Any]]:
        return self.blind_spots_report.get("blind_spots", [])

    def get_blind_spot_dict(self) -> Dict[str, Dict[str, Any]]:
        return {bs["blind_spot_id"]: bs for bs in self.get_blind_spots()}
