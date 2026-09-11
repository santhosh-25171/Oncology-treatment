"""
Master Pipeline Orchestrator for Stage 5 GenAI Scenario Generation.
Consumes Stage 5 Data Engineering baselines and EDA blind spots.
Executes scenario generation, strict validation, and exports:
- scenarios/synthetic_edge_cases.jsonl (Exactly 20 records)
- scenarios/generation_metadata.json
- scenarios/generation_summary.json
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone

from ..generators.scenario_generator import ScenarioGenerator
from .scenario_validator import ScenarioValidator
from .reference_loader import ReferenceDataLoader
from .blind_spot_loader import BlindSpotLoader

BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = BASE_DIR / "scenarios"
CONFIG_FILE = BASE_DIR / "config" / "generation_config.json"

OUTPUT_JSONL = SCENARIOS_DIR / "synthetic_edge_cases.jsonl"
OUTPUT_METADATA = SCENARIOS_DIR / "generation_metadata.json"
OUTPUT_SUMMARY = SCENARIOS_DIR / "generation_summary.json"


class GenerationPipeline:
    """Orchestrates end-to-end synthetic scenario generation and validation."""

    def __init__(self, config_path: Path = CONFIG_FILE):
        self.config_path = config_path
        self.config = self._load_config()
        self.validator = ScenarioValidator()
        self.ref_loader = ReferenceDataLoader()
        self.blind_spot_loader = BlindSpotLoader()
        self.generator = ScenarioGenerator(
            mode=self.config.get("generation_mode", "template"),
            seed=self.config.get("random_seed", 42)
        )

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "number_of_scenarios": 20,
            "generation_mode": "template",
            "fallback_mode": "template",
            "temperature": 0.7,
            "max_tokens": 2500,
            "random_seed": 42,
            "require_provenance": true,
            "require_synthetic_label": true,
            "require_blind_spot": true
        }

    def run(self) -> Dict[str, Any]:
        """Execute full scenario generation pipeline."""
        SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
        count = self.config.get("number_of_scenarios", 20)
        mode = self.config.get("generation_mode", "template")

        # 1. Generate scenarios
        raw_scenarios = self.generator.generate(count=count)

        # 2. Validate all scenarios
        validation_results = self.validator.validate_scenario_set(raw_scenarios)
        if validation_results["status"] != "PASS":
            raise ValueError(f"Pipeline validation failed: {validation_results['issues']}")

        # 3. Write synthetic_edge_cases.jsonl
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for sc in raw_scenarios:
                f.write(json.dumps(sc) + "\n")

        # 4. Generate metadata
        blind_spots_used = list({sc["target_blind_spot"]["blind_spot_id"] for sc in raw_scenarios})
        metadata = {
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "generation_method": mode,
            "number_requested": count,
            "number_generated": len(raw_scenarios),
            "number_validated": validation_results["valid_scenarios"],
            "number_rejected": validation_results["rejected_scenarios"],
            "random_seed": self.config.get("random_seed", 42),
            "input_reference_files": [
                "genai_reference_baseline.jsonl",
                "mutation_frequency.json",
                "mutation_cooccurrence.json",
                "biomarker_distributions.json",
                "treatment_resistance_distributions.json",
                "blind_spot_report.json"
            ],
            "blind_spots_used": sorted(blind_spots_used)
        }
        with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 5. Generate summary statistics
        cat_dist = {}
        for sc in raw_scenarios:
            cat = sc.get("scenario_category", "unknown")
            cat_dist[cat] = cat_dist.get(cat, 0) + 1

        bs_cov = {}
        for sc in raw_scenarios:
            bs_id = sc["target_blind_spot"]["blind_spot_id"]
            bs_cov[bs_id] = bs_cov.get(bs_id, 0) + 1

        summary = {
            "total_scenarios": len(raw_scenarios),
            "validated_scenarios": validation_results["valid_scenarios"],
            "rejected_scenarios": validation_results["rejected_scenarios"],
            "generation_method": mode,
            "category_distribution": cat_dist,
            "blind_spot_coverage": bs_cov,
            "rare_pattern_count": sum(1 for sc in raw_scenarios if "rare" in sc["scenario_category"]),
            "compound_mutation_count": sum(1 for sc in raw_scenarios if "compound" in sc["scenario_category"]),
            "resistance_scenario_count": sum(1 for sc in raw_scenarios if "resistance" in sc["scenario_category"]),
            "sparse_evidence_count": sum(1 for sc in raw_scenarios if "sparse" in sc["scenario_category"]),
            "wildcard_count": sum(1 for sc in raw_scenarios if sc["scenario_category"] == "wildcard"),
            "validation_status": validation_results["status"]
        }
        with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Pipeline executed successfully! Output: {OUTPUT_JSONL}")
        return {
            "metadata": metadata,
            "summary": summary
        }


if __name__ == "__main__":
    pipeline = GenerationPipeline()
    pipeline.run()
