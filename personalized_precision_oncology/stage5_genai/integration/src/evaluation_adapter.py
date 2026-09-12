"""
Evaluation Adapter for Stage 5 Testing Dashboard.
Interfaces directly with the existing Stage 5 Evaluation module (ScenarioEvaluator)
without modifying any evaluation scoring algorithms, fabricating scores, or altering thresholds.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from personalized_precision_oncology.stage5_genai.evaluation.src.evaluator import ScenarioEvaluator

REPORTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "evaluation"
    / "reports"
)

SCENARIO_AUDIT_JSONL = REPORTS_DIR / "scenario_audit.jsonl"
EVALUATION_SUMMARY_JSON = REPORTS_DIR / "evaluation_summary.json"


class EvaluationAdapter:
    """Bridges the integration testing dashboard to the Stage 5 evaluation engine."""

    def __init__(self, evaluator: Optional[ScenarioEvaluator] = None):
        # Reuse existing ScenarioEvaluator
        self._evaluator = evaluator

    @property
    def evaluator(self) -> ScenarioEvaluator:
        if self._evaluator is None:
            self._evaluator = ScenarioEvaluator()
        return self._evaluator

    def load_cached_audits(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads pre-computed scenario audits from the evaluation report directory if available.
        Returns a mapping of scenario_id -> audit_dict.
        """
        if not SCENARIO_AUDIT_JSONL.exists():
            return {}

        cached: Dict[str, Dict[str, Any]] = {}
        with open(SCENARIO_AUDIT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        sid = record.get("scenario_id")
                        if sid:
                            cached[sid] = record
                    except json.JSONDecodeError:
                        continue
        return cached

    def load_cached_summary(self) -> Optional[Dict[str, Any]]:
        """Loads the pre-computed evaluation summary if available."""
        if not EVALUATION_SUMMARY_JSON.exists():
            return None
        try:
            with open(EVALUATION_SUMMARY_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs live evaluation for a single scenario through the official ScenarioEvaluator.
        """
        sid = scenario.get("scenario_id", "UNKNOWN")
        raw_result = self.evaluator.evaluate_scenario(scenario)

        # Format according to dashboard_result_schema
        bs_audit = raw_result.get("blind_spot_audit", {})
        stress_score = raw_result.get("decision_stress_score", {})
        realism_score = raw_result.get("realism_score", {})

        return {
            "scenario_id": sid,
            "synthetic": True,
            "synthetic_badge": "[SYNTHETIC TEST SCENARIO - RESEARCH ONLY]",
            "evaluation_status": raw_result.get("evaluation_status", "NOT EVALUATED"),
            "decision_stress_score": stress_score,
            "realism_score": realism_score,
            "blind_spot_targeted": bs_audit.get("blind_spot_targeted", "NOT AVAILABLE"),
            "flags": raw_result.get("flags", []),
            "stress_dimensions": raw_result.get("stress_dimensions", []),
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "5.0.0",
            "detailed_audit": raw_result
        }

    def evaluate_batch(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a batch of synthetic scenarios and computes aggregated metrics.
        """
        if not scenarios:
            return {
                "total_scenarios": 0,
                "passed": 0,
                "review": 0,
                "failed": 0,
                "average_stress_score": 0.0,
                "average_realism_score": 0.0,
                "blind_spot_coverage": 0,
                "results": []
            }

        results = []
        pass_count = 0
        review_count = 0
        fail_count = 0
        stress_sum = 0.0
        realism_sum = 0.0
        targeted_blind_spots = set()

        for sc in scenarios:
            res = self.evaluate_scenario(sc)
            results.append(res)
            status = res.get("evaluation_status")
            if status == "PASS":
                pass_count += 1
            elif status == "REVIEW":
                review_count += 1
            else:
                fail_count += 1

            s_val = res.get("decision_stress_score", {}).get("overall_stress_score", 0.0)
            r_val = res.get("realism_score", {}).get("overall_realism_score", 0.0)
            stress_sum += float(s_val) if isinstance(s_val, (int, float)) else 0.0
            realism_sum += float(r_val) if isinstance(r_val, (int, float)) else 0.0

            bs = res.get("blind_spot_targeted")
            if bs and bs != "NOT AVAILABLE":
                targeted_blind_spots.add(bs)

        n = len(scenarios)
        return {
            "total_scenarios": n,
            "passed": pass_count,
            "review": review_count,
            "failed": fail_count,
            "average_stress_score": round(stress_sum / n, 2) if n > 0 else 0.0,
            "average_realism_score": round(realism_sum / n, 2) if n > 0 else 0.0,
            "blind_spot_coverage": len(targeted_blind_spots),
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results
        }
