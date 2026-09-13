"""
Evaluation History Manager for Stage 5 Integration Layer.
Provides append-only persistence of all scenario evaluation executions,
enabling longitudinal tracking, run comparisons, and audit reproducibility.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

HISTORY_DIR = Path(__file__).resolve().parent.parent / "history"
DEFAULT_HISTORY_FILE = HISTORY_DIR / "evaluation_history.jsonl"


class HistoryManager:
    """Manages append-only persistence of evaluation results."""

    def __init__(self, history_file: Optional[Path] = None):
        self.history_file = Path(history_file) if history_file else DEFAULT_HISTORY_FILE
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def record_evaluation(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appends an evaluation result to the history log.
        Computes run_number incrementally for the scenario.
        """
        sid = evaluation_result.get("scenario_id", "UNKNOWN")
        existing = self.get_history_for_scenario(sid)
        run_number = len(existing) + 1

        stress_score = evaluation_result.get("decision_stress_score", {})
        if isinstance(stress_score, dict):
            stress_val = stress_score.get("overall_stress_score")
        else:
            stress_val = stress_score

        realism_score = evaluation_result.get("realism_score", {})
        if isinstance(realism_score, dict):
            realism_val = realism_score.get("overall_realism_score")
        else:
            realism_val = realism_score

        sq = evaluation_result.get("synthetic_quality", {})
        gen_meta = evaluation_result.get("generation_metadata", {})
        seed_comp = evaluation_result.get("seed_compliance", {})

        record = {
            "scenario_id": sid,
            "run_number": run_number,
            "run_id": f"RUN_{sid}_{run_number:03d}",
            "evaluation_status": evaluation_result.get("evaluation_status", "NOT EVALUATED"),
            "decision_stress_score": stress_val,
            "realism_score": realism_val,
            "blind_spot_targeted": evaluation_result.get("blind_spot_targeted", "NOT AVAILABLE"),
            "flags": evaluation_result.get("flags", []),
            "synthetic_quality": sq,
            "seed_compliance": seed_comp,
            "generation_source": gen_meta.get("generation_source") or evaluation_result.get("generation_source", "TEMPLATE"),
            "model": gen_meta.get("model") or evaluation_result.get("model", "deterministic_v5"),
            "seed_conditions": gen_meta.get("seed_conditions") or evaluation_result.get("seed_conditions"),
            "evaluation_timestamp": evaluation_result.get("evaluation_timestamp", datetime.now(timezone.utc).isoformat()),
            "pipeline_version": evaluation_result.get("pipeline_version", "5.0.0")
        }

        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def record_batch(self, batch_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Appends all evaluation results from a batch execution."""
        recorded = []
        for res in batch_result.get("results", []):
            rec = self.record_evaluation(res)
            recorded.append(rec)
        return recorded

    def get_all_history(self) -> List[Dict[str, Any]]:
        """Retrieves all historical evaluation records in chronological order."""
        if not self.history_file.exists():
            return []

        records = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                clean = line.strip()
                if clean:
                    try:
                        records.append(json.loads(clean))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_history_for_scenario(self, scenario_id: str) -> List[Dict[str, Any]]:
        """Retrieves historical runs for a specific scenario ID."""
        all_records = self.get_all_history()
        return [r for r in all_records if r.get("scenario_id") == scenario_id]

    def seed_initial_history(self, initial_audits: List[Dict[str, Any]]) -> int:
        """
        Seeds baseline history if history file does not exist or is empty.
        Returns count of seeded records.
        """
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return 0  # Already seeded

        count = 0
        for audit in initial_audits:
            self.record_evaluation(audit)
            count += 1
        return count
