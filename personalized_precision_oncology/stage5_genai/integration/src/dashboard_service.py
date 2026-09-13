"""
Dashboard Service for Stage 5 Synthetic Oncology Scenario Testing.
Encapsulates business logic for scenario loading, live/batch evaluation,
dynamic filtering, change detection, history tracking, and KPI aggregation.
"""

import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .scenario_loader import ScenarioLoader
from .scenario_adapter import ScenarioAdapter
from .evaluation_adapter import EvaluationAdapter
from .history_manager import HistoryManager
from personalized_precision_oncology.stage5_genai.genai.generators.interactive_generator import InteractivePatientGenerator


class DashboardService:
    """Core coordinator for Stage 5 scenario integration and dashboard operations."""

    def __init__(
        self,
        loader: Optional[ScenarioLoader] = None,
        eval_adapter: Optional[EvaluationAdapter] = None,
        history_mgr: Optional[HistoryManager] = None,
        generator: Optional[InteractivePatientGenerator] = None
    ):
        self.loader = loader or ScenarioLoader()
        self.eval_adapter = eval_adapter or EvaluationAdapter()
        self.history_mgr = history_mgr or HistoryManager()
        self.generator = generator or InteractivePatientGenerator()

        # In-memory caches for fast dashboard responsiveness
        self._scenarios: List[Dict[str, Any]] = []
        self._generated_scenarios: List[Dict[str, Any]] = []
        self._rejected: List[Dict[str, Any]] = []
        self._latest_evaluations: Dict[str, Dict[str, Any]] = {}
        self._last_loaded_mtime: float = 0.0
        self._last_loaded_hash: str = ""
        self._last_eval_time: Optional[str] = None

        # Initial load and seeding
        self.reload_scenarios()
        self._init_evaluations()

    def _compute_file_hash(self) -> str:
        """Computes SHA-256 of the input scenario file."""
        p = self.loader.scenario_path
        if not p.exists():
            return ""
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def reload_scenarios(self) -> Dict[str, Any]:
        """Loads scenarios from disk and updates caching state."""
        import json
        valid, rejected = self.loader.load_scenarios()
        self._scenarios = valid
        self._rejected = rejected
        self._generated_scenarios = []

        # Load dynamically generated synthetic scenarios into separate container
        gen_path = self.generator.scenarios_file
        if gen_path.exists():
            seen_ids = {s.get("scenario_id") for s in self._scenarios}
            with open(gen_path, "r", encoding="utf-8") as f:
                for line in f:
                    clean = line.strip()
                    if clean:
                        try:
                            record = json.loads(clean)
                            sid = record.get("scenario_id")
                            if sid and sid not in seen_ids and record.get("synthetic") is True:
                                seen_ids.add(sid)
                                self._generated_scenarios.append(record)
                        except Exception:
                            pass

        p = self.loader.scenario_path
        if p.exists():
            self._last_loaded_mtime = p.stat().st_mtime
            self._last_loaded_hash = self._compute_file_hash()

        return {
            "loaded_count": len(self._scenarios),
            "rejected_count": len(self._rejected),
            "file_hash": self._last_loaded_hash
        }

    def _init_evaluations(self):
        """Initializes evaluation state from cached reports or existing history."""
        cached_audits = self.eval_adapter.load_cached_audits()
        if cached_audits:
            for sid, audit in cached_audits.items():
                self._latest_evaluations[sid] = self.eval_adapter.evaluate_scenario(
                    self.loader.get_scenario_by_id(sid) or {"scenario_id": sid}
                ) if not audit else audit

        # Pre-seed history if empty
        if cached_audits:
            seed_items = []
            for sid, audit in cached_audits.items():
                seed_items.append({
                    "scenario_id": sid,
                    "evaluation_status": audit.get("evaluation_status", "NOT EVALUATED"),
                    "decision_stress_score": audit.get("decision_stress_score", {}),
                    "realism_score": audit.get("realism_score", {}),
                    "blind_spot_targeted": audit.get("blind_spot_audit", {}).get("blind_spot_targeted", "NOT AVAILABLE"),
                    "flags": audit.get("flags", []),
                    "evaluation_timestamp": audit.get("audit_timestamp", datetime.now(timezone.utc).isoformat()),
                    "pipeline_version": "5.0.0"
                })
            self.history_mgr.seed_initial_history(seed_items)

    def check_for_updates(self) -> bool:
        """
        Continuous evaluation hook: checks if scenario file has changed on disk.
        Returns True if a reload occurred.
        """
        p = self.loader.scenario_path
        if not p.exists():
            return False

        current_mtime = p.stat().st_mtime
        if current_mtime != self._last_loaded_mtime:
            current_hash = self._compute_file_hash()
            if current_hash != self._last_loaded_hash:
                self.reload_scenarios()
                return True
        return False

    def get_scenarios(
        self,
        category: Optional[str] = None,
        blind_spot: Optional[str] = None,
        status: Optional[str] = None,
        uncertainty: Optional[str] = None,
        method: Optional[str] = None,
        include_generated: bool = False
    ) -> List[Dict[str, Any]]:
        """Returns filtered list of dashboard scenario summaries."""
        # Auto-detect file changes
        self.check_for_updates()

        source_scenarios = list(self._scenarios)
        if include_generated or (method and method.lower() in ["llm", "deterministic_fallback"]):
            source_scenarios.extend(self._generated_scenarios)

        items = []
        for sc in source_scenarios:
            sid = sc.get("scenario_id")
            eval_data = self._latest_evaluations.get(sid)
            item = ScenarioAdapter.to_dashboard_item(sc, eval_data)

            # Filtering logic
            if category and item["scenario_category"].lower() != category.lower():
                continue
            if blind_spot and item["target_blind_spot"].upper() != blind_spot.upper():
                continue
            if status and item["evaluation_status"].upper() != status.upper():
                continue
            if uncertainty and item["uncertainty_level"].lower() != uncertainty.lower():
                continue
            if method and item["generation_method"].lower() != method.lower():
                continue

            items.append(item)

        return items

    def get_scenario_detail(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Returns comprehensive detail view for a specific scenario."""
        self.check_for_updates()
        all_sc = self._scenarios + self._generated_scenarios
        sc = next((s for s in all_sc if s.get("scenario_id") == scenario_id), None)
        if not sc:
            return None

        eval_data = self._latest_evaluations.get(scenario_id)
        return ScenarioAdapter.to_detailed_view(sc, eval_data)

    def evaluate_single_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """
        Executes evaluation for a single scenario, updates state and appends to history.
        """
        self.check_for_updates()
        all_sc = self._scenarios + self._generated_scenarios
        sc = next((s for s in all_sc if s.get("scenario_id") == scenario_id), None)
        if not sc:
            raise ValueError(f"Scenario '{scenario_id}' not found in loaded dataset.")

        # Live evaluation through adapter
        result = self.eval_adapter.evaluate_scenario(sc)

        # Cache latest evaluation
        self._latest_evaluations[scenario_id] = result.get("detailed_audit", result)
        self._last_eval_time = result.get("evaluation_timestamp")

        # Record in append-only history
        history_record = self.history_mgr.record_evaluation(result)
        result["history_record"] = history_record

        return result

    def evaluate_all_scenarios(self) -> Dict[str, Any]:
        """
        Executes live batch evaluation for all loaded scenarios.
        """
        self.check_for_updates()
        if not self._scenarios:
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

        batch_result = self.eval_adapter.evaluate_batch(self._scenarios)

        # Update cached latest evaluations
        for res in batch_result.get("results", []):
            sid = res.get("scenario_id")
            if sid:
                self._latest_evaluations[sid] = res.get("detailed_audit", res)

        self._last_eval_time = batch_result.get("execution_timestamp")

        # Persist all evaluations to history
        self.history_mgr.record_batch(batch_result)

        return batch_result

    def get_summary(self) -> Dict[str, Any]:
        """
        Computes aggregated dashboard metrics and KPIs.
        """
        self.check_for_updates()
        total = len(self._scenarios)
        pass_count = 0
        review_count = 0
        fail_count = 0
        not_evaluated_count = 0
        stress_scores = []
        realism_scores = []
        covered_blind_spots = set()
        categories: Dict[str, int] = {}
        methods: Dict[str, int] = {}

        for sc in self._scenarios:
            sid = sc.get("scenario_id")
            cat = sc.get("scenario_category", "unspecified")
            categories[cat] = categories.get(cat, 0) + 1

            meth = sc.get("generation_method", "unspecified")
            methods[meth] = methods.get(meth, 0) + 1

            eval_data = self._latest_evaluations.get(sid)
            if not eval_data:
                not_evaluated_count += 1
                continue

            status = eval_data.get("evaluation_status")
            if status == "PASS":
                pass_count += 1
            elif status == "REVIEW":
                review_count += 1
            elif status == "FAIL":
                fail_count += 1
            else:
                not_evaluated_count += 1

            # Decision stress
            ds = eval_data.get("decision_stress_score", {})
            s_val = ds.get("overall_stress_score") if isinstance(ds, dict) else ds
            if isinstance(s_val, (int, float)):
                stress_scores.append(s_val)

            # Realism
            rs = eval_data.get("realism_score", {})
            r_val = rs.get("overall_realism_score") if isinstance(rs, dict) else rs
            if isinstance(r_val, (int, float)):
                realism_scores.append(r_val)

            # Blind spot
            bs = eval_data.get("blind_spot_audit", {}).get("blind_spot_targeted")
            if not bs:
                bs_obj = sc.get("target_blind_spot", {})
                bs = bs_obj.get("blind_spot_id") if isinstance(bs_obj, dict) else str(bs_obj)
            if bs and bs != "NOT AVAILABLE":
                covered_blind_spots.add(bs)

        total_targets = self._get_total_blind_spot_targets()
        covered_count = len(covered_blind_spots)
        cov_pct = round((covered_count / total_targets) * 100.0, 1) if total_targets > 0 else 0.0
        cov_display = f"Blind-spot coverage: {covered_count}/{total_targets} targets ({cov_pct}%)"

        avg_stress = round(sum(stress_scores) / len(stress_scores), 2) if stress_scores else 0.0
        avg_realism = round(sum(realism_scores) / len(realism_scores), 2) if realism_scores else 0.0

        return {
            "synthetic_scenarios_only": True,
            "synthetic_badge": "[SYNTHETIC TEST SCENARIOS - RESEARCH ONLY]",
            "total_scenarios": total,
            "validated_scenarios": total,
            "rejected_scenarios": len(self._rejected),
            "rejections": self._rejected,
            "passed": pass_count,
            "review": review_count,
            "failed": fail_count,
            "not_evaluated": not_evaluated_count,
            "average_stress_score": avg_stress,
            "average_realism_score": avg_realism,
            "blind_spot_coverage": covered_count,
            "total_blind_spot_targets": total_targets,
            "blind_spot_coverage_pct": cov_pct,
            "blind_spot_coverage_display": cov_display,
            "category_distribution": categories,
            "generation_methods": methods,
            "last_evaluation_time": self._last_eval_time or "NOT AVAILABLE"
        }

    def _get_total_blind_spot_targets(self) -> int:
        """Dynamically retrieves total documented blind spots from EDA reports if available, else defaults to 24."""
        eda_report = (
            Path(__file__).resolve().parent.parent.parent
            / "eda_prompteng"
            / "reports"
            / "blind_spot_report.json"
        )
        if eda_report.exists():
            try:
                with open(eda_report, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total = data.get("summary", {}).get("total_blind_spots")
                    if total:
                        return int(total)
                    spots = data.get("blind_spots", [])
                    if spots:
                        return len(spots)
            except Exception:
                pass
        return 24

    def get_history(self, scenario_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves history records."""
        if scenario_id:
            return self.history_mgr.get_history_for_scenario(scenario_id)
        return self.history_mgr.get_all_history()

    def generate_patient(
        self,
        seed_conditions: Dict[str, Any],
        blind_spot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a new synthetic oncology patient scenario using LLM or deterministic fallback,
        immediately runs evaluation (discriminator + seed compliance + stress scoring),
        registers the scenario in memory and appends to evaluation history.
        """
        scenario, metadata = self.generator.generate_patient(seed_conditions, blind_spot_id)
        sid = scenario.get("scenario_id")

        # Run live evaluation with seed conditions
        eval_result = self.eval_adapter.evaluate_scenario(scenario, seed_conditions=seed_conditions)

        # Merge generation metadata into eval_result for recording
        eval_result["generation_metadata"] = metadata
        eval_result["generation_source"] = metadata.get("generation_source")
        eval_result["model"] = metadata.get("model")
        eval_result["seed_conditions"] = seed_conditions

        # Add to generated scenarios list if not already present
        if not any(s.get("scenario_id") == sid for s in self._generated_scenarios):
            self._generated_scenarios.append(scenario)

        self._latest_evaluations[sid] = eval_result.get("detailed_audit", eval_result)
        self._last_eval_time = eval_result.get("evaluation_timestamp")

        # Record to history
        history_rec = self.history_mgr.record_evaluation(eval_result)

        # Detailed view for dashboard
        detail_view = ScenarioAdapter.to_detailed_view(scenario, self._latest_evaluations[sid])

        return {
            "scenario": scenario,
            "metadata": metadata,
            "evaluation": eval_result,
            "detail_view": detail_view,
            "history_record": history_rec
        }

    def get_analytics(self) -> Dict[str, Any]:
        """
        Computes comprehensive generation and realism analytics from evaluation history.
        Tracks LLM vs Fallback distributions, pass/review/fail rates, realism metrics,
        and blind spot coverage without any hardcoded falsifications.
        """
        all_records = self.history_mgr.get_all_history()
        total_evals = len(all_records)

        llm_count = 0
        fallback_count = 0
        template_count = 0
        pass_count = 0
        review_count = 0
        fail_count = 0

        realism_scores = []
        stress_scores = []
        stat_similarities = []
        cohort_similarities = []
        covered_blind_spots = set()

        for rec in all_records:
            source = str(rec.get("generation_source", "")).upper()
            if "LLM" in source:
                llm_count += 1
            elif "DETERMINISTIC" in source or "FALLBACK" in source:
                fallback_count += 1
            else:
                template_count += 1

            status = str(rec.get("evaluation_status", "")).upper()
            if status == "PASS":
                pass_count += 1
            elif status == "REVIEW":
                review_count += 1
            elif status == "FAIL":
                fail_count += 1

            rs = rec.get("realism_score")
            if isinstance(rs, (int, float)):
                realism_scores.append(rs)

            ss = rec.get("decision_stress_score")
            if isinstance(ss, (int, float)):
                stress_scores.append(ss)

            sq = rec.get("synthetic_quality", {})
            if isinstance(sq, dict):
                sim = sq.get("statistical_similarity")
                if isinstance(sim, (int, float)):
                    stat_similarities.append(sim)
                csim = sq.get("cohort_similarity")
                if isinstance(csim, (int, float)):
                    cohort_similarities.append(csim)

            bs = rec.get("blind_spot_targeted")
            if bs and bs != "NOT AVAILABLE":
                covered_blind_spots.add(bs)

        total_targets = self._get_total_blind_spot_targets()
        cov_count = len(covered_blind_spots)
        cov_pct = round((cov_count / total_targets) * 100.0, 1) if total_targets > 0 else 0.0

        avg_realism = round(sum(realism_scores) / len(realism_scores), 2) if realism_scores else 0.0
        avg_stress = round(sum(stress_scores) / len(stress_scores), 2) if stress_scores else 0.0
        avg_stat_sim = round(sum(stat_similarities) / len(stat_similarities), 3) if stat_similarities else 0.0
        avg_cohort_sim = round(sum(cohort_similarities) / len(cohort_similarities), 3) if cohort_similarities else 0.0

        recent_gen = []
        for rec in reversed(all_records):
            sid = rec.get("scenario_id", "")
            if sid.startswith("SYN-") or rec.get("generation_source") in ("LLM", "DETERMINISTIC_FALLBACK"):
                recent_gen.append({
                    "scenario_id": sid,
                    "generation_source": rec.get("generation_source", "UNKNOWN"),
                    "model": rec.get("model", "UNKNOWN"),
                    "evaluation_status": rec.get("evaluation_status", "NOT EVALUATED"),
                    "realism_score": rec.get("realism_score"),
                    "decision_stress_score": rec.get("decision_stress_score"),
                    "blind_spot_targeted": rec.get("blind_spot_targeted"),
                    "timestamp": rec.get("evaluation_timestamp")
                })
                if len(recent_gen) >= 10:
                    break

        return {
            "total_evaluations": total_evals,
            "total_interactive_generated": llm_count + fallback_count,
            "source_distribution": {
                "LLM": llm_count,
                "DETERMINISTIC_FALLBACK": fallback_count,
                "TEMPLATE_BASELINE": template_count
            },
            "status_distribution": {
                "PASS": pass_count,
                "REVIEW": review_count,
                "FAIL": fail_count
            },
            "pass_rate_pct": round((pass_count / total_evals) * 100.0, 1) if total_evals > 0 else 0.0,
            "review_rate_pct": round((review_count / total_evals) * 100.0, 1) if total_evals > 0 else 0.0,
            "fail_rate_pct": round((fail_count / total_evals) * 100.0, 1) if total_evals > 0 else 0.0,
            "average_realism_score": avg_realism,
            "average_stress_score": avg_stress,
            "average_statistical_similarity": avg_stat_sim,
            "average_cohort_similarity": avg_cohort_sim,
            "blind_spot_coverage": {
                "covered_count": cov_count,
                "total_targets": total_targets,
                "coverage_pct": cov_pct,
                "unique_blind_spots": sorted(list(covered_blind_spots))
            },
            "recent_generations": recent_gen
        }
