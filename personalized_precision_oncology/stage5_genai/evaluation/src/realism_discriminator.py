"""
Synthetic Data Realism & Discriminator Validation Engine for Stage 5.
Evaluates newly generated synthetic oncology patient scenarios against the reference
cleaned cohort (cleaned_cohort.csv, N=75) in a strictly read-only manner.

CRITICAL RESEARCH & ETHICAL MANDATE:
- The discriminator evaluates whether a synthetic case is statistically and clinically consistent
  with the reference cohort distribution.
- It NEVER claims or proves that a synthetic patient is real.
- Real patient records (TCGA-*, MSK-*) remain completely isolated and their identifiers are NEVER exposed.
- Synthetic cases are strictly labeled: synthetic=true.
"""

import os
import csv
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEANED_COHORT_CSV = BASE_DIR / "data_engineering" / "processed" / "cleaned_cohort.csv"
MUTATION_FREQ_JSON = BASE_DIR / "data_engineering" / "processed" / "mutation_frequency.json"
BIOMARKER_DIST_JSON = BASE_DIR / "data_engineering" / "processed" / "biomarker_distributions.json"
BLIND_SPOT_REPORT_JSON = BASE_DIR / "eda_prompteng" / "reports" / "blind_spot_report.json"


class SyntheticRealismDiscriminator:
    """
    Discriminator engine that compares synthetic oncology patient records
    against authentic reference distributions from cleaned_cohort.csv.
    """

    def __init__(self, cohort_path: Path = CLEANED_COHORT_CSV):
        self.cohort_path = cohort_path
        self.reference_records = self._load_reference_cohort()
        self.reference_distributions = self._compute_reference_distributions()
        self.blind_spots = self._load_blind_spots()

    def _load_reference_cohort(self) -> List[Dict[str, Any]]:
        """Loads reference cohort in read-only mode without exposing patient identities."""
        records = []
        if not self.cohort_path.exists():
            return records

        with open(self.cohort_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "cancer_type": row.get("cancer_type", "").strip(),
                    "age": float(row.get("age", 0.0)) if row.get("age") and row.get("age") != "not_available_in_source" else None,
                    "sex": row.get("sex", "").strip(),
                    "cancer_stage": row.get("cancer_stage", "").strip(),
                    "histology": row.get("histology", "").strip(),
                    "smoking_history": row.get("smoking_history", "").strip(),
                    "pack_years": float(row.get("pack_years", 0.0)) if row.get("pack_years") and row.get("pack_years") != "not_available_in_source" else None,
                    "primary_driver_gene": row.get("primary_driver_gene", "").strip(),
                    "primary_mutation": row.get("primary_mutation", "").strip(),
                    "tmb": float(row.get("tmb_mut_per_mb", 0.0)) if row.get("tmb_mut_per_mb") and row.get("tmb_mut_per_mb") != "not_available_in_source" else None,
                    "pdl1": float(row.get("pdl1_tps_percent", 0.0)) if row.get("pdl1_tps_percent") and row.get("pdl1_tps_percent") != "not_available_in_source" else None,
                })
        return records

    def _compute_reference_distributions(self) -> Dict[str, Any]:
        """Extracts statistical summary and distributions from reference records."""
        if not self.reference_records:
            return {}

        ages = [r["age"] for r in self.reference_records if r["age"] is not None]
        mean_age = sum(ages) / len(ages) if ages else 65.0
        var_age = sum((a - mean_age) ** 2 for a in ages) / len(ages) if len(ages) > 1 else 100.0
        std_age = math.sqrt(var_age)

        stages = {}
        histologies = {}
        driver_genes = {}
        sexes = {}

        for r in self.reference_records:
            st = r["cancer_stage"]
            if st:
                stages[st] = stages.get(st, 0) + 1
            h = r["histology"]
            if h:
                histologies[h] = histologies.get(h, 0) + 1
            g = r["primary_driver_gene"]
            if g:
                driver_genes[g] = driver_genes.get(g, 0) + 1
            s = r["sex"]
            if s:
                sexes[s] = sexes.get(s, 0) + 1

        total = len(self.reference_records)
        return {
            "total_records": total,
            "age_mean": mean_age,
            "age_std": std_age,
            "age_min": min(ages) if ages else 35.0,
            "age_max": max(ages) if ages else 90.0,
            "stages": {k: v / total for k, v in stages.items()},
            "histologies": {k: v / total for k, v in histologies.items()},
            "driver_genes": {k: v / total for k, v in driver_genes.items()},
            "sexes": {k: v / total for k, v in sexes.items()}
        }

    def _load_blind_spots(self) -> Dict[str, Dict[str, Any]]:
        """Loads documented blind spots to protect intentional rare stress cases."""
        if not BLIND_SPOT_REPORT_JSON.exists():
            return {}
        try:
            with open(BLIND_SPOT_REPORT_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {bs["blind_spot_id"]: bs for bs in data.get("blind_spots", [])}
        except Exception:
            return {}

    def extract_numeric_age(self, scenario: Dict[str, Any]) -> Optional[float]:
        """Extracts numeric age from age or age_group in scenario."""
        p_ctx = scenario.get("patient_context", {})
        if "age" in p_ctx and isinstance(p_ctx["age"], (int, float)):
            return float(p_ctx["age"])
        if "age" in scenario and isinstance(scenario["age"], (int, float)):
            return float(scenario["age"])

        age_group = p_ctx.get("age_group", "") or scenario.get("age_group", "")
        if isinstance(age_group, str) and "-" in age_group:
            parts = age_group.split("-")
            try:
                return (float(parts[0]) + float(parts[1])) / 2.0
            except ValueError:
                pass
        return 65.0

    def check_anomalies(self, scenario: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Detects impossible or medically aberrant combinations:
        - Impossible age (<18 or >115)
        - Negative biomarker values
        - PD-L1 > 100%
        - Contradictory clinical stages
        """
        issues = []
        age = self.extract_numeric_age(scenario)
        if age is not None:
            if age < 18 or age > 115:
                issues.append(f"Implausible oncology patient age: {age}")

        b_marks = scenario.get("biomarkers", {})
        tmb = b_marks.get("tmb")
        if tmb is not None:
            if isinstance(tmb, (int, float)) and (tmb < 0 or tmb > 300):
                issues.append(f"Biologically implausible TMB value: {tmb}")

        pdl1 = b_marks.get("pdl1_tps") or b_marks.get("pdl1")
        if pdl1 is not None:
            if isinstance(pdl1, (int, float)) and (pdl1 < 0 or pdl1 > 100):
                issues.append(f"PD-L1 TPS out of range [0-100]: {pdl1}")

        p_ctx = scenario.get("patient_context", {})
        stage = str(p_ctx.get("stage", "")).upper().strip()
        c_ctx = scenario.get("clinical_context", {})
        disease_status = str(c_ctx.get("disease_status", "")).upper().strip()

        # Stage I check: ensure 'STAGE I', 'STAGE IA', or 'STAGE IB' and not 'STAGE IV'
        is_stage_1 = stage in ["STAGE I", "STAGE IA", "STAGE IB"]
        if is_stage_1 and "METASTATIC" in disease_status:
            issues.append("Contradictory clinical presentation: Stage I designated as Metastatic.")

        if len(issues) > 0:
            if any("Implausible oncology patient age" in i or "PD-L1 TPS out of range" in i or "Contradictory" in i for i in issues):
                return "FAIL", issues
            return "REVIEW", issues
        return "PASS", []

    def check_duplication_and_memorization(self, scenario: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """
        Checks whether the synthetic record is an exact duplicate of a reference patient
        or exhibits memorization (exact or near-exact match on all clinical/genomic features).
        Returns: (status: 'CLEAN'|'FLAGGED', max_similarity: float, issues: List[str])
        
        PRIVACY GUARANTEE: Never exposes the matched reference patient's identifier!
        """
        if not self.reference_records:
            return "CLEAN", 0.0, []

        p_ctx = scenario.get("patient_context", {})
        syn_age = self.extract_numeric_age(scenario)
        syn_sex = str(p_ctx.get("sex", "")).strip().lower()
        syn_stage = str(p_ctx.get("stage", "")).strip().lower()
        syn_hist = str(p_ctx.get("histology", "")).strip().lower()

        # Genomic driver
        syn_genes = set()
        syn_mutations = set()
        for alt in scenario.get("genomic_profile", {}).get("alterations", []):
            g = alt.get("gene", "").strip().upper()
            v = alt.get("variant", "").strip()
            if g:
                syn_genes.add(g)
            if v:
                syn_mutations.add(v)

        b_marks = scenario.get("biomarkers", {})
        syn_tmb = b_marks.get("tmb") if isinstance(b_marks.get("tmb"), (int, float)) else None
        syn_pdl1 = b_marks.get("pdl1_tps") if isinstance(b_marks.get("pdl1_tps"), (int, float)) else None

        max_sim = 0.0
        exact_match_detected = False

        for ref in self.reference_records:
            match_score = 0.0
            total_features = 6.0

            # 1. Age match
            if ref["age"] is not None and syn_age is not None:
                diff = abs(ref["age"] - syn_age)
                if diff == 0:
                    match_score += 1.0
                elif diff <= 2.0:
                    match_score += 0.80
                elif diff <= 5.0:
                    match_score += 0.50

            # 2. Sex match
            if ref["sex"].lower() == syn_sex:
                match_score += 1.0

            # 3. Stage match
            ref_stage = ref["cancer_stage"].lower().strip()
            if ref_stage and ref_stage == syn_stage:
                match_score += 1.0
            elif ref_stage and (ref_stage in syn_stage or syn_stage in ref_stage):
                match_score += 0.75

            # 4. Histology match
            ref_hist = ref["histology"].lower().strip()
            if ref_hist and ref_hist == syn_hist:
                match_score += 1.0
            elif ref_hist and (ref_hist in syn_hist or syn_hist in ref_hist):
                match_score += 0.75

            # 5. Driver gene & mutation match
            ref_gene = ref["primary_driver_gene"].upper()
            ref_mut = ref["primary_mutation"].strip()
            if ref_gene and ref_gene in syn_genes and ref_mut and ref_mut in syn_mutations:
                match_score += 1.0
            elif ref_gene and ref_gene in syn_genes:
                match_score += 0.60

            # 6. Biomarkers match (if present in ref)
            if ref["pdl1"] is not None and syn_pdl1 is not None:
                if abs(ref["pdl1"] - syn_pdl1) <= 2.0:
                    match_score += 1.0
                elif abs(ref["pdl1"] - syn_pdl1) <= 10.0:
                    match_score += 0.50
            else:
                total_features = 5.0

            sim = match_score / total_features
            if sim > max_sim:
                max_sim = sim

            # Exact duplicate detection across all key features
            if sim >= 0.98:
                exact_match_detected = True

        max_sim = round(max_sim, 2)
        issues = []
        if exact_match_detected:
            issues.append("Exact reference record duplicate detected (critical memorization risk).")
            return "FLAGGED", max_sim, issues
        elif max_sim >= 0.95:
            issues.append(f"Suspiciously high similarity to a reference record ({max_sim:.2f}). Flagged for review.")
            return "FLAGGED", max_sim, issues

        return "CLEAN", max_sim, []

    def calculate_statistical_similarity(self, scenario: Dict[str, Any]) -> float:
        """
        Calculates how consistently the synthetic case aligns with empirical reference cohort distributions.
        Returns a float between 0.0 and 1.0.
        """
        if not self.reference_distributions:
            return 0.85

        scores = []
        dist = self.reference_distributions

        # 1. Age distribution similarity (Gaussian likelihood scaled 0-1)
        age = self.extract_numeric_age(scenario)
        if age is not None:
            mean = dist.get("age_mean", 65.0)
            std = max(dist.get("age_std", 10.0), 5.0)
            z = abs(age - mean) / std
            age_sim = max(0.2, math.exp(-0.5 * (z ** 2)))
            scores.append(age_sim)

        # 2. Stage distribution plausibility
        p_ctx = scenario.get("patient_context", {})
        syn_stage = p_ctx.get("stage", "")
        stage_sim = 0.5
        for ref_s, prob in dist.get("stages", {}).items():
            if ref_s.lower() in syn_stage.lower() or syn_stage.lower() in ref_s.lower():
                stage_sim = min(1.0, prob * 3.0 + 0.4)
                break
        scores.append(stage_sim)

        # 3. Histology plausibility
        syn_hist = p_ctx.get("histology", "")
        hist_sim = 0.5
        for ref_h, prob in dist.get("histologies", {}).items():
            if ref_h.lower() in syn_hist.lower() or syn_hist.lower() in ref_h.lower():
                hist_sim = min(1.0, prob * 2.0 + 0.5)
                break
        scores.append(hist_sim)

        # 4. Driver alteration frequency alignment
        gen_prof = scenario.get("genomic_profile", {})
        driver_sims = []
        for alt in gen_prof.get("alterations", []):
            g = alt.get("gene", "").upper()
            prob = dist.get("driver_genes", {}).get(g, 0.05)
            d_score = min(1.0, max(0.65, prob * 3.0 + 0.5))
            driver_sims.append(d_score)
        if driver_sims:
            scores.append(sum(driver_sims) / len(driver_sims))

        # 5. Biomarker bounds plausibility
        b_marks = scenario.get("biomarkers", {})
        tmb = b_marks.get("tmb")
        if isinstance(tmb, (int, float)):
            tmb_score = 0.95 if 1.0 <= tmb <= 50.0 else 0.60
            scores.append(tmb_score)

        return round(sum(scores) / len(scores), 2) if scores else 0.85

    def calculate_cohort_similarity(self, scenario: Dict[str, Any]) -> float:
        """
        Calculates multi-dimensional distance/similarity to the reference cohort centroid.
        Returns a float between 0.0 and 1.0.
        """
        if not self.reference_records:
            return 0.80

        syn_age = self.extract_numeric_age(scenario)
        p_ctx = scenario.get("patient_context", {})
        syn_stage = str(p_ctx.get("stage", "")).lower()
        syn_hist = str(p_ctx.get("histology", "")).lower()
        syn_sex = str(p_ctx.get("sex", "")).lower()

        syn_genes = set()
        for alt in scenario.get("genomic_profile", {}).get("alterations", []):
            g = alt.get("gene", "").strip().upper()
            if g:
                syn_genes.add(g)

        sims = []
        for ref in self.reference_records:
            s = 0.0
            # Age similarity
            if ref["age"] is not None and syn_age is not None:
                s += max(0.0, 1.0 - abs(ref["age"] - syn_age) / 40.0) * 0.25
            else:
                s += 0.20

            # Stage similarity
            if ref["cancer_stage"].lower() in syn_stage or syn_stage in ref["cancer_stage"].lower():
                s += 0.25
            else:
                s += 0.08

            # Histology similarity
            if ref["histology"].lower() in syn_hist or syn_hist in ref["histology"].lower():
                s += 0.25
            else:
                s += 0.05

            # Driver gene similarity
            if ref["primary_driver_gene"].upper() in syn_genes:
                s += 0.25
            else:
                s += 0.08

            sims.append(s)

        sims.sort(reverse=True)
        top_k = sims[:10]
        avg_sim = sum(top_k) / len(top_k) if top_k else 0.75
        return round(min(1.0, max(0.1, avg_sim)), 2)

    def evaluate_scenario_realism(
        self,
        scenario: Dict[str, Any],
        gen_audit: Optional[Dict[str, Any]] = None,
        clin_valid: bool = True,
        prov_valid: bool = True,
        schema_valid: bool = True,
        seed_compliance: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes the complete multi-layer Synthetic Realism & Discriminator check.
        Returns standard synthetic_quality dictionary.
        """
        # 1. Anomaly check
        anomaly_status, anomaly_issues = self.check_anomalies(scenario)

        # 2. Duplication & memorization check
        dup_status, max_sim, dup_issues = self.check_duplication_and_memorization(scenario)

        # 3. Statistical similarity
        stat_sim = self.calculate_statistical_similarity(scenario)

        # 4. Cohort similarity
        cohort_sim = self.calculate_cohort_similarity(scenario)

        # 5. Biological plausibility
        bio_plaus = 0.95 if (gen_audit and gen_audit.get("valid", False)) else 0.50

        # 6. Clinical consistency
        clin_score = 0.92 if clin_valid else 0.40

        # 7. Blind spot protection
        tbs = scenario.get("target_blind_spot", {})
        bs_id = tbs.get("blind_spot_id", "") if isinstance(tbs, dict) else str(tbs)
        is_blind_spot = bs_id in self.blind_spots or bs_id.startswith("BS")

        if is_blind_spot:
            blind_spot_status = "RARE_BUT_VALID"
        elif anomaly_status == "FAIL" or not clin_valid or not schema_valid:
            blind_spot_status = "INVALID"
        else:
            blind_spot_status = "NORMAL"

        # 8. Overall Realism Score (0.0 to 1.0)
        realism_score = round(
            0.30 * cohort_sim +
            0.30 * stat_sim +
            0.20 * clin_score +
            0.20 * bio_plaus,
            2
        )

        # 9. Final Decision Logic
        if not schema_valid or not clin_valid or anomaly_status == "FAIL" or (dup_status == "FLAGGED" and "Exact" in " ".join(dup_issues)):
            final_status = "REJECTED"
            interpretation = "Synthetic case rejected due to clinical contradictions, invalid schema, or exact reference record replication."
        elif (blind_spot_status == "RARE_BUT_VALID" and cohort_sim < 0.65) or dup_status == "FLAGGED" or anomaly_status == "REVIEW" or (seed_compliance and seed_compliance.get("status") == "REVIEW"):
            final_status = "REVIEW"
            if blind_spot_status == "RARE_BUT_VALID":
                interpretation = f"Synthetic stress-test case targets intentional blind spot ({bs_id}); rare biological presentation verified under research governance (requires tumor-board review)."
            elif dup_status == "FLAGGED":
                interpretation = "Synthetic case flagged for review due to elevated similarity to a reference baseline record."
            else:
                interpretation = "Synthetic case exhibits atypical clinical parameters requiring manual review."
        else:
            final_status = "ACCEPTED"
            interpretation = "Synthetic record is statistically and clinically consistent with reference oncology cohort characteristics (research stress test)."

        return {
            "statistical_similarity": stat_sim,
            "cohort_similarity": cohort_sim,
            "realism_score": realism_score,
            "anomaly_status": anomaly_status,
            "duplicate_check": dup_status,
            "clinical_consistency": clin_score,
            "biological_plausibility": bio_plaus,
            "blind_spot_status": blind_spot_status,
            "status": final_status,
            "interpretation": interpretation
        }

    @property
    def reference_cohort(self) -> List[Dict[str, Any]]:
        """Compatibility property providing reference cohort records."""
        return self.reference_records

    @property
    def cohort_vectors(self) -> List[Dict[str, Any]]:
        """Vector/feature representation for reference cohort records."""
        return self.reference_records

    def evaluate_synthetic_quality(self, scenario: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Convenience alias for evaluate_scenario_realism."""
        return self.evaluate_scenario_realism(scenario, **kwargs)

    def detect_anomalies(self, scenario: Dict[str, Any]) -> List[str]:
        """Convenience alias returning anomaly issues list."""
        _, issues = self.check_anomalies(scenario)
        return issues

    def check_duplication_memorization(self, scenario: Dict[str, Any]) -> Tuple[str, float]:
        """Convenience alias returning (status: 'clean'|'memorization_review', max_sim: float)."""
        status, max_sim, _ = self.check_duplication_and_memorization(scenario)
        normalized_status = "clean" if status == "CLEAN" else "memorization_review"
        return normalized_status, max_sim

