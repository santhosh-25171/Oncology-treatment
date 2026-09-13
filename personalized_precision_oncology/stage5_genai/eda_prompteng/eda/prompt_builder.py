"""
Interactive Prompt Builder for Stage 5 Synthetic Oncology Patient Generation.
Combines user-specified seed conditions, relevant blind-spot definitions,
verified evidence constraints, 3-tier evidence governance, and synthetic data instructions.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
REPORTS_DIR = BASE_DIR / "reports"
BLIND_SPOT_REPORT = REPORTS_DIR / "blind_spot_report.json"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"


class PromptBuilder:
    """Builds clinical and genomic prompt payloads for LLM synthetic patient generation."""

    def __init__(self, prompts_dir: Path = PROMPTS_DIR, blind_spots_path: Path = BLIND_SPOT_REPORT):
        self.prompts_dir = prompts_dir
        self.blind_spots_path = blind_spots_path
        self.system_prompt = self._load_system_prompt()
        self.blind_spots = self._load_blind_spots()

    def _load_system_prompt(self) -> str:
        if SYSTEM_PROMPT_FILE.exists():
            with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "You are an expert Precision Oncology AI generating synthetic patient scenarios for computational stress-testing."

    def _load_blind_spots(self) -> Dict[str, Dict[str, Any]]:
        if self.blind_spots_path.exists():
            try:
                with open(self.blind_spots_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {bs["blind_spot_id"]: bs for bs in data.get("blind_spots", [])}
            except Exception:
                pass
        return {}

    def build_prompt(
        self,
        seed_conditions: Dict[str, Any],
        scenario_id: str,
        blind_spot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Constructs system and user prompt strings for LLM generation.
        Returns:
            {
                "system_prompt": str,
                "user_prompt": str,
                "target_blind_spot": dict,
                "schema_instructions": str
            }
        """
        # Determine blind spot
        bs_info = {}
        if blind_spot_id and blind_spot_id in self.blind_spots:
            bs_info = self.blind_spots[blind_spot_id]
        elif blind_spot_id:
            bs_info = {
                "blind_spot_id": blind_spot_id,
                "category": "user_specified_blind_spot",
                "coverage_status": "rare",
                "reason": f"Custom targeted blind spot: {blind_spot_id}"
            }
        else:
            # Select default based on driver or category
            driver = str(seed_conditions.get("driver_alteration", "")).upper()
            if "BRAF" in driver:
                bs_info = self.blind_spots.get("BS001", {"blind_spot_id": "BS001", "category": "rare_mutation", "reason": "BRAF p.V600E rare presentation"})
            elif "MET" in driver:
                bs_info = self.blind_spots.get("BS002", {"blind_spot_id": "BS002", "category": "rare_mutation", "reason": "MET exon 14 splice/skipping"})
            elif "RET" in driver:
                bs_info = self.blind_spots.get("BS005", {"blind_spot_id": "BS005", "category": "rare_mutation", "reason": "RET fusions in NSCLC"})
            elif "ERBB2" in driver or "HER2" in driver:
                bs_info = self.blind_spots.get("BS004", {"blind_spot_id": "BS004", "category": "rare_mutation", "reason": "ERBB2 insertion/amplification"})
            else:
                bs_info = {"blind_spot_id": "BS024", "category": "synthetic_stress", "reason": "General oncology decision-logic stress testing"}

        # Format user prompt
        age = seed_conditions.get("age", 65)
        sex = seed_conditions.get("sex", "Female")
        cancer_type = seed_conditions.get("cancer_type", "NSCLC")
        stage = seed_conditions.get("stage", "Stage IV")
        histology = seed_conditions.get("histology", "Lung Adenocarcinoma")
        smoking = seed_conditions.get("smoking_status", "Never Smoker")
        tmb = seed_conditions.get("tmb", 7.5)
        pdl1 = seed_conditions.get("pd_l1", seed_conditions.get("pdl1", 25))
        driver_gene = seed_conditions.get("driver_alteration", "EGFR")

        user_prompt = f"""GENERATE A NEW SYNTHETIC ONCOLOGY PATIENT SCENARIO

MANDATORY SCIENTIFIC & SAFETY CONSTRAINTS:
1. This is a SYNTHETIC oncology scenario created strictly for computational stress-testing.
2. The scenario_id MUST be exactly: "{scenario_id}".
3. The field "synthetic" MUST be boolean: true.
4. DO NOT copy, reproduce, or claim any relationship to real living patients or historical trial participants.
5. All clinical and molecular parameters must adhere to the user-selected seed conditions below.

USER-SELECTED SEED CONDITIONS:
- Primary Cancer Type: {cancer_type}
- Clinical Stage: {stage}
- Histology: {histology}
- Patient Age: {age}
- Biological Sex: {sex}
- Smoking History: {smoking}
- Tumor Mutation Burden (TMB): {tmb} mut/Mb
- PD-L1 TPS: {pdl1}%
- Targeted Driver Alteration: {driver_gene}
- Targeted Blind Spot: {bs_info.get('blind_spot_id')} ({bs_info.get('reason', '')})

REQUIRED JSON STRUCTURE:
You must output a single valid JSON object strictly matching this schema:
{{
  "scenario_id": "{scenario_id}",
  "synthetic": true,
  "generation_method": "llm",
  "scenario_category": "{bs_info.get('category', 'synthetic_stress')}",
  "target_blind_spot": {{
    "blind_spot_id": "{bs_info.get('blind_spot_id')}",
    "category": "{bs_info.get('category')}",
    "coverage_status": "{bs_info.get('coverage_status', 'rare')}",
    "reason": "{bs_info.get('reason')}"
  }},
  "patient_context": {{
    "age": {age},
    "age_group": "{int(age//5*5)}-{int(age//5*5+4)}",
    "sex": "{sex}",
    "cancer_type": "{cancer_type}",
    "histology": "{histology}",
    "stage": "{stage}",
    "prior_treatment_context": "Initial diagnosis or prior systemic therapy context"
  }},
  "genomic_profile": {{
    "alterations": [
      {{
        "gene": "{driver_gene}",
        "variant": "e.g. specific missense or fusion",
        "alteration_type": "Kinase Activating / Amplification / Fusion",
        "status": "biologically_observed"
      }}
    ],
    "cooccurring_alterations": [
      {{
        "gene": "TP53",
        "variant": "p.R273H",
        "status": "historically_observed"
      }}
    ],
    "resistance_related_features": []
  }},
  "biomarkers": {{
    "tmb": {tmb},
    "tmb_status": "{'high' if float(tmb) >= 10 else 'low'}",
    "pdl1_tps": {pdl1},
    "msi_status": "MSS"
  }},
  "clinical_context": {{
    "disease_status": "{'Metastatic' if 'IV' in str(stage) else 'Locally Advanced'}",
    "progression_context": "Detailed clinical narrative of disease progression or presentation"
  }},
  "synthetic_assumptions": [
    "Explicit modeled assumption for algorithmic stress-testing"
  ],
  "reference_evidence": [
    "Peer-reviewed clinical evidence or guideline benchmark (e.g. NCCN/FDA)"
  ],
  "uncertainty": {{
    "level": "moderate",
    "reason": "Explicit justification for clinical/evidence uncertainty"
  }},
  "provenance": {{
    "reference_sources": ["TCGA PanCancer Atlas", "MSK-IMPACT", "CIViC / ClinVar Curated Registries"],
    "blind_spot_source": "Stage 5 EDA / blind_spot_report.json",
    "generation_method": "llm",
    "generation_timestamp": "ISO8601 string",
    "random_seed": 42
  }}
}}

Output ONLY the raw JSON object. Do NOT wrap in markdown quotes if possible or output any explanatory text outside the JSON.
"""

        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "target_blind_spot": bs_info
        }
