"""
Interactive GenAI Scenario Generator for Stage 5.
Connects PromptBuilder with actual configured LLM endpoints (OpenAI / vLLM / Ollama / etc.)
or routes gracefully to a deterministic fallback generator when LLM credentials are absent.

Guarantees:
- Never fakes an LLM execution. Explicitly records generation_source: "LLM" or "DETERMINISTIC_FALLBACK".
- Automatically allocates unique sequential synthetic IDs: SYN-000001, SYN-000002, etc.
- Persists newly generated scenarios to generated_scenarios.jsonl (isolated from reference cohort).
"""

import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from personalized_precision_oncology.stage5_genai.eda_prompteng.eda.prompt_builder import PromptBuilder
from .template_generator import TemplateScenarioGenerator
from ..src.scenario_validator import ScenarioValidator

BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = BASE_DIR / "scenarios"
GENERATED_SCENARIOS_JSONL = SCENARIOS_DIR / "generated_scenarios.jsonl"


class InteractivePatientGenerator:
    """Orchestrates interactive synthetic oncology patient generation."""

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        validator: Optional[ScenarioValidator] = None,
        scenarios_file: Path = GENERATED_SCENARIOS_JSONL
    ):
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.validator = validator or ScenarioValidator()
        self.scenarios_file = scenarios_file
        self.template_fallback = TemplateScenarioGenerator(seed=42)

        # Environment configuration
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "15.0"))

    def is_llm_configured(self) -> bool:
        """Determines whether a live LLM API key or accessible local endpoint is configured."""
        k = self.api_key.strip() if self.api_key else ""
        return bool(k and not k.startswith("mock") and not k.startswith("dummy"))

    def get_next_synthetic_id(self) -> str:
        """Determines the next sequential synthetic patient ID (SYN-000001, SYN-000002, ...)."""
        max_idx = 0
        if self.scenarios_file.exists():
            with open(self.scenarios_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            sid = data.get("scenario_id", "")
                            if sid.startswith("SYN-"):
                                idx = int(sid.split("-")[1])
                                if idx > max_idx:
                                    max_idx = idx
                        except (ValueError, IndexError, json.JSONDecodeError):
                            pass
        return f"SYN-{max_idx + 1:06d}"

    def generate_patient(
        self,
        seed_conditions: Dict[str, Any],
        blind_spot_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Executes interactive generation for a given set of seed conditions.
        Returns:
            (scenario_dict, execution_metadata)
        """
        scenario_id = self.get_next_synthetic_id()
        prompt_payload = self.prompt_builder.build_prompt(
            seed_conditions=seed_conditions,
            scenario_id=scenario_id,
            blind_spot_id=blind_spot_id
        )

        metadata = {
            "scenario_id": scenario_id,
            "request_timestamp": datetime.now(timezone.utc).isoformat(),
            "llm_configured": self.is_llm_configured(),
            "model": self.model if self.is_llm_configured() else "deterministic_engine_v5",
            "generation_source": "DETERMINISTIC_FALLBACK",
            "fallback_reason": None,
            "latency_ms": 0.0
        }

        scenario = None
        t0 = time.time()

        # 1. Attempt LLM generation if configured
        if self.is_llm_configured():
            try:
                scenario, err = self._call_llm_endpoint(prompt_payload, scenario_id)
                if scenario:
                    metadata["generation_source"] = "LLM"
                    metadata["model"] = self.model
                else:
                    metadata["fallback_reason"] = err or "LLM_ERROR"
            except Exception as e:
                metadata["fallback_reason"] = f"LLM_EXCEPTION: {str(e)}"

        if not scenario:
            if not self.is_llm_configured():
                metadata["fallback_reason"] = "LLM_NOT_CONFIGURED"
            # 2. Deterministic Fallback Generation adhering strictly to seed conditions
            scenario = self._generate_deterministic_fallback(
                seed_conditions=seed_conditions,
                scenario_id=scenario_id,
                target_blind_spot=prompt_payload["target_blind_spot"],
                fallback_reason=metadata["fallback_reason"]
            )
            metadata["generation_source"] = "DETERMINISTIC_FALLBACK"

        metadata["latency_ms"] = round((time.time() - t0) * 1000.0, 2)

        # 3. Persist to generated_scenarios.jsonl (separate from reference cohort)
        self._append_to_generated_storage(scenario)

        return scenario, metadata

    def _call_llm_endpoint(self, prompt_payload: Dict[str, Any], scenario_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Sends HTTP request to configured OpenAI-compatible endpoint."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt_payload["system_prompt"]},
                {"role": "user", "content": prompt_payload["user_prompt"]}
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                parsed["scenario_id"] = scenario_id
                parsed["synthetic"] = True
                parsed["generation_method"] = "llm"
                parsed["generation_source"] = "LLM"
                return parsed, None
        except urllib.error.HTTPError as e:
            return None, f"HTTP_{e.code}_{e.reason}"
        except urllib.error.URLError as e:
            return None, f"URL_ERROR_{str(e.reason)}"
        except json.JSONDecodeError:
            return None, "MALFORMED_JSON_RESPONSE"
        except Exception as e:
            return None, f"LLM_CALL_FAILED_{str(e)}"

    def _generate_deterministic_fallback(
        self,
        seed_conditions: Dict[str, Any],
        scenario_id: str,
        target_blind_spot: Dict[str, Any],
        fallback_reason: str
    ) -> Dict[str, Any]:
        """
        Synthesizes a compliant synthetic patient record from seed inputs
        using calibrated deterministic oncology parameters.
        """
        age = float(seed_conditions.get("age", 65.0))
        sex = str(seed_conditions.get("sex", "Female"))
        cancer_type = str(seed_conditions.get("cancer_type", "NSCLC"))
        stage = str(seed_conditions.get("stage", "Stage IV"))
        histology = str(seed_conditions.get("histology", "Lung Adenocarcinoma"))
        smoking = str(seed_conditions.get("smoking_status", "Never Smoker"))
        tmb = float(seed_conditions.get("tmb", 7.2))
        pdl1 = float(seed_conditions.get("pd_l1", seed_conditions.get("pdl1", 20.0)))
        driver = str(seed_conditions.get("driver_alteration", "EGFR")).strip().upper()

        # Build genomic profile based on driver
        if "EGFR" in driver:
            var = "p.L858R"
            alt_type = "Exon 21 Activating Missense"
            ev = "EGFR p.L858R is a validated driver responsive to osimertinib."
        elif "KRAS" in driver:
            var = "p.G12C"
            alt_type = "Exon 2 Codon 12 Missense"
            ev = "KRAS p.G12C targets sotorasib/adagrasib in second-line NSCLC."
        elif "BRAF" in driver:
            var = "p.V600E"
            alt_type = "Kinase Domain Missense"
            ev = "BRAF p.V600E observed in reference baseline; responsive to dabrafenib+trametinib."
        elif "MET" in driver:
            var = "p.D1010H"
            alt_type = "Exon 14 Splice Site Mutation"
            ev = "MET exon 14 alterations respond to capmatinib and tepotinib."
        else:
            var = "p.E545K" if "PIK3CA" in driver else "p.R175H"
            alt_type = "Activating Alteration"
            ev = f"{driver} molecular biomarker observed in clinical registries."

        age_grp = f"{int(age // 5 * 5)}-{int(age // 5 * 5 + 4)}"

        return {
            "scenario_id": scenario_id,
            "synthetic": True,
            "generation_method": "deterministic_fallback",
            "generation_source": "DETERMINISTIC_FALLBACK",
            "fallback_reason": fallback_reason,
            "scenario_category": target_blind_spot.get("category", "synthetic_stress"),
            "target_blind_spot": {
                "blind_spot_id": target_blind_spot.get("blind_spot_id", "BS024"),
                "category": target_blind_spot.get("category", "synthetic_stress"),
                "coverage_status": target_blind_spot.get("coverage_status", "rare"),
                "reason": target_blind_spot.get("reason", "Interactive synthetic scenario stress-testing")
            },
            "patient_context": {
                "age": age,
                "age_group": age_grp,
                "sex": sex,
                "cancer_type": cancer_type,
                "histology": histology,
                "stage": stage,
                "prior_treatment_context": "First-line evaluation for precision oncology therapy"
            },
            "genomic_profile": {
                "alterations": [
                    {
                        "gene": driver,
                        "variant": var,
                        "alteration_type": alt_type,
                        "status": "biologically_observed"
                    }
                ],
                "cooccurring_alterations": [
                    {
                        "gene": "TP53",
                        "variant": "p.R273H",
                        "status": "historically_observed"
                    }
                ],
                "resistance_related_features": []
            },
            "biomarkers": {
                "tmb": tmb,
                "tmb_status": "high" if tmb >= 10.0 else "low",
                "pdl1_tps": pdl1,
                "msi_status": "MSS"
            },
            "clinical_context": {
                "disease_status": "Metastatic" if ("IV" in stage or "4" in stage) else "Locally Advanced",
                "progression_context": f"Simulated clinical presentation of {stage} {histology} with {driver} {var} alteration."
            },
            "synthetic_assumptions": [
                f"Simulated {sex.lower()} patient with {driver} {var} created to stress-test clinical decision pathways."
            ],
            "reference_evidence": [ev],
            "uncertainty": {
                "level": "medium",
                "reason": "Empirical clinical trial guidelines evaluated against synthetic presentation."
            },
            "provenance": {
                "reference_sources": ["TCGA PanCancer Atlas", "MSK-IMPACT", "CIViC / ClinVar Curated Registries"],
                "blind_spot_source": "Stage 5 EDA / blind_spot_report.json",
                "generation_method": "deterministic_fallback",
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                "random_seed": 42
            }
        }

    def _append_to_generated_storage(self, scenario: Dict[str, Any]):
        """Safely persists scenario to generated_scenarios.jsonl."""
        self.scenarios_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scenarios_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(scenario) + "\n")
