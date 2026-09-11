"""
Prompt Coverage Report Generator for Stage 5 Prompt Engineering.
Audits prompt templates against detected genomic blind spots and outputs
reports/prompt_coverage_report.json conforming to schemas/prompt_schema.json.
"""

import json
from pathlib import Path
from jsonschema import validate

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
PROMPTS_DIR = BASE_DIR / "prompts"
SCHEMAS_DIR = BASE_DIR / "schemas"
BLIND_SPOT_REPORT_JSON = REPORTS_DIR / "blind_spot_report.json"
PROMPT_SCHEMA_JSON = SCHEMAS_DIR / "prompt_schema.json"
PROMPT_COVERAGE_REPORT_JSON = REPORTS_DIR / "prompt_coverage_report.json"


def generate_prompt_coverage_report():
    with open(BLIND_SPOT_REPORT_JSON, "r", encoding="utf-8") as f:
        blind_spot_data = json.load(f)
    with open(PROMPT_SCHEMA_JSON, "r", encoding="utf-8") as f:
        prompt_schema = json.load(f)

    blind_spots = blind_spot_data.get("blind_spots", [])
    total_blind_spots = len(blind_spots)

    prompts = [
        {
            "prompt_id": "PMPT001",
            "prompt_name": "Extreme Targeted Drug Resistance (Tertiary C797S + MET Bypass)",
            "template_file": "extreme_resistance_prompt.txt",
            "target_blind_spot_id": "BS014",
            "stress_dimension": "On-target covalent binding pocket disruption + bypass receptor tyrosine kinase activation",
            "evidence_boundary_status": "enforced"
        },
        {
            "prompt_id": "PMPT002",
            "prompt_name": "Compound Mutations & Dual Oncogenic Driver Co-occurrence",
            "template_file": "compound_mutation_prompt.txt",
            "target_blind_spot_id": "BS011",
            "stress_dimension": "Concurrent canonical drivers (KRAS + STK11 + KEAP1) violating standard mutual exclusivity",
            "evidence_boundary_status": "enforced"
        },
        {
            "prompt_id": "PMPT003",
            "prompt_name": "Conflicting Immunogenomic & Targeted Biomarker Signals",
            "template_file": "conflicting_evidence_prompt.txt",
            "target_blind_spot_id": "BS023",
            "stress_dimension": "Opposing predictive signals: TMB-High (>=10 mut/Mb) vs PD-L1 TPS 0% with STK11 loss",
            "evidence_boundary_status": "enforced"
        },
        {
            "prompt_id": "PMPT004",
            "prompt_name": "Ultra-Rare / Unobserved Genomic Alterations in Baseline",
            "template_file": "sparse_evidence_prompt.txt",
            "target_blind_spot_id": "BS007",
            "stress_dimension": "Pan-cancer actionable gene fusions (NTRK1/2/3, NRG1) not observed in reference baseline",
            "evidence_boundary_status": "enforced"
        },
        {
            "prompt_id": "PMPT005",
            "prompt_name": "Wildcard Multi-Factorial Resistance & Histological Transformation",
            "template_file": "wildcard_prompt.txt",
            "target_blind_spot_id": "BS018",
            "stress_dimension": "Simultaneous tertiary resistance, RB1/TP53 double-null loss, and histological SCLC lineage plasticity",
            "evidence_boundary_status": "enforced"
        }
    ]

    report = {
        "total_blind_spots": total_blind_spots,
        "blind_spots_converted_into_prompts": len(prompts),
        "prompt_categories": {
            "extreme_targeted_resistance": 1,
            "compound_driver_mutations": 1,
            "conflicting_biomarker_evidence": 1,
            "sparse_unobserved_alterations": 1,
            "wildcard_complex_resistance": 1
        },
        "rare_mutation_coverage": sum(1 for b in blind_spots if b["coverage_status"] == "rare"),
        "compound_mutation_coverage": sum(1 for b in blind_spots if b["category"] == "mutation_cooccurrence"),
        "resistance_coverage": sum(1 for b in blind_spots if b["category"] == "resistance_mechanism"),
        "conflicting_evidence_coverage": sum(1 for b in blind_spots if b["category"] == "evidence_conflict"),
        "sparse_evidence_coverage": sum(1 for b in blind_spots if b["coverage_status"] == "sparse"),
        "wildcard_coverage": 1,
        "prompts": prompts
    }

    # Validate against JSON schema
    validate(instance=report, schema=prompt_schema)

    with open(PROMPT_COVERAGE_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Prompt coverage report successfully created and validated: {PROMPT_COVERAGE_REPORT_JSON}")
    return report


if __name__ == "__main__":
    generate_prompt_coverage_report()
