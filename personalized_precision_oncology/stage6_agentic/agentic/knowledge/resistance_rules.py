"""
Resistance Rules and Stage 5 Stress-Testing Knowledge Interface for Stage 6 Agentic AI.

Consumes Stage 5 GenAI artifacts in a STRICTLY READ-ONLY mode:
- stage5_genai/genai/scenarios/synthetic_edge_cases.jsonl
- stage5_genai/genai/scenarios/generated_scenarios.jsonl
- stage5_genai/integration/history/evaluation_history.jsonl
- stage5_genai/data_engineering/processed/genai_reference_baseline.jsonl

Indexes on-target mutations, solvent-front steric clashes, bypass RTK activations,
histological lineage transformations, and immune microenvironment evasion mechanisms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import (
    EvidenceCategory,
    EvidenceLevel,
    EvidenceRecord,
    EvidenceStage,
    ProvenanceRecord,
    ResistanceRule,
)


class ResistanceRuleEngine:
    """
    Deterministic resistance rule evaluator grounded in Stage 5 stress-testing datasets
    and curated genomic registries (CIViC, ClinVar, MSK-IMPACT).
    """

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        if repo_root is None:
            curr = Path(__file__).resolve()
            while curr.name != "personalized_precision_oncology" and curr.parent != curr:
                curr = curr.parent
            self._repo_root = curr.parent
        else:
            self._repo_root = repo_root

        self._stage5_dir = self._repo_root / "personalized_precision_oncology" / "stage5_genai"
        self._rules: List[ResistanceRule] = []
        self._edge_cases: List[Dict[str, Any]] = []
        self._evaluation_history: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load all Stage 5 reference information in read-only mode."""
        self._load_reference_baseline()
        self._load_synthetic_edge_cases()
        self._load_evaluation_history()

    def _load_reference_baseline(self) -> None:
        """Parse CIViC/ClinVar resistance patterns from genai_reference_baseline.jsonl."""
        baseline_file = self._stage5_dir / "data_engineering" / "processed" / "genai_reference_baseline.jsonl"
        if not baseline_file.exists():
            return

        with open(baseline_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    if record.get("data_domain") == "resistance" and "resistance_patterns" in record:
                        for item in record["resistance_patterns"]:
                            rule = ResistanceRule(
                                gene=item.get("gene", "UNKNOWN"),
                                variant=item.get("variant", "UNKNOWN"),
                                drug_associated=item.get("drug_associated", "UNKNOWN"),
                                drug_class=item.get("drug_class", "Targeted Agent"),
                                resistance_phenotype=item.get("resistance_phenotype", "Acquired Resistance"),
                                resistance_category=self._categorize_mechanism(item.get("resistance_phenotype", "")),
                                evidence_level=item.get("evidence_level", "Level A"),
                                clinical_significance=item.get("clinical_significance", "Therapeutic Resistance"),
                                pubmed_id=item.get("pubmed_id"),
                                description=f"{item.get('gene')} {item.get('variant')} mediates {item.get('resistance_phenotype')} to {item.get('drug_associated')}."
                            )
                            self._rules.append(rule)
                except json.JSONDecodeError:
                    continue

    def _load_synthetic_edge_cases(self) -> None:
        """Load the 20 immutable synthetic edge cases from Stage 5."""
        edge_file = self._stage5_dir / "genai" / "scenarios" / "synthetic_edge_cases.jsonl"
        if not edge_file.exists():
            return

        with open(edge_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    scenario = json.loads(line_str)
                    self._edge_cases.append(scenario)

                    # Extract resistance features as rules
                    genomic = scenario.get("genomic_profile", {})
                    res_features = genomic.get("resistance_related_features", [])
                    blind_spot = scenario.get("target_blind_spot", {})
                    assumptions = scenario.get("synthetic_assumptions", [""])[0]

                    for rf in res_features:
                        gene = rf.get("gene", "UNKNOWN")
                        variant = rf.get("variant", "UNKNOWN")
                        alt_type = rf.get("alteration_type", "Resistance")

                        rule = ResistanceRule(
                            gene=gene,
                            variant=variant,
                            drug_associated=scenario.get("patient_context", {}).get("prior_treatment_context", "Prior Therapy"),
                            drug_class="Targeted / Multi-line Therapy",
                            resistance_phenotype=f"{alt_type}: {variant}",
                            resistance_category=scenario.get("scenario_category", "compound_resistance"),
                            evidence_level="Level B",
                            clinical_significance=f"Stress-tested in Stage 5 ({blind_spot.get('blind_spot_id', 'EDGE')})",
                            blind_spot_id=blind_spot.get("blind_spot_id"),
                            description=assumptions
                        )
                        self._rules.append(rule)
                except json.JSONDecodeError:
                    continue

    def _load_evaluation_history(self) -> None:
        """Read evaluation history log records."""
        history_file = self._stage5_dir / "integration" / "history" / "evaluation_history.jsonl"
        if not history_file.exists():
            return

        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    eval_record = json.loads(line_str)
                    self._evaluation_history.append(eval_record)
                except json.JSONDecodeError:
                    continue

    def _categorize_mechanism(self, phenotype: str) -> str:
        """Categorize resistance mechanism based on phenotype description."""
        pheno_lower = phenotype.lower()
        if "gatekeeper" in pheno_lower:
            return "gatekeeper_mutation"
        if "solvent" in pheno_lower:
            return "solvent_front_mutation"
        if "covalent" in pheno_lower or "c797s" in pheno_lower or "tertiary" in pheno_lower:
            return "on_target_tertiary"
        if "bypass" in pheno_lower or "amplification" in pheno_lower:
            return "bypass_rtk_activation"
        if "sclc" in pheno_lower or "histological" in pheno_lower or "plasticity" in pheno_lower:
            return "histological_transformation"
        if "cold" in pheno_lower or "evasion" in pheno_lower or "stk11" in pheno_lower:
            return "immune_cold_microenvironment"
        return "on_target_resistance"

    def get_resistance_rules_for_gene(self, gene: str) -> List[ResistanceRule]:
        """Query all resistance rules matching target gene symbol."""
        target = gene.strip().upper()
        return [r for r in self._rules if r.gene.upper() == target]

    def get_resistance_rules_for_variant(self, gene: str, variant: str) -> List[ResistanceRule]:
        """Query resistance rules for a specific gene and variant."""
        gene_target = gene.strip().upper()
        var_target = variant.strip().lower()

        results: List[ResistanceRule] = []
        for r in self._rules:
            if r.gene.upper() == gene_target:
                if var_target in r.variant.lower() or r.variant.lower() in var_target:
                    results.append(r)
        return results

    def get_edge_case_scenarios(
        self,
        category: Optional[str] = None,
        gene: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter the 20 Stage 5 edge-case scenarios by category or gene alteration."""
        results: List[Dict[str, Any]] = []
        cat_lower = category.lower() if category else None
        gene_upper = gene.upper() if gene else None

        for sc in self._edge_cases:
            if cat_lower and cat_lower not in sc.get("scenario_category", "").lower():
                continue
            if gene_upper:
                # Check genomic alterations
                alts = sc.get("genomic_profile", {}).get("alterations", [])
                res_f = sc.get("genomic_profile", {}).get("resistance_related_features", [])
                cooc = sc.get("genomic_profile", {}).get("cooccurring_alterations", [])
                all_genes = [a.get("gene", "").upper() for a in alts + res_f + cooc]
                if gene_upper not in all_genes:
                    continue
            results.append(sc)

        return results

    def get_all_rules(self) -> List[ResistanceRule]:
        """Return all indexed resistance rules."""
        return list(self._rules)

    def as_evidence_records(self) -> List[EvidenceRecord]:
        """Convert resistance rules and edge-case insights into EvidenceRecord instances."""
        records: List[EvidenceRecord] = []
        for idx, rule in enumerate(self._rules, start=1):
            rec = EvidenceRecord(
                evidence_id=f"KB-RESIST-{rule.gene}-{idx:03d}",
                source_stage=EvidenceStage.STAGE5_EVIDENCE if rule.blind_spot_id else EvidenceStage.KNOWLEDGE_BASE_EVIDENCE,
                source_module="stage6_agentic.agentic.knowledge.resistance_rules",
                category=EvidenceCategory.RESISTANCE.value,
                topic=f"Resistance Mechanism: {rule.gene} {rule.variant} to {rule.drug_associated}",
                matched_terms=[
                    rule.gene.lower(),
                    rule.variant.lower(),
                    rule.drug_associated.lower(),
                    "resistance",
                    rule.resistance_category.lower(),
                    rule.resistance_phenotype.lower()
                ],
                evidence_text=(
                    f"Genomic Alteration: {rule.gene} {rule.variant}. Associated Drug: {rule.drug_associated} "
                    f"({rule.drug_class}). Mechanism: {rule.resistance_phenotype}. "
                    f"Clinical Significance: {rule.clinical_significance}."
                ),
                evidence_level=EvidenceLevel.LEVEL_A if rule.evidence_level == "Level A" else EvidenceLevel.LEVEL_B,
                safety_note=f"High risk of disease progression or treatment failure under {rule.drug_associated}.",
                provenance=ProvenanceRecord(
                    source_name="Stage 5 Stress-Testing / CIViC-ClinVar Curated Registries",
                    source_dataset="genai_reference_baseline.jsonl / synthetic_edge_cases.jsonl",
                    source_module="stage5_genai.data_engineering / stage5_genai.genai",
                    publication="Griffith et al. Nat Genet 2017; MSK-IMPACT Zehir et al. Nat Med 2017",
                    doi_or_pmid=rule.pubmed_id or rule.doi or "Stage 5 GenAI Curation",
                    citation=f"Stage 5 Resistance Rule: {rule.gene} {rule.variant} ({rule.resistance_category})",
                    access_date="2026-09-11",
                    license="Creative Commons Zero (CC0) / Public Domain"
                ),
                limitations="Acquired resistance patterns evolve dynamically under selective targeted pressure.",
                metadata={
                    "gene": rule.gene,
                    "variant": rule.variant,
                    "drug": rule.drug_associated,
                    "category": rule.resistance_category,
                    "blind_spot_id": rule.blind_spot_id
                }
            )
            records.append(rec)
        return records
