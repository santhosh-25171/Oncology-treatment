"""
Pydantic Schemas for Stage 6 Knowledge and Evidence Layer.

Defines the core data contracts for evidence records, provenance tracking,
guideline rules, drug interaction rules, resistance patterns, and retrieval results.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class EvidenceStage(str, Enum):
    """Source stage producing or supplying the evidence."""
    STAGE1_EVIDENCE = "STAGE1_EVIDENCE"
    STAGE2_EVIDENCE = "STAGE2_EVIDENCE"
    STAGE3_EVIDENCE = "STAGE3_EVIDENCE"
    STAGE4_EVIDENCE = "STAGE4_EVIDENCE"
    STAGE5_EVIDENCE = "STAGE5_EVIDENCE"
    KNOWLEDGE_BASE_EVIDENCE = "KNOWLEDGE_BASE_EVIDENCE"


class EvidenceLevel(str, Enum):
    """Grading of evidence quality following oncology consensus standards."""
    LEVEL_A = "Level A"  # High-quality RCT / FDA CDx approved / NCCN Category 1
    LEVEL_B = "Level B"  # Prospective cohort / well-designed non-randomized / NCCN Category 2A
    LEVEL_C = "Level C"  # Retrospective cohort / case series / case report
    LEVEL_D = "Level D"  # Preclinical model / in vitro mechanism
    EXPERT_CONSENSUS = "Expert Consensus"  # Consensus panel / clinical guideline consensus
    MODEL_INFERENCE = "Model Inference"  # Output of calibrated/validated Stage 1-5 models


class EvidenceCategory(str, Enum):
    """Broad domain classification of evidence."""
    GUIDELINE = "guideline"
    GENOMIC = "genomic"
    TREATMENT_RESPONSE = "treatment_response"
    TOXICITY = "toxicity"
    RESISTANCE = "resistance"
    DRUG_INTERACTION = "drug_interaction"
    CLINICAL_TRIAL = "clinical_trial"
    MULTIMODAL_IMAGING = "multimodal_imaging"
    TRIAGE_NLP = "triage_nlp"
    BEDSIDE_BRIEFING = "bedside_briefing"


class ProvenanceRecord(BaseModel):
    """Full provenance and attribution trace for any evidence record."""
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(description="Name of the source institution, guideline, or repository")
    source_dataset: Optional[str] = Field(default=None, description="Dataset file or registry name")
    source_study: Optional[str] = Field(default=None, description="Specific clinical trial, paper, or benchmark study")
    source_module: Optional[str] = Field(default=None, description="Exact python module/class/function producing or storing the evidence")
    publication: Optional[str] = Field(default=None, description="Peer-reviewed publication citation")
    doi_or_pmid: Optional[str] = Field(default=None, description="Unique DOI or PubMed ID identifier")
    citation: Optional[str] = Field(default=None, description="Formal academic citation string")
    url: Optional[str] = Field(default=None, description="Public repository or access URL")
    access_date: Optional[str] = Field(default=None, description="Date accessed or generated (YYYY-MM-DD)")
    license: Optional[str] = Field(default=None, description="Data license or usage agreement")


class EvidenceRecord(BaseModel):
    """
    Primary atomic evidence record in Stage 6.
    Every clinical statement, model prediction, and guideline rule must be expressed
    as an EvidenceRecord with full provenance and no hallucinated facts.
    """
    model_config = ConfigDict(extra="allow")

    evidence_id: str = Field(description="Deterministic unique ID for the evidence record")
    source_stage: EvidenceStage = Field(description="Stage generating or storing this evidence")
    source_module: str = Field(description="Module/file/function of origin")
    category: str = Field(description="Domain category (genomic, toxicity, resistance, guideline, etc.)")
    topic: str = Field(description="Subject topic, e.g., 'EGFR L858R First-Line Therapy'")
    matched_terms: List[str] = Field(default_factory=list, description="Keywords or normalized terms associated with record")
    evidence_text: str = Field(description="Clinical or computational evidence narrative")
    evidence_level: EvidenceLevel = Field(description="Grading of evidence quality")
    safety_note: Optional[str] = Field(default=None, description="Clinical safety warning, contraindication, or precaution")
    provenance: ProvenanceRecord = Field(description="Full traceability attribution")
    limitations: Optional[str] = Field(default=None, description="Known evidence limitations, uncertainty, or cohort gaps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Structured numerical or categorical payloads")


class DrugInteractionRule(BaseModel):
    """Structured pharmacological interaction and safety rule."""
    model_config = ConfigDict(extra="forbid")

    drug_or_class: str = Field(description="Primary drug or pharmacological class")
    interacting_drug_or_class: str = Field(description="Interacting drug, class, or dietary factor")
    interaction_type: str = Field(description="Mechanism, e.g., CYP3A4_INDUCTION, QT_PROLONGATION, SYNERGISTIC_NEPHROTOXICITY")
    severity: str = Field(description="Severity classification: CONTRAINDICATED, MAJOR, MODERATE, MINOR")
    warning: str = Field(description="Clinical warning and management recommendation")
    source: str = Field(description="Authoritative pharmacology source (e.g. FDA Label, NCCN Compendium)")
    evidence_level: str = Field(default="Level A", description="Evidence grade supporting interaction")
    review_required: bool = Field(default=True, description="Flag indicating multidisciplinary clinical review is mandatory")


class ResistanceRule(BaseModel):
    """Structured resistance mechanism rule linked to clinical or synthetic evidence."""
    model_config = ConfigDict(extra="allow")

    gene: str = Field(description="Target gene symbol (e.g. EGFR, ALK, KRAS)")
    variant: str = Field(description="Specific alteration or variant (e.g. p.T790M, p.C797S, Amplification)")
    drug_associated: str = Field(description="Drug or therapy receiving resistance (e.g. Osimertinib, Erlotinib, Sotorasib)")
    drug_class: str = Field(description="Drug class (e.g. 1st/2nd Gen EGFR TKI, KRAS G12C Inhibitor)")
    resistance_phenotype: str = Field(description="Phenotypic description of resistance mechanism")
    resistance_category: str = Field(default="on_target_resistance", description="Category: on_target, bypass, histological, co_mutation")
    evidence_level: str = Field(default="Level A", description="Clinical or benchmark evidence level")
    clinical_significance: str = Field(description="Clinical implication or recommended action")
    pubmed_id: Optional[str] = Field(default=None, description="PubMed ID for supporting literature")
    doi: Optional[str] = Field(default=None, description="DOI for supporting literature")
    blind_spot_id: Optional[str] = Field(default=None, description="Corresponding Stage 5 blind spot ID if applicable")
    description: Optional[str] = Field(default=None, description="Detailed biological rationale")


class TermMatch(BaseModel):
    """Normalized ontology term match result."""
    model_config = ConfigDict(extra="forbid")

    canonical_id: str = Field(description="Unique ontology term identifier (e.g. GENE:EGFR, MUT:L858R)")
    category: str = Field(description="Category: cancer_type, gene, mutation, biomarker, drug, adverse_event, response, progression, resistance")
    canonical_name: str = Field(description="Standardized canonical display name")
    matched_text: str = Field(description="Original token string matched in text")
    synonyms: List[str] = Field(default_factory=list, description="Known aliases/synonyms")
    description: Optional[str] = Field(default=None, description="Brief biological/clinical definition")


class RetrievalResult(BaseModel):
    """Ranked retrieval payload returned to agents and callers."""
    model_config = ConfigDict(extra="allow")

    evidence_id: str = Field(description="ID of the retrieved evidence record")
    relevance_score: float = Field(description="Deterministic lexical/BM25 relevance score")
    matched_terms: List[str] = Field(description="Query terms matched against this record")
    source: str = Field(description="Source stage or knowledge base identifier")
    evidence_level: str = Field(description="Evidence grade")
    safety_note: Optional[str] = Field(default=None, description="Safety or warning notice")
    provenance: ProvenanceRecord = Field(description="Attribution record")
    record: EvidenceRecord = Field(description="Complete underlying evidence record")


class RetrievalResponse(BaseModel):
    """Structured response container for query execution."""
    model_config = ConfigDict(extra="allow")

    query: str = Field(description="Original input query")
    status: str = Field(description="SUCCESS or EVIDENCE_NOT_FOUND")
    total_results: int = Field(description="Count of matched evidence records")
    results: List[RetrievalResult] = Field(default_factory=list, description="Ranked list of results")
    disclaimer: str = Field(
        default="Academic Clinical Decision-Support Knowledge Base. Not a clinical prescribing system.",
        description="Mandatory clinical safety disclaimer"
    )
