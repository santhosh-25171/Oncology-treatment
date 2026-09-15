"""
Stage 6: Agentic AI — Personalized Precision Medicine for Oncology Treatment Optimization.

Role-wise modular architecture:
- data: Data schemas, adapters, preprocessing, contracts, and provenance
- eda: Exploratory data analysis, quality audits, distribution checks
- agentic: Specialist oncology agents, deterministic knowledge retrieval, and workflow orchestration
- evaluation: Multi-agent evaluation benchmarks, metrics, and validation reports
- integration: Clinical API, Streamlit dashboard, physician review interface, and audit trail
"""

from personalized_precision_oncology.stage6_agentic import agentic
from personalized_precision_oncology.stage6_agentic import data
from personalized_precision_oncology.stage6_agentic import eda
from personalized_precision_oncology.stage6_agentic import evaluation
from personalized_precision_oncology.stage6_agentic import integration

# Backward-compatibility aliases
from personalized_precision_oncology.stage6_agentic.agentic import agents
from personalized_precision_oncology.stage6_agentic.agentic import knowledge
from personalized_precision_oncology.stage6_agentic.agentic import workflow

__version__ = "0.1.0"

__all__ = [
    "agentic",
    "data",
    "eda",
    "evaluation",
    "integration",
    "agents",
    "knowledge",
    "workflow",
]
