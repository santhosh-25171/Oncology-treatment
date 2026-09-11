"""
Data Provenance and Audit Trail Utilities.
Ensures every generated synthetic scenario carries complete source traceability,
timestamps, random seeds, and generator identity.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List

DEFAULT_SOURCES = [
    "TCGA PanCancer Atlas (GDC)",
    "MSK-IMPACT Targeted Sequencing (cBioPortal)",
    "CIViC / ClinVar Curated Registries",
    "SEER Oncology Benchmarks"
]

BLIND_SPOT_REPORT_SOURCE = "Stage 5 EDA / blind_spot_report.json"


def generate_provenance_block(
    generation_method: str,
    random_seed: int = 42,
    reference_sources: List[str] = None,
    blind_spot_source: str = BLIND_SPOT_REPORT_SOURCE
) -> Dict[str, Any]:
    """Build a compliant provenance metadata object for a synthetic scenario."""
    return {
        "reference_sources": reference_sources or DEFAULT_SOURCES,
        "blind_spot_source": blind_spot_source,
        "generation_method": generation_method,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed
    }
