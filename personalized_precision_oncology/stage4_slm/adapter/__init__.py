"""
Stage 4 SLM Adapter Package
Provides schema mapping and validation between Stage 1/2/3 runtime outputs and Stage 4 SLM.
"""

from .context_adapter import (
    adapt_stage123_to_stage4_context,
    validate_stage_responses,
    get_optimal_cpu_threads,
    build_stage4_prompt,
)

__all__ = [
    "adapt_stage123_to_stage4_context",
    "validate_stage_responses",
    "get_optimal_cpu_threads",
    "build_stage4_prompt",
]
