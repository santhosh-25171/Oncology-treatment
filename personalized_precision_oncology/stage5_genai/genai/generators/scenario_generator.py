"""
Unified Scenario Generator Facade.
Directs generation to LLM or Template mode based on runtime configuration.
"""

from typing import Dict, Any, List
from .template_generator import TemplateScenarioGenerator
from .llm_generator import LLMScenarioGenerator


class ScenarioGenerator:
    """Facade orchestrating scenario generation according to configuration."""

    def __init__(self, mode: str = "template", seed: int = 42):
        self.mode = mode
        self.seed = seed
        self.template_gen = TemplateScenarioGenerator(seed=seed)
        self.llm_gen = LLMScenarioGenerator(seed=seed)

    def generate(self, count: int = 20) -> List[Dict[str, Any]]:
        if self.mode.lower() == "llm":
            return self.llm_gen.generate_scenarios(count=count)
        else:
            return self.template_gen.generate_20_edge_cases()[:count]
