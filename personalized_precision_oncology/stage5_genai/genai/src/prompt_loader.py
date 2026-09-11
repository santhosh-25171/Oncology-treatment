"""
Prompt Template Loader.
Loads Stage 5 prompt templates and manages placeholder injection for generative scenarios.
"""

from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EDA_PROMPTS_DIR = BASE_DIR / "eda_prompteng" / "prompts"
GENAI_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptLoader:
    """Loads and formats prompt templates for structured scenario generation."""

    def __init__(self):
        self.eda_prompts = self._load_prompt_files(EDA_PROMPTS_DIR)
        self.genai_prompts = self._load_prompt_files(GENAI_PROMPTS_DIR)

    def _load_prompt_files(self, dir_path: Path) -> Dict[str, str]:
        prompts = {}
        if not dir_path.exists():
            return prompts
        for file in dir_path.glob("*.txt"):
            prompts[file.name] = file.read_text(encoding="utf-8")
        return prompts

    def get_system_prompt(self) -> str:
        """Retrieve the primary system prompt."""
        if "system_prompt.txt" in self.genai_prompts:
            return self.genai_prompts["system_prompt.txt"]
        if "system_prompt.txt" in self.eda_prompts:
            return self.eda_prompts["system_prompt.txt"]
        return "You are generating a synthetic oncology stress-test scenario, not a real patient."

    def get_generation_prompt_template(self) -> str:
        """Retrieve the scenario generation template with placeholders."""
        if "scenario_generation_prompt.txt" in self.genai_prompts:
            return self.genai_prompts["scenario_generation_prompt.txt"]
        return ""

    def format_generation_prompt(self, context: Dict[str, Any]) -> str:
        """Inject values into placeholders."""
        template = self.get_generation_prompt_template()
        for k, v in context.items():
            placeholder = f"{{{{{k}}}}}"
            template = template.replace(placeholder, str(v))
        return template
