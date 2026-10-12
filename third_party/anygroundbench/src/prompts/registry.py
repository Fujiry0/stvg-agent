"""Model-aware prompt registry for grounding tasks."""

from typing import Callable

from src.prompts.spatial import build_spatial_prompt
from src.prompts.spatio_temporal import build_spatio_temporal_prompt
from src.prompts.temporal import build_temporal_prompt

PromptBuilder = Callable[..., str]


def normalize_model_family(model_name: str) -> str:
    """Normalize model names to prompt-family names."""
    name = model_name.lower()
    if name in {"qwen", "qwen2", "qwen2_5", "qwen3", "qwen3_5"}:
        return "qwen"
    if name == "internvl":
        return "internvl"
    if name == "eagle":
        return "eagle"
    if name == "gpt":
        return "gpt"
    if name == "gemini":
        return "gemini"
    if name == "llava_st":
        return "llava_st"
    return name


_PROMPT_BUILDERS: dict[str, PromptBuilder] = {
    "temporal": build_temporal_prompt,
    "spatial": build_spatial_prompt,
    "spatio_temporal": build_spatio_temporal_prompt,
}


def build_prompt(
    task: str,
    model_name: str,
    query_text: str,
    text_only: bool=False,
    domain_name: str | None = None,
    prompt_variant: str = "base",
) -> str:
    """Build a prompt string for a task and model."""
    builder = _PROMPT_BUILDERS[task]

    return builder(
        model_family=normalize_model_family(model_name),
        query_text=query_text,
        text_only=text_only
    )
