"""Select message builder functions for supported video-language models."""

from collections.abc import Callable
from typing import Any

from src.message_builders.eagle_message import (
    eagle_prepare_demonstration,
    eagle_prepare_demonstration_text_only,
    eagle_prepare_query,
)
from src.message_builders.llava_st_message import (
    llava_st_prepare_demonstration,
    llava_st_prepare_demonstration_text_only,
    llava_st_prepare_query,
)
from src.message_builders.qwen_message import (
    qwen_prepare_demonstration,
    qwen_prepare_demonstration_text_only,
    qwen_prepare_query,
    qwen_prepare_query_text_only,
)

MessageBuilder = Callable[..., list[dict[str, Any]]]


def build_demonstration_message_builder(
    model_name: str,
    text_only: bool = False
) -> MessageBuilder:
    """Return the demonstration message builder for a model."""
    if model_name in {"qwen3", "qwen3_5", "internvl"}:
        if text_only:
            return qwen_prepare_demonstration_text_only
        return qwen_prepare_demonstration
    if model_name == "llava_st":
        if text_only:
            return llava_st_prepare_demonstration_text_only
        return llava_st_prepare_demonstration
    if model_name == "eagle":
        if text_only:
            return eagle_prepare_demonstration_text_only
        return eagle_prepare_demonstration
    if model_name == "gemini":
        from src.message_builders.gemini_message import gemini_prepare_demonstration
        if text_only:
            return qwen_prepare_demonstration_text_only
        return gemini_prepare_demonstration
    if model_name == "gpt":
        from src.message_builders.gpt_message import gpt_prepare_demonstration
        return gpt_prepare_demonstration
    raise ValueError(f"Unsupported model_name for demonstration builder: {model_name}")


def build_query_message_builder(
    model_name: str,
    text_only: bool = False,
) -> MessageBuilder:
    """Return the query message builder for a model."""
    if model_name in {"qwen2", "qwen2_5", "qwen3", "qwen3_5", "internvl"}:
        if text_only:
            return qwen_prepare_query_text_only
        return qwen_prepare_query
    if model_name == "llava_st":
        return llava_st_prepare_query
    if model_name == "eagle":
        return eagle_prepare_query
    if model_name == "gemini":
        from src.message_builders.gemini_message import (
            gemini_prepare_query,
            gemini_prepare_query_text_only,
        )
        if text_only:
            return gemini_prepare_query_text_only
        return gemini_prepare_query
    if model_name == "gpt":
        from src.message_builders.gpt_message import gpt_prepare_query
        return gpt_prepare_query
    raise ValueError(f"Unsupported model_name for query builder: {model_name}")
