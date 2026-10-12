"""Validate command-line arguments for AnyGroundBench runs."""

import argparse

DOMAINS = {"animal", "industry", "safety", "sports", "surgery"}
ALLOWED_ADD_PROMPTS = {"timestamp", "duration"}
TIMESTAMP_SUPPORTED_MODELS = {"qwen3", "qwen3_5", "internvl", "eagle", "llava_st"}
TEXT_ONLY_DEMONSTRATION_SUPPORTED_MODELS = {"qwen3", "qwen3_5", "internvl", "eagle", "llava_st", "gemini"}


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate argument combinations and report parser errors."""
    errors = []

    if args.domain_name not in DOMAINS:
        errors.append(f"--domain_name must be in {sorted(DOMAINS)}")

    # task-specific args
    if args.task in {"temporal", "spatio_temporal", "spatial"}:
        if args.n_shot == 0 and args.alpha is not None:
            errors.append(f"task={args.task} with n_shot=0 does not support --alpha")
        if args.alpha is None and args.n_shot != 0:
            errors.append(f"task={args.task} requires --alpha")

    # temporal prompt constraints
    add_prompt = getattr(args, "add_prompt", []) or []

    invalid_add_prompt = sorted(set(add_prompt) - ALLOWED_ADD_PROMPTS)
    if invalid_add_prompt:
        errors.append(f"--add_prompt must contain only {sorted(ALLOWED_ADD_PROMPTS)}")

    if len(add_prompt) != len(set(add_prompt)):
        errors.append("--add_prompt must not contain duplicate values")

    if "timestamp" in add_prompt:
        if args.model_name not in TIMESTAMP_SUPPORTED_MODELS:
            errors.append(
                f"--add_prompt timestamp requires model_name in {sorted(TIMESTAMP_SUPPORTED_MODELS)}"
            )

    if getattr(args, "text_only_demonstration", False):
        if args.model_name not in TEXT_ONLY_DEMONSTRATION_SUPPORTED_MODELS:
            errors.append(
                "--text_only_demonstration currently supports only model_name in "
                f"{sorted(TEXT_ONLY_DEMONSTRATION_SUPPORTED_MODELS)}"
            )

    if errors:
        parser.error("Invalid arguments:\n- " + "\n- ".join(errors))
