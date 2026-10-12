"""Provide the model registry."""


def load_model_and_processor(model_name: str, device: str, model_id: str | None = None):
    """Load the model and processor for a model name."""
    # NOTE: Use lazy imports to preserve virtual environment separation.
    if model_name == "qwen2":
        from .qwen2 import load_qwen2vl_model
        return load_qwen2vl_model(device)
    if model_name == "qwen2_5":
        from .qwen2_5 import load_qwen2_5vl_model
        return load_qwen2_5vl_model(device, model_id)
    if model_name == "qwen3":
        from .qwen3 import load_qwen3vl_model
        return load_qwen3vl_model(device, model_id)
    if model_name == "qwen3_5":
        from .qwen3_5 import load_qwen3_5_model
        return load_qwen3_5_model(device, model_id)
    if model_name == "internvl":
        from .intern3 import load_intern3_model
        return load_intern3_model(device, model_id)
    if model_name == "eagle":
        from .eagle import load_eagle_model
        return load_eagle_model(device, model_id)
    if model_name == "llava_video":
        from .llava_video import load_llava_video_model
        return load_llava_video_model(device)
    if model_name == "llava_st":
        from .llava_st import load_llava_st_model
        return load_llava_st_model(device, model_id)
    if model_name == "video_llama3":
        from .video_llama3 import load_videollama3_model
        return load_videollama3_model(device)
    if model_name == "gemini":
        from .gemini import load_gemini_model
        return load_gemini_model(model_id)
    if model_name == "gpt":
        from .gpt import load_gpt_model
        return load_gpt_model(model_id)
    raise ValueError(f"Unsupported model_name: {model_name}")
