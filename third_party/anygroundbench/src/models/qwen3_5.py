"""Load Qwen3.5 multimodal models and prepare their inputs."""

from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_qwen3_5_model(device: str, model_id: str | None = None):
    """Load a Qwen3.5 multimodal model.

    Notes:
        The following variants are available:
            - Qwen/Qwen3.5-0.8B
            - Qwen/Qwen3.5-2B
            - Qwen/Qwen3.5-4B
            - Qwen/Qwen3.5-9B
            - Qwen/Qwen3.5-27B
            - Qwen/Qwen3.5-35B-A3B
            - Qwen/Qwen3.5-122B-A10B
            - Qwen/Qwen3.5-397B-A17B
    """
    resolved_model_id = model_id or "Qwen/Qwen3.5-2B"
    model = AutoModelForImageTextToText.from_pretrained(
        resolved_model_id,
        dtype="auto",
        device_map=device,
        # device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(resolved_model_id)
    return model, processor


def get_qwen3_5_inputs(
    messages: list,
    processor: AutoProcessor,
    text: str,
    text_only: bool = False
):
    """Prepare inputs for a Qwen3.5 model.

    Notes:
        Supports uniform sampling, random selection, and reinforce selection.
    """
    if text_only:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        return inputs
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    # split the video_inputs and according metadatas (example pattern)
    if videos is not None:
        video_inputs, video_metadatas = zip(*videos)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None

    selected_video_inputs = video_inputs

    inputs = processor(
        text=[text],
        images=images,
        videos=selected_video_inputs,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    del images
    del video_inputs
    del video_metadatas
    del selected_video_inputs
    del video_kwargs
    return inputs
