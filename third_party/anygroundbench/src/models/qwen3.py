"""Load Qwen3-VL models and prepare their inputs."""

from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_qwen3vl_model(device: str, model_id: str | None = None):
    """Load a Qwen3-VL model and processor."""
    resolved_model_id = model_id or "Qwen/Qwen3-VL-8B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        resolved_model_id,
        dtype="auto",
        device_map=device,
        # device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        resolved_model_id
    )
    return model, processor

def get_qwen3vl_inputs(
    messages: list,
    processor: AutoProcessor,
    text: str,
):
    """Prepare inputs for a Qwen3-VL model.

    Notes:
        Supports uniform sampling, random selection, and reinforce selection.
    """
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
    # print(len(selected_video_inputs))
    # print(selected_video_inputs[0].shape)
    del images
    del video_inputs
    del video_metadatas
    del selected_video_inputs
    del video_kwargs
    return inputs
