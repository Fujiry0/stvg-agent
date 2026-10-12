"""Load InternVL3 models and prepare their inputs."""

import numpy as np
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_intern3_model(device: str, model_id: str | None = None):
    """Load an InternVL3 multimodal model and processor.

    Notes:
        The following variants are available:
            - OpenGVLab/InternVL3-1B
            - OpenGVLab/InternVL3-2B
            - OpenGVLab/InternVL3-8B
            - OpenGVLab/InternVL3-9B
            - OpenGVLab/InternVL3-14B
            - OpenGVLab/InternVL3-38B
            - OpenGVLab/InternVL3-78B

            - OpenGVLab/InternVL3_5-8B
            - OpenGVLab/InternVL3_5-14B
            - OpenGVLab/InternVL3_5-30B-A3B
            - OpenGVLab/InternVL3_5-38B
            - OpenGVLab/InternVL3_5-241B-A28B

        These are final checkpoint variants without suffixes.
    """
    resolved_model_id = model_id or "OpenGVLab/InternVL3-8B-hf"
    model = AutoModelForImageTextToText.from_pretrained(
        resolved_model_id,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(resolved_model_id)
    return model, processor


def get_intern3_inputs(
    messages: list,
    processor: AutoProcessor,
    text: str,
):
    """Prepare inputs for an InternVL3 model."""
    images, videos = process_vision_info(messages)
    video_inputs = videos

    selected_video_inputs = video_inputs

    if selected_video_inputs:
        query_num_frames = int(selected_video_inputs[-1].shape[0])
        aligned_video_inputs = []
        for video_input in selected_video_inputs[:-1]:
            current_num_frames = int(video_input.shape[0])
            if current_num_frames == query_num_frames:
                aligned_video_inputs.append(video_input)
                continue
            frame_indices = np.linspace(
                0,
                current_num_frames - 1,
                query_num_frames,
                dtype=np.int64,
            )
            aligned_video_inputs.append(video_input[frame_indices])
        aligned_video_inputs.append(selected_video_inputs[-1])
        selected_video_inputs = aligned_video_inputs

    inputs = processor(
        text=[text],
        images=images,
        videos=selected_video_inputs,
        padding=True,
        return_tensors="pt",
    )

    del images
    del video_inputs
    del selected_video_inputs
    return inputs
