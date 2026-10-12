"""Load Eagle models and prepare their inputs."""

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, BatchFeature
from transformers.models.siglip.modeling_siglip import SiglipVisionModel


def process_eagle_vision_info(
    processor: AutoProcessor,
    messages: list,
    *,
    return_video_kwargs: bool = True,
    return_video_metadata: bool = False,
    fps: int | None = None,
):
    """Normalize Eagle vision info to match qwen_vl_utils-style structure."""
    image_inputs, video_inputs, video_kwargs = processor.process_vision_info(
        messages,
        return_video_kwargs=return_video_kwargs,
    )
    if not return_video_metadata:
        return image_inputs, video_inputs, video_kwargs

    timestamps_all = video_kwargs.get("timestamps") if isinstance(video_kwargs, dict) else None
    videos_with_metadata = []
    for idx, video in enumerate(video_inputs):
        if video.ndim == 4:
            frame_count = int(video.shape[0])
        elif video.ndim == 3:
            frame_count = 1
        else:
            raise ValueError(f"Unexpected eagle video tensor shape: {tuple(video.shape)}")

        frame_ids = list(range(frame_count))
        if timestamps_all and idx < len(timestamps_all):
            ts_list = timestamps_all[idx]
            if isinstance(ts_list, list) and len(ts_list) == frame_count:
                used_fps = float(fps) if fps is not None else 1.0
                frame_ids = [int(round(float(ts) * used_fps)) for ts in ts_list]

        videos_with_metadata.append((video, {"frames_indices": frame_ids}))

    return image_inputs, videos_with_metadata, video_kwargs


def load_eagle_model(device: str, model_id: str | None = None):
    """Load an Eagle model and processor.

    Notes:
        The following variants are available:
            - nvidia/Eagle2.5-8B

            - nvidia/Eagle2-1B
            - nvidia/Eagle2-2B
            - nvidia/Eagle2-9B
    """
    resolved_model_id = model_id or "nvidia/Eagle2.5-8B"
    config = AutoConfig.from_pretrained(
        resolved_model_id,
        trust_remote_code=True,
    )
    config._attn_implementation = "sdpa"
    config._attn_implementation_internal = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation_internal = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.text_config._attn_implementation_internal = "sdpa"
    vision_model = SiglipVisionModel(config.vision_config).to(dtype=torch.bfloat16)

    model = AutoModel.from_pretrained(
        resolved_model_id,
        config=config,
        vision_model=vision_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.mlp1.to(dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(
        resolved_model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    processor.tokenizer.padding_side = "left"
    return model, processor


def get_eagle_inputs(
    messages: list,
    processor: AutoProcessor,
    text: str,
) -> BatchFeature:
    """Prepare inputs for an Eagle2.5 model."""
    image_inputs, video_inputs, video_kwargs = process_eagle_vision_info(
        processor=processor,
        messages=messages,
        return_video_kwargs=True,
        return_video_metadata=False,
    )

    selected_video_inputs = video_inputs

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=selected_video_inputs,
        padding=True,
        return_tensors="pt",
        videos_kwargs=video_kwargs,
    )

    del image_inputs
    del video_inputs
    del selected_video_inputs
    del video_kwargs
    return inputs
