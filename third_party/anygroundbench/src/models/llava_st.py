"""Load LLaVA-ST models and prepare their inputs."""

import copy
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def _load_video_at_1fps(video_path: str, max_frames: int, default_load_video):
    """Load video frames at 1fps; fallback to default loader if over max."""
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Empty video: {video_path}")

    native_fps = float(vr.get_avg_fps() or 0.0)
    step = max(int(round(native_fps)), 1)
    frame_idx = list(range(0, total_frames, step))
    if not frame_idx:
        frame_idx = [0]

    if max_frames > 0 and len(frame_idx) > max_frames:
        return default_load_video(video_path=video_path, n_frms=max_frames)

    return vr.get_batch(frame_idx).asnumpy()

def _ensure_llava_st_import_path() -> None:
    """Add the local LLaVA-ST repository to sys.path."""
    llava_st_root = _get_llava_st_root()
    llava_st_path = str(llava_st_root)
    if llava_st_path not in sys.path:
        sys.path.insert(0, llava_st_path)


def _get_llava_st_root() -> Path:
    """Return the expected local LLaVA-ST repository path."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "LLaVA-ST"


@contextmanager
def _llava_st_cwd():
    """Temporarily change the current directory to the LLaVA-ST root."""
    prev_cwd = os.getcwd()
    llava_st_root = _get_llava_st_root()
    os.chdir(llava_st_root)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def load_llava_st_model(device: str, model_id: str | None = None):
    """Load a LLaVA-ST model bundle."""
    _ensure_llava_st_import_path()
    from llava.model.builder import load_pretrained_model

    resolved_model_id = model_id or os.getenv("LLAVA_ST_MODEL_ID", "appletea2333/LLaVA-ST-Qwen2-7B")
    device_map: str | dict[str, str] = {"": device} if device else "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        resolved_model_id,
        None,
        "llava_qwen",
        device_map=device_map,
    )
    model.model.init_vision_config(tokenizer)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "max_length": max_length,
        "vision_config": model.model.vision_config,
    }, None


def get_llava_st_inputs(
    conversations: list[dict[str, str]],
    query_video_path: str,
    query_max_frames: int,
    llava_bundle: dict,
):
    """Prepare LLaVA-ST input tensors from conversations and a query video."""
    _ensure_llava_st_import_path()
    with _llava_st_cwd():
        from demo.utils import format_conversations
        from inference.multi_task_inference import (
            preprocess_multimodal,
            preprocess_qwen,
        )
        from inference.src.datasets import load_video

        tokenizer = llava_bundle["tokenizer"]
        image_processor = llava_bundle["image_processor"]
        vision_config = llava_bundle["vision_config"]
        expected_frames = int(getattr(vision_config, "fast_frame_num", 100))

        selected_video_inputs = _load_video_at_1fps(
            video_path=query_video_path,
            max_frames=query_max_frames,
            default_load_video=load_video,
        )
        # LLaVA-ST inference path expects a fixed temporal token length
        # (typically 100). If 1fps sampling yields a different count,
        # fallback to the original loader with the expected frame count.
        if selected_video_inputs.shape[0] != expected_frames:
            selected_video_inputs = load_video(
                video_path=query_video_path,
                n_frms=expected_frames,
            )
        image_tensors = image_processor.preprocess(
            selected_video_inputs,
            return_tensors="pt",
        )["pixel_values"].cuda().half()

        formatted_conversations, variables = format_conversations(conversations)
        sources = preprocess_multimodal(copy.deepcopy([formatted_conversations]), vision_config)
        input_ids = preprocess_qwen(
            [sources[0][0], {"from": "gpt", "value": None}],
            tokenizer,
            has_image=True,
        ).cuda()
    return {
        "input_ids": input_ids,
        "image_tensors": image_tensors,
        "variables": [variables],
        "modality": "video",
    }
