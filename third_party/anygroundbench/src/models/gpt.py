"""GPT / Azure OpenAI helpers."""
import base64
import io
import logging
import os
from typing import Any

from decord import VideoReader, cpu
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from src.message_builders.gpt_message import normalize_gpt_message_for_responses

load_dotenv()
logger = logging.getLogger("video_icl")



def load_gpt_model(model_id: str | None = None):
    """Load an OpenAI or Azure OpenAI client."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client, model_id


def _sample_video_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int | None = None,
    resize: int | None = None,
) -> list[str]:
    """Sample video frames and convert them to JPEG data URLs."""
    if VideoReader is None:
        raise RuntimeError("decord is required for GPT video processing.")

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    video_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 1.0
    if video_fps <= 0:
        video_fps = 1.0
    sample_every = max(1, int(round(video_fps / max(fps, 1e-6))))
    frame_indices = list(range(0, total_frames, sample_every))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    frames: list[str] = []
    for idx in frame_indices:
        frame = vr[idx].asnumpy()
        image = Image.fromarray(frame)
        frames.append(_pil_to_jpeg_data_url(image, resize))
    return frames


def _video_part_to_image_urls(part: dict[str, Any], resize: int | None = None) -> list[str]:
    """Convert a custom video content part into image data URLs."""
    video_path = part.get("video")
    if not isinstance(video_path, str):
        raise ValueError(f"GPT video part must have string video path: {part}")

    fps = part.get("fps")
    max_frames = part.get("max_frames")

    if VideoReader is None:
        raise RuntimeError("decord is required for GPT video processing.")

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    avg_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 1.0
    duration_sec = total_frames / avg_fps

    resolved_fps = float(fps) if fps is not None else 1.0
    if max_frames is not None:
        resolved_max_frames = max_frames
    else:
        resolved_max_frames = None if duration_sec < 120.0 else 120

    image_urls = _sample_video_frames(
        video_path=video_path,
        fps=resolved_fps,
        max_frames=resolved_max_frames,
        resize=resize,
    )
    frame_count = len(image_urls)
    if frame_count > 0:
        first_frame_size = _get_image_size_from_data_url(image_urls[0])
        logger.info(
            "GPT image inputs: video=%s frames=%d resolution=%dx%d fps=%.3f max_frames=%s",
            video_path,
            frame_count,
            first_frame_size[0],
            first_frame_size[1],
            resolved_fps,
            resolved_max_frames,
        )
    else:
        logger.info(
            "GPT image inputs: video=%s frames=0 fps=%.3f max_frames=%s",
            video_path,
            resolved_fps,
            resolved_max_frames,
        )
    return image_urls


def _pil_to_jpeg_data_url(image: Image.Image, max_side: int | None) -> str:
    """Convert an RGB image to a JPEG data URL after optional long-edge resizing."""
    image = image.convert("RGB")
    if max_side is not None:
        cap = max(64, int(max_side))
        if max(image.size) > cap:
            resized = image.copy()
            resized.thumbnail((cap, cap), Image.Resampling.LANCZOS)
            image = resized
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _get_image_size_from_data_url(data_url: str) -> tuple[int, int]:
    """Return the image size encoded in a data URL."""
    _, b64_data = data_url.split(",", 1)
    image_bytes = base64.b64decode(b64_data)
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


def get_gpt_inputs(
    messages: list[dict[str, Any]],
    resize: int | None = None,
) -> list[dict[str, Any]]:
    """Convert custom messages into GPT Responses API inputs."""
    side: int | None = max(64, int(resize)) if resize is not None else None

    def video_to_image_urls(part: dict[str, Any]) -> list[str]:
        """Convert one video part into GPT image URLs."""
        return _video_part_to_image_urls(part, resize=side)

    return [
        normalize_gpt_message_for_responses(
            message,
            video_to_image_urls=video_to_image_urls,
        )
        for message in messages
    ]


def build_gpt_response_create_kwargs(
    model_name: str,
    inputs: list[dict[str, Any]],
    max_output_tokens: int = 1024,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build kwargs for responses.create(...)."""
    kwargs: dict[str, Any] = {
        "model": model_name,
        "input": inputs,
        "max_output_tokens": max_output_tokens,
    }

    # Add reasoning only for GPT-5.1-family models.
    if reasoning_effort and model_name.startswith("gpt-5.1"):
        kwargs["reasoning"] = {"effort": reasoning_effort}

    return kwargs


def extract_text_from_response(response: Any) -> str:
    """Extract text from a Responses API return value."""
    # Newer SDK versions often expose output_text.
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # Fallback.
    try:
        texts: list[str] = []
        for item in response.output:
            for content in getattr(item, "content", []):
                text = getattr(content, "text", None)
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()
    except Exception:
        return str(response)
