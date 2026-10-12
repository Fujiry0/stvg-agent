"""Gemini helpers."""
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai

load_dotenv()

_UPLOADED_VIDEO_CACHE: dict[tuple[str, int | None], Any] = {}


def load_gemini_model(model_id: str | None = None):
    """Load the Gemini client and model name."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client, model_id


def upload_video_file(
    client: genai.Client,
    video_path: str,
    resize: int | None = None,
):
    """Upload a video file to Gemini and wait until it is active."""
    cache_key = (os.path.realpath(video_path), resize)
    cached = _UPLOADED_VIDEO_CACHE.get(cache_key)
    if cached is not None:
        return cached

    upload_path = video_path
    temp_path: str | None = None
    if resize is not None:
        from src.utils.video_resize import encode_video_max_long_edge

        temp_path = encode_video_max_long_edge(video_path, resize)
        upload_path = temp_path

    try:
        uploaded = client.files.upload(
            file=upload_path,
            config={"mime_type": "video/mp4"},
        )
    finally:
        if temp_path is not None and os.path.isfile(temp_path):
            os.remove(temp_path)

    file_name = getattr(uploaded, "name", None)
    if not file_name:
        _UPLOADED_VIDEO_CACHE[cache_key] = uploaded
        return uploaded

    for _ in range(120):
        current = client.files.get(name=file_name)
        state = str(getattr(current, "state", "")).upper()

        if "ACTIVE" in state:
            _UPLOADED_VIDEO_CACHE[cache_key] = current
            return current

        if "FAILED" in state:
            raise RuntimeError(f"Gemini file processing failed: {video_path}")

        time.sleep(2)
    raise TimeoutError(f"Gemini file processing timed out: {video_path}")


def cleanup_gemini_uploaded_files(client: genai.Client, file_names: list[str]) -> None:
    """Delete uploaded files from the API and invalidate the local cache."""
    seen: set[str] = set()
    for name in file_names:
        if not name or name in seen:
            continue
        seen.add(name)
        for key, val in list(_UPLOADED_VIDEO_CACHE.items()):
            if getattr(val, "name", None) == name:
                _UPLOADED_VIDEO_CACHE.pop(key, None)
        try:
            client.files.delete(name=name)
        except Exception as exc:
            print(f"Gemini files.delete failed ({name}): {exc}")


def delete_all_gemini_files(client: genai.Client) -> tuple[int, int]:
    """Delete all files from the Gemini Files API."""
    deleted = 0
    failed = 0
    for f in client.files.list():
        name = getattr(f, "name", None)
        if not isinstance(name, str) or not name:
            continue
        try:
            client.files.delete(name=name)
            deleted += 1
        except Exception:
            failed += 1
    return deleted, failed


def wait_until_gemini_file_active(
    client: genai.Client,
    uploaded_file: Any,
    timeout_sec: int,
    poll_interval_sec: int = 2,
):
    """Wait until a newly uploaded Gemini file becomes ACTIVE."""
    start = time.monotonic()
    current = uploaded_file
    while getattr(current.state, "name", None) != "ACTIVE":
        waited = time.monotonic() - start
        if waited > timeout_sec:
            raise TimeoutError(
                "Timed out waiting Gemini uploaded file to become ACTIVE. "
                f"file={getattr(current, 'name', '<unknown>')}, waited={int(waited)}s"
            )
        time.sleep(poll_interval_sec)
        current = client.files.get(name=current.name)
    return current


def get_gemini_inputs(
    client: genai.Client,
    messages: list[dict[str, Any]],
    fps: int | None = None,
    resize: int | None = None,
    uploaded_file_names_out: list[str] | None = None,
):
    """Convert OpenAI-style messages into Gemini SDK types.Content values.

    uploaded_file_names_out:
        When provided, append referenced `files/...` names for cleanup after inference.
    """
    from src.message_builders.gemini_message import normalize_gemini_messages

    return normalize_gemini_messages(
        messages=messages,
        upload_video=lambda p: upload_video_file(client, p, resize=resize),
        default_fps=fps,
        uploaded_file_names_out=uploaded_file_names_out,
    )
