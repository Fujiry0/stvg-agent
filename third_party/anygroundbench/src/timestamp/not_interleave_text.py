"""Implement non-interleaved textual encoding from arXiv 2512.14698v1 Appendix B.2."""
from src.timestamp.get_timestamp import get_duration, get_uniform_sample_timestamps


def add_timestamp(
    video_path: str,
    fps: int,
    max_frames: int,
    model_name: str
) -> str:
    """Add temporal metadata at the beginning of a question prompt.

    Note: This supports only Qwen uniform sampling.
        - Be careful with input-frame handling when using other models.
        - Random frame selection is not supported either.
    """
    if model_name in ["qwen2", "qwen2_5", "qwen3", "qwen3_5", "internvl", "eagle"]:
        num_frames, total_time, _, timestamp_at_each_frame = get_uniform_sample_timestamps(
            video_path = video_path,
            fps = fps,
            max_frames = max_frames
        )
        timestamp_prompt = (
            f" This video samples {num_frames} frames of a {total_time:.2f}-second video at "
            + ", ".join(f"t_{i+1}={t:.2f}" for i, t in enumerate(timestamp_at_each_frame))
            + " seconds. "
        )

    return timestamp_prompt

def add_duration(
    video_path: str,
    model_name: str | None = None,
):
    """Return a text prompt that describes the video duration."""
    duration_sec = get_duration(video_path)
    if model_name == "gemini":
        total_seconds = int(duration_sec)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        duration_prompt = f"This video is {minutes:02d}:{seconds:02d} seconds long.\n"
    else:
        duration_prompt = f"This video is {duration_sec:.2f} seconds long.\n"
    return duration_prompt
