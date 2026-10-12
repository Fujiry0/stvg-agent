"""Compute sampled frame timestamps and video durations."""

import math

import torch
from decord import VideoReader

FRAME_FACTOR=2

def get_uniform_sample_timestamps(
    video_path: str,
    fps: int,
    max_frames: int,
) -> tuple[int, float, list[int], list[float]]:
    """Use the same frame sampling strategy as Qwen."""
    vr = VideoReader(video_path)
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())

    min_frames = math.ceil(4 / FRAME_FACTOR) * FRAME_FACTOR
    max_frames = math.floor(min(max_frames, total_frames) / FRAME_FACTOR) * FRAME_FACTOR

    num_frames = total_frames / video_fps * fps
    num_frames = min(min(max(num_frames, min_frames), max_frames), total_frames)
    num_frames = math.floor(num_frames / FRAME_FACTOR) * FRAME_FACTOR


    frame_indices = torch.linspace(0, total_frames - 1, num_frames).round().long().tolist()
    timestamp_at_each_frame = [idx / video_fps for idx in frame_indices]
    total_time = total_frames / video_fps
    return num_frames, total_time, frame_indices, timestamp_at_each_frame

def get_duration(video_path: str):
    """Return the video duration in seconds."""
    vr = VideoReader(video_path)
    duration_sec = len(vr) / float(vr.get_avg_fps())

    return duration_sec
