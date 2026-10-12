"""VideoClip: a rendered mp4 the agent can ``show()`` or pass to ``vlm.ask_with_thinking``.

Extension beyond SpatialClaw (whose observations are images only): temporal
grounding needs the model to see motion, so clips of the video are sent as
``video_url`` inputs. Rendering choices:
  - frames are sampled at ``fps`` (<= 2, the vLLM Qwen3-VL loader default, so every
    rendered frame reaches the model) and at most ``max_frames``;
  - frames are resized to at most ``max_pixels`` (384*384, as the single-pass baseline);
  - the absolute time ``t=..s`` is burned into every frame, because the model's own
    video timestamps start at 0 at the first frame of the clip;
  - optional per-frame boxes are drawn (``tools.Draw.tube``).
"""

import math
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_CLIP_FPS = 2.0
DEFAULT_MAX_PIXELS = 384 * 384


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _resize_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / (w * h))
    return img.resize((max(2, round(w * scale) // 2 * 2), max(2, round(h * scale) // 2 * 2)), Image.LANCZOS)


MARK_COLORS = [(255, 40, 40), (40, 140, 255), (255, 200, 0), (40, 220, 90), (220, 60, 255), (255, 130, 30),
               (0, 230, 230), (255, 255, 255)]


def annotate_frame(img: Image.Image, t: float, boxes: Optional[List[List[float]]] = None,
                   labels: Optional[List[str]] = None, note: str = "",
                   colors: Optional[List[tuple]] = None) -> Image.Image:
    """Draw pixel boxes (in ``img`` coordinates) and a ``t=..s`` banner. Labels get a filled tag."""
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width = max(2, img.height // 120)
    if boxes:
        font = _font(max(11, img.height // 18))
        for j, box in enumerate(boxes):
            if box is None:
                continue
            color = colors[j] if colors and j < len(colors) else (255, 40, 40)
            draw.rectangle([float(v) for v in box], outline=color, width=width)
            if labels and j < len(labels) and labels[j]:
                l, tp, r, b = draw.textbbox((0, 0), labels[j], font=font)
                x0, y0 = float(box[0]), float(box[1])
                draw.rectangle([x0, y0, x0 + r - l + 8, y0 + b - tp + 6], fill=color)
                draw.text((x0 + 4 - l, y0 + 3 - tp), labels[j], fill=(0, 0, 0), font=font)
    text = f"t={t:.1f}s" + (f" {note}" if note else "")
    font = _font(max(12, img.height // 16))
    l, tp, r, b = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, r - l + 8, b - tp + 8], fill=(0, 0, 0))
    draw.text((4 - l, 4 - tp), text, fill=(255, 255, 255), font=font)
    return img


class VideoClip:
    """A rendered clip: ``frame_indices`` (absolute) sampled from the source video."""

    def __init__(self, path: str, frame_indices: List[int], video_fps: float, clip_fps: float, label: str = ""):
        self.path = path
        self.frame_indices = list(frame_indices)
        self.video_fps = video_fps
        self.clip_fps = clip_fps
        self.label = label

    @property
    def start_time(self) -> float:
        return self.frame_indices[0] / self.video_fps

    @property
    def end_time(self) -> float:
        return self.frame_indices[-1] / self.video_fps

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    def describe(self) -> str:
        """Text sent next to the clip so the model can map its internal timestamps to video time."""
        return (f"Video clip{(' ' + repr(self.label)) if self.label else ''}: {self.num_frames} frames at "
                f"{self.clip_fps:.2f} fps covering video t={self.start_time:.1f}s-{self.end_time:.1f}s "
                f"(video frames {self.frame_indices[0]}-{self.frame_indices[-1]}). The clip's own time 0 = "
                f"video t={self.start_time:.1f}s; each frame shows its absolute video time in the top-left.")

    def __repr__(self) -> str:
        return (f"VideoClip({self.num_frames} frames @ {self.clip_fps:.2f} fps, t={self.start_time:.1f}-"
                f"{self.end_time:.1f}s, frames {self.frame_indices[0]}-{self.frame_indices[-1]}"
                + (f", label={self.label!r}" if self.label else "") + ")")


def sample_clip_frames(start_frame: int, end_frame: int, video_fps: float, fps: float,
                       max_frames: int) -> tuple[List[int], float]:
    """Frame indices in [start_frame, end_frame] at ``fps`` (capped), at most ``max_frames``."""
    fps = min(float(fps), MAX_CLIP_FPS)
    duration = (end_frame - start_frame) / video_fps
    n = int(math.floor(duration * fps)) + 1
    if n > max_frames:
        n = max_frames
    n = max(n, 2)
    indices = sorted(set(np.linspace(start_frame, end_frame, n).round().astype(int).tolist()))
    clip_fps = (len(indices) - 1) / duration if duration > 0 and len(indices) > 1 else fps
    return indices, clip_fps


def render_clip(frame_loader, frame_indices: List[int], video_fps: float, clip_fps: float, out_path: str,
                boxes_by_frame: Optional[Dict[int, List[List[float]]]] = None, box_labels: Optional[List[str]] = None,
                source_size: Optional[tuple] = None, max_pixels: int = DEFAULT_MAX_PIXELS, label: str = "",
                note_fn=None, box_colors: Optional[List[tuple]] = None) -> VideoClip:
    """Write an mp4 of ``frame_indices``; ``boxes_by_frame`` are pixel boxes in ``source_size`` coordinates."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = None
    try:
        for fi in frame_indices:
            img = frame_loader(fi)
            src_w, src_h = source_size or img.size
            small = _resize_max_pixels(img.convert("RGB"), max_pixels)
            sx, sy = small.width / src_w, small.height / src_h
            boxes = None
            if boxes_by_frame is not None:
                boxes = [None if b is None else [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                         for b in boxes_by_frame.get(fi, [])]
            note = note_fn(fi) if note_fn else ""
            frame = annotate_frame(small, fi / video_fps, boxes, box_labels, note, colors=box_colors)
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), clip_fps, frame.size)
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    return VideoClip(out_path, frame_indices, video_fps, clip_fps, label)
