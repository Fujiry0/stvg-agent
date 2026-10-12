"""``tools.Video`` (clips for temporal inspection) and ``tools.Draw`` (visualize a predicted tube).

Not in SpatialClaw: added so the agent can watch the video (temporal grounding)
and check the tube it is about to submit.
"""

import math
import os
from typing import Dict, List, Optional

from PIL import Image

from stvg_agent.kernel_types import coords
from stvg_agent.kernel_types.video_clip import (
    DEFAULT_MAX_PIXELS,
    VideoClip,
    annotate_frame,
    render_clip,
    sample_clip_frames,
)
from stvg_agent.kernel_types.visual_feedback import VisualFeedback


class _ClipFactory:
    def __init__(self, input_images, video_fps: float, clip_dir: str):
        self._ii = input_images
        self._fps = video_fps
        self._dir = clip_dir
        self._n = len(input_images)
        self._counter = 0
        self._size = (input_images[0].width, input_images[0].height)

    def _load(self, fi: int) -> Image.Image:
        return self._ii[fi].image

    def _range(self, start_frame: Optional[int], end_frame: Optional[int]) -> tuple[int, int]:
        s = 0 if start_frame is None else max(0, int(start_frame))
        e = self._n - 1 if end_frame is None else min(self._n - 1, int(end_frame))
        if e <= s:
            raise ValueError(f"end_frame ({end_frame}) must be greater than start_frame ({start_frame}).")
        return s, e

    def _next_path(self, prefix: str) -> str:
        self._counter += 1
        return os.path.join(self._dir, f"{prefix}_{self._counter:04d}.mp4")


class VideoTool(_ClipFactory):
    def clip(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, fps: float = 1.0,
             max_frames: int = 120, label: str = "") -> VideoClip:
        """Clip of video frames [start_frame, end_frame] (inclusive) at ``fps`` (<= 2), at most ``max_frames``.

        Each frame carries its absolute time. Pass it to ``show(clip)`` or
        ``vlm.ask_with_thinking(clip, question)``.
        """
        s, e = self._range(start_frame, end_frame)
        indices, clip_fps = sample_clip_frames(s, e, self._fps, fps, max_frames)
        return render_clip(self._load, indices, self._fps, clip_fps, self._next_path("clip"),
                           max_pixels=DEFAULT_MAX_PIXELS, label=label)

    def __repr__(self) -> str:
        return "VideoTool(clip)"


class DrawTool(_ClipFactory):
    @staticmethod
    def _as_track(tube, object) -> Dict[int, List[float]]:
        if hasattr(tube, "get_track"):
            tube = tube.get_track(object)
        elif object != 0:
            raise ValueError("`object` applies only to PerFrameBoxes; a {frame: box} dict already holds one "
                             "object's boxes, so check which object you copied the boxes from.")
        if not isinstance(tube, dict) or not tube:
            raise TypeError("tube must be a non-empty dict {frame_index: [x1, y1, x2, y2]} or PerFrameBoxes")
        return {int(k): [float(v) for v in b] for k, b in tube.items() if b is not None}

    def tube(self, tube, object=0, start_frame: Optional[int] = None, end_frame: Optional[int] = None,
             margin_sec: float = 2.0, as_video: bool = True, fps: float = 2.0, max_frames: int = 120,
             num_frames: int = 8, label: str = "pred"):
        """Visualize a tube ({frame: pixel box} or PerFrameBoxes) over its span +/- ``margin_sec``.

        ``as_video=True`` → VideoClip (boxes drawn, frames without a box marked "(no box)").
        ``as_video=False`` → one grid image (VisualFeedback) of ``num_frames`` evenly spaced frames.
        ``start_frame``/``end_frame`` set the rendered range explicitly.
        """
        track = {fi: coords.active().to_pixel(b, *self._size) for fi, b in self._as_track(tube, object).items()}
        frames = sorted(track)
        margin = int(round(margin_sec * self._fps))
        s, e = self._range(frames[0] - margin if start_frame is None else start_frame,
                           frames[-1] + margin if end_frame is None else end_frame)
        if as_video:
            indices, clip_fps = sample_clip_frames(s, e, self._fps, fps, max_frames)
        else:
            n = max(2, num_frames)
            indices = sorted({round(s + i * (e - s) / (n - 1)) for i in range(n)})
            clip_fps = None
        tol = max(1, int(round(0.5 * self._fps / (clip_fps or 1.0))))
        if len(frames) > 1:  # sparse tube: a box covers half its own sampling gap on each side
            gaps = sorted(b - a for a, b in zip(frames, frames[1:]))
            tol = max(tol, int(math.ceil(gaps[len(gaps) // 2] / 2)))

        def box_at(fi):
            nearest = min(frames, key=lambda f: abs(f - fi))
            return track[nearest] if abs(nearest - fi) <= tol else None

        boxes_by_frame = {fi: [box_at(fi)] for fi in indices}
        note_fn = (lambda fi: "" if box_at(fi) is not None else "(no box)")
        if as_video:
            clip = render_clip(self._load, indices, self._fps, clip_fps, self._next_path("tube"),
                               boxes_by_frame=boxes_by_frame, box_labels=[label], source_size=self._size,
                               label=label, note_fn=note_fn)
            print(f"[Draw.tube] {clip} (tube frames {frames[0]}-{frames[-1]}, "
                  f"t={frames[0] / self._fps:.1f}-{frames[-1] / self._fps:.1f}s)")
            return clip
        tiles = [annotate_frame(self._load(fi), fi / self._fps, boxes_by_frame[fi], [label], note_fn(fi))
                 for fi in indices]
        cols = min(2, len(tiles))
        rows = math.ceil(len(tiles) / cols)
        tw, th = tiles[0].size
        grid = Image.new("RGB", (cols * tw, rows * th), (255, 255, 255))
        for k, tile in enumerate(tiles):
            grid.paste(tile, ((k % cols) * tw, (k // cols) * th))
        return VisualFeedback(grid, source="Draw.tube",
                              description=f"tube frames {frames[0]}-{frames[-1]} on {len(tiles)} frames {indices}")

    def __repr__(self) -> str:
        return "DrawTool(tube)"
