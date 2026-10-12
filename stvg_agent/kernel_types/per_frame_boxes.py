"""PerFrameBoxes: per-frame, per-object bounding boxes returned by ``tools.SAM3``.

Box-based counterpart of SpatialClaw's ``PerFrameMask`` (kernel_types/per_frame_types.py):
same access pattern (absolute frame indices, ``labels``, ``num_objects``,
``visualize(frame_idx)``), but it stores the tight box around each SAM3 mask
instead of the mask, because the grounding answer is a box per frame.
Boxes are ``[x1, y1, x2, y2]`` in **pixel** coordinates of ``InputImages``.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stvg_agent.kernel_types import coords
from stvg_agent.kernel_types.visual_feedback import VisualFeedback

_PALETTE = [(255, 64, 64), (64, 200, 255), (255, 210, 0), (80, 230, 120), (220, 90, 255), (255, 140, 40)]


class PerFrameBoxes:
    def __init__(self, frame_indices: List[int], boxes: np.ndarray, scores: np.ndarray, labels: List[str],
                 object_ids: List[int], frame_loader: Optional[Callable[[int], Image.Image]] = None,
                 frame_size: Optional[tuple] = None, frame_tolerance: int = 0):
        """``boxes``: (T, N_obj, 4) pixel xyxy, NaN where the object is absent; ``scores``: (T, N_obj)."""
        self._frame_indices = list(frame_indices)
        self.boxes = boxes
        self.scores = scores
        self.labels = labels
        self.object_ids = object_ids
        self._frame_loader = frame_loader
        self._index = {fi: i for i, fi in enumerate(self._frame_indices)}
        self._frame_width, self._frame_height = frame_size if frame_size else (None, None)
        self._tolerance = frame_tolerance  # tracked on every k-th frame: other frames use the nearest tracked one

    @property
    def frame_indices(self) -> List[int]:
        return list(self._frame_indices)

    @property
    def num_frames(self) -> int:
        return len(self._frame_indices)

    @property
    def num_objects(self) -> int:
        return len(self.labels)

    def get_by_frame_index(self, abs_frame_idx: int) -> int:
        if abs_frame_idx not in self._index and self._tolerance:
            near = [f for f in range(abs_frame_idx - self._tolerance, abs_frame_idx + self._tolerance + 1)
                    if f in self._index]
            if near:
                return self._index[min(near, key=lambda f: abs(f - abs_frame_idx))]
        if abs_frame_idx not in self._index:
            fi = self._frame_indices
            rng = f"{fi[0]}..{fi[-1]}" if fi else "empty"
            raise KeyError(f"Frame index {abs_frame_idx} not in this result (frame_indices: {rng}, "
                           f"{len(fi)} frames). Use absolute indices from .frame_indices.")
        return self._index[abs_frame_idx]

    def _resolve_object(self, object: Union[int, str]) -> int:
        if isinstance(object, str):
            if object not in self.labels:
                raise KeyError(f"Unknown label '{object}'. Labels: {self.labels}")
            return self.labels.index(object)
        if not 0 <= object < self.num_objects:
            raise IndexError(f"object index {object} out of range (num_objects={self.num_objects})")
        return object

    def _out(self, box) -> List[float]:
        """Internal pixel box -> the agent's coordinate system (identity in pixel mode)."""
        return coords.active().from_pixel(box, self._frame_width or 1.0, self._frame_height or 1.0)

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """(N_obj, 4) boxes at an absolute frame index (NaN rows = absent)."""
        rows = self.boxes[self.get_by_frame_index(frame_idx)]
        if coords.active().is_pixel:
            return rows
        return np.array([r if np.isnan(r).any() else self._out(r) for r in rows], dtype=float)

    def get_box(self, frame: int, object: Union[int, str] = 0) -> Optional[List[float]]:
        """[x1, y1, x2, y2] pixel box of one object at an absolute frame index, or None if absent."""
        box = self.boxes[self.get_by_frame_index(frame), self._resolve_object(object)]
        return None if np.isnan(box).any() else self._out(box)

    def get_score(self, frame: int, object: Union[int, str] = 0) -> float:
        return float(self.scores[self.get_by_frame_index(frame), self._resolve_object(object)])

    def get_track(self, object: Union[int, str] = 0) -> Dict[int, List[float]]:
        """{absolute frame index: [x1, y1, x2, y2]} over the frames where the object is present."""
        j = self._resolve_object(object)
        return {fi: self._out(self.boxes[i, j]) for i, fi in enumerate(self._frame_indices)
                if not np.isnan(self.boxes[i, j]).any()}

    def present_frames(self, object: Union[int, str] = 0) -> List[int]:
        return list(self.get_track(object))

    def visualize(self, frame_idx=None, background: Optional[Image.Image] = None, **kwargs) -> VisualFeedback:
        """Draw all objects' boxes (with labels) on the frame; returns VisualFeedback for show()."""
        if frame_idx is None and "frame" in kwargs:
            frame_idx = kwargs.pop("frame")
        if frame_idx is None:
            raise TypeError("visualize() requires a frame index: boxes.visualize(31) or visualize(frame=31)")
        if kwargs:
            raise TypeError(f"visualize() got unexpected keyword arguments: {list(kwargs)}")
        i = self.get_by_frame_index(frame_idx)
        if background is None and self._frame_loader is not None:
            background = self._frame_loader(frame_idx)
        if background is None:
            raise ValueError("No background frame available; pass background=InputImages[i]")
        img = (background.image if hasattr(background, "image") else background).convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, img.height // 24))
        except OSError:
            font = ImageFont.load_default()
        width = max(2, img.height // 150)
        present = []
        for j in range(self.num_objects):
            box = self.boxes[i, j]
            if np.isnan(box).any():
                continue
            color = _PALETTE[j % len(_PALETTE)]
            draw.rectangle([float(v) for v in box], outline=color, width=width)
            text = f"{j}:{self.labels[j]} {self.scores[i, j]:.2f}"
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            ty = max(0.0, float(box[1]) - (b - t) - 4)
            draw.rectangle([float(box[0]), ty, float(box[0]) + r - l + 6, ty + b - t + 4], fill=color)
            draw.text((float(box[0]) + 3, ty + 2 - t), text, fill=(0, 0, 0), font=font)
            present.append(self.labels[j])
        desc = f"frame {frame_idx}: {len(present)} object(s) {present}" if present else f"frame {frame_idx}: no objects"
        return VisualFeedback(img, source="SAM3 boxes", description=desc, frame_index=frame_idx)

    def _position(self, j: int) -> str:
        """Median box center / size of object j and a coarse left/center/right, top/middle/bottom word."""
        boxes = self.boxes[:, j]
        valid = boxes[~np.isnan(boxes).any(axis=1)]
        if len(valid) == 0:
            return ""
        cx = float(np.median((valid[:, 0] + valid[:, 2]) / 2))
        cy = float(np.median((valid[:, 1] + valid[:, 3]) / 2))
        w = float(np.median(valid[:, 2] - valid[:, 0]))
        h = float(np.median(valid[:, 3] - valid[:, 1]))
        W = self._frame_width or max(float(np.nanmax(boxes[:, 2])), 1.0)
        H = self._frame_height or max(float(np.nanmax(boxes[:, 3])), 1.0)
        horiz = "left" if cx < W / 3 else ("right" if cx > 2 * W / 3 else "center")
        vert = "top" if cy < H / 3 else ("bottom" if cy > 2 * H / 3 else "middle")
        c = coords.active()
        if c.is_pixel:
            return f", median center ({cx:.0f}, {cy:.0f}) px = {vert}-{horiz}, size {w:.0f}x{h:.0f}"
        ocx, ocy, ow, oh = c.from_pixel([cx, cy, w, h], W, H)
        return f", median center ({ocx:g}, {ocy:g}) {c.unit_name} = {vert}-{horiz}, size {ow:g}x{oh:g}"

    def __repr__(self) -> str:
        fi = self._frame_indices
        rng = f"{fi[0]}..{fi[-1]}" if fi else "empty"
        parts = []
        for j in range(self.num_objects):
            pf = self.present_frames(j)
            span = f"{pf[0]}..{pf[-1]}" if pf else "never"
            parts.append(f"{j}:{self.labels[j]} (present {len(pf)}/{self.num_frames} frames, {span}, "
                         f"mean score {np.nanmean(self.scores[:, j]) if len(fi) else 0:.2f}{self._position(j)})")
        return f"PerFrameBoxes(frames {rng} [{self.num_frames}], objects: {'; '.join(parts) or 'none'})"
