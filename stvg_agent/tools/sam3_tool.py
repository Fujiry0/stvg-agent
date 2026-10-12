"""``tools.SAM3``: client for ``tools_server`` with SpatialClaw's method names and semantics.

Follows SpatialClaw's ``spatial_agent/tools/sam3_tool.py`` (SAM3Tool): pixel
coordinates in, absolute frame indices out, ``start_frame``/``end_frame``
select the window (end exclusive, at most ``sam3_max_video_frames`` frames),
``prompt_frame_idx`` is local to the window, and flat point/box inputs are
auto-wrapped. Results are ``PerFrameBoxes`` (boxes around the SAM3 masks)
instead of ``PerFrameMask``.
"""

import base64
import io
import time
from typing import List, Optional

import numpy as np
import requests
from PIL import Image

from stvg_agent.kernel_types import coords, timing
from stvg_agent.tools import server_wait
from stvg_agent.kernel_types.frame_image import FrameImage
from stvg_agent.kernel_types.per_frame_boxes import PerFrameBoxes
from stvg_agent.kernel_types.visual_feedback import VisualFeedback


def _normalize_points_per_object(points_per_object):
    """Normalize LLM point formats to [[[x, y], ...], ...] (as SpatialClaw)."""
    if not isinstance(points_per_object, (list, tuple)) or len(points_per_object) == 0:
        return points_per_object
    if len(points_per_object) == 2 and all(isinstance(v, (int, float)) for v in points_per_object):
        result = [[list(points_per_object)]]
        print(f"[SAM3] Auto-wrapped flat [x, y] into [[[x, y]]]: {result}")
        return result
    if all(isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)
           for pt in points_per_object):
        result = [list(points_per_object)]
        print(f"[SAM3] Auto-wrapped [[x, y], ...] into [[[x, y], ...]]: {result}")
        return result
    return points_per_object


def _normalize_boxes(boxes):
    """Auto-wrap flat [x1, y1, x2, y2] into [[x1, y1, x2, y2]] (as SpatialClaw)."""
    if isinstance(boxes, (list, tuple)) and len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
        result = [list(boxes)]
        print(f"[SAM3] Auto-wrapped flat [x1,y1,x2,y2] into [[x1,y1,x2,y2]]: {result}")
        return result
    return boxes



MAX_EXIST_IMAGES = 128  # per is_object_exist call (one SAM3 image call each)

class SAM3Tool:
    """Usage::

        det = tools.SAM3.segment_image_by_text(InputImages[40], "lion")
        trk = tools.SAM3.segment_video_by_box([[x1, y1, x2, y2]], ["lion"], prompt_frame_idx=10,
                                              start_frame=300, end_frame=600)
    """

    def __init__(self, server_url: str, input_images, video_source: str, fps: float,
                 max_video_frames: int = 1000, max_retries: int = 3, timeout_sec: float = 1100.0,
                 frame_stride: int = 1, interpolate: bool = True):
        self._url = server_url.rstrip("/")
        self._input_images = input_images
        self._video_source = video_source
        self._fps = fps
        self._max_video_frames = max_video_frames
        self._max_retries = max_retries
        self._timeout = timeout_sec
        self._stride = max(1, int(frame_stride))
        self._interpolate = interpolate
        first = input_images[0]
        self._W, self._H = first.width, first.height
        self._total_video_frames = len(input_images)

    # ------------------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        last = None
        t0 = time.time()
        for attempt in range(max(1, self._max_retries)):
            try:
                resp = server_wait.post(self._url, path, payload, self._timeout)  # waits out a server outage
                out = resp.json()
                # server side: queued_sec = waiting for the other workers' calls, the rest = SAM3 itself
                timing.record(f"server{path}", time.time() - t0, queued=out.get("queued_sec"),
                              frames=(None if "frame_indices" not in out else len(out["frame_indices"])))
            except requests.ReadTimeout:  # the server accepted it but is busy: retrying only waits again
                raise RuntimeError(f"SAM3 {path.strip('/')} got no reply within {self._timeout:.0f}s "
                                   f"(the tool server is busy). Use a shorter frame window or fewer prompts.")
            except requests.RequestException as exc:
                last = exc
                time.sleep(min(2 ** attempt, 10))
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"SAM3 {path.strip('/')} failed: {out.get('error')}")
            return out
        raise RuntimeError(f"SAM3 server unreachable at {self._url}: {last}")

    def _frame_loader(self, frame_index: int):
        return self._input_images[frame_index].image

    def _image_payload(self, image):
        if isinstance(image, FrameImage):
            return {"video": self._video_source, "frame_index": image.frame_index}, image.frame_index, image.image
        pil = image.image if isinstance(image, VisualFeedback) else image
        if isinstance(pil, np.ndarray):
            pil = Image.fromarray(pil)
        if not isinstance(pil, Image.Image):
            raise TypeError(f"image must be a FrameImage (e.g. InputImages[i]) or PIL image, got {type(image).__name__}")
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="PNG")
        fi = getattr(image, "frame_index", None)
        return {"image_b64": base64.b64encode(buf.getvalue()).decode()}, (0 if fi is None else fi), pil

    def _single_frame_result(self, out: dict, frame_index: int, label: str, image: Image.Image) -> PerFrameBoxes:
        W, H = image.size
        objs = sorted(out["objects"], key=lambda o: -o["score"])
        boxes = np.full((1, len(objs), 4), np.nan)
        scores = np.zeros((1, len(objs)))
        for j, o in enumerate(objs):
            x0, y0, x1, y1 = o["box"]
            boxes[0, j] = [x0 * W, y0 * H, x1 * W, y1 * H]
            scores[0, j] = o["score"]
        return PerFrameBoxes([frame_index], boxes, scores, [label] * len(objs), [o["obj_id"] for o in objs],
                             frame_loader=lambda _fi: image, frame_size=(W, H))

    # ------------------------------------------------------------------
    # Image methods
    # ------------------------------------------------------------------
    def segment_image_by_text(self, image, prompt: str, label: Optional[str] = None) -> PerFrameBoxes:
        """All instances matching ``prompt`` in one image -> PerFrameBoxes (1 frame, N objects, best first)."""
        payload, fi, pil = self._image_payload(image)
        out = self._post("/segment_image", {**payload, "prompt": prompt})
        return self._single_frame_result(out, fi, label or prompt, pil)

    def segment_image_by_points(self, image, points, point_labels, label: str = "object") -> PerFrameBoxes:
        """Segment the object at pixel ``points`` ([[x, y], ...]; labels 1=fg, 0=bg)."""
        payload, fi, pil = self._image_payload(image)
        W, H = pil.size
        points = [coords.active().to_unit([x, y], W, H) for x, y in points]
        out = self._post("/segment_image", {**payload, "points": points, "point_labels": list(point_labels)})
        return self._single_frame_result(out, fi, label, pil)

    def segment_image_by_box(self, image, box, label: str = "object") -> PerFrameBoxes:
        """Segment the object inside pixel ``box`` [x1, y1, x2, y2]."""
        payload, fi, pil = self._image_payload(image)
        W, H = pil.size
        x1, y1, x2, y2 = coords.active().to_unit(_normalize_boxes(box)[0], W, H)
        xywh = [[x1, y1, x2 - x1, y2 - y1]]
        out = self._post("/segment_image", {**payload, "bounding_boxes": xywh, "bounding_box_labels": [1]})
        return self._single_frame_result(out, fi, label, pil)

    def is_object_exist(self, images, object_name: str) -> dict:
        """Whether ``object_name`` appears in each image, and how many instances."""
        images = images if isinstance(images, (list, tuple)) else [images]
        if len(images) > MAX_EXIST_IMAGES:
            # One SAM3 image call per frame: the whole InputImages (hundreds of frames) blocks the shared tool
            # server for minutes and makes every other call time out.
            raise ValueError(f"is_object_exist got {len(images)} images; pass at most {MAX_EXIST_IMAGES}, e.g. "
                             f"[InputImages[i] for i in range(0, len(InputImages), len(InputImages) // 16)].")
        counts = [self.segment_image_by_text(img, object_name).num_objects for img in images]
        exists = [c > 0 for c in counts]
        return {"exists": exists, "counts": counts,
                "summary": f"'{object_name}' found in {sum(exists)}/{len(images)} image(s)"}

    # ------------------------------------------------------------------
    # Video methods
    # ------------------------------------------------------------------
    def _validate_frame_range(self, start_frame: Optional[int], end_frame: Optional[int]) -> None:
        s = start_frame or 0
        e = end_frame if end_frame is not None else self._total_video_frames
        n = e - s
        if n <= 0:
            raise ValueError(f"end_frame ({end_frame}) must be greater than start_frame ({start_frame}).")
        if n > self._max_video_frames:
            if start_frame is not None and end_frame is not None:
                raise ValueError(f"Requested frame range ({start_frame}–{end_frame}) is {n} frames, which exceeds "
                                 f"the maximum of {self._max_video_frames}. Use a narrower start_frame/end_frame window.")
            raise ValueError(f"Video has {n} frames (indices {s}–{e}), which exceeds the maximum of "
                             f"{self._max_video_frames}. You MUST specify start_frame and end_frame to select a "
                             f"window of at most {self._max_video_frames} frames. "
                             f"Example: start_frame=0, end_frame={self._max_video_frames}")

    def _video_post(self, payload: dict) -> dict:
        """POST /segment_video; with frame_stride > 1 the window is shifted so the prompt frame is tracked."""
        window = (payload.get("start_frame") or 0,
                  self._total_video_frames if payload.get("end_frame") is None else int(payload["end_frame"]))
        stride = self._stride
        if stride > 1:
            local = int(payload.get("prompt_frame_idx") or 0)
            start = payload.get("start_frame") or 0
            payload = {**payload, "start_frame": start + local % stride,
                       "prompt_frame_idx": local // stride, "frame_stride": stride}
        out = self._post("/segment_video", payload)
        out["_window"] = window
        out["_stride"] = stride
        return out

    def _video_result(self, out: dict, labels: List[str]) -> PerFrameBoxes:
        objs = out["objects"]
        stride = out.get("_stride", self._stride)
        if stride > 1 and self._interpolate and "_window" in out:
            # Tracked on every stride-th frame: fill the frames in between by linear interpolation so the result
            # looks like dense tracking (every frame of the requested window, as SpatialClaw's SAM3 returns).
            frame_indices = list(range(*out["_window"]))
        else:
            frame_indices = out["frame_indices"]
        index = {fi: i for i, fi in enumerate(frame_indices)}
        boxes = np.full((len(frame_indices), len(objs), 4), np.nan)
        scores = np.zeros((len(frame_indices), len(objs)))
        for j, o in enumerate(objs):
            tracked = sorted((int(fi), [x0 * self._W, y0 * self._H, x1 * self._W, y1 * self._H], o["scores"][fi])
                             for fi, (x0, y0, x1, y1) in o["boxes"].items())
            for fi, box, sc in tracked:
                if fi in index:
                    boxes[index[fi], j], scores[index[fi], j] = box, sc
            if len(frame_indices) == len(out["frame_indices"]):
                continue
            for (fa, ba, sa), (fb, bb, sb) in zip(tracked, tracked[1:]):
                if fb - fa > stride:  # object missing at a tracked frame in between: leave the gap
                    continue
                for f in range(fa + 1, fb):
                    if f in index:
                        w = (f - fa) / (fb - fa)
                        boxes[index[f], j] = [(1 - w) * a + w * b for a, b in zip(ba, bb)]
                        scores[index[f], j] = (1 - w) * sa + w * sb
            if tracked:  # window edges: hold the nearest tracked box for less than one stride
                (f0, b0, s0), (f1, b1, s1) = tracked[0], tracked[-1]
                for f in range(f0 - stride + 1, f0):
                    if f in index:
                        boxes[index[f], j], scores[index[f], j] = b0, s0
                for f in range(f1 + 1, f1 + stride):
                    if f in index:
                        boxes[index[f], j], scores[index[f], j] = b1, s1
        obj_labels = [labels[min(j, len(labels) - 1)] if labels else f"object_{j}" for j in range(len(objs))]
        return PerFrameBoxes(frame_indices, boxes, scores, obj_labels, [o["obj_id"] for o in objs],
                             frame_loader=self._frame_loader, frame_size=(self._W, self._H),
                             frame_tolerance=0 if self._interpolate else stride // 2)

    def segment_video_by_text(self, prompts, labels=None, prompt_frame_idx: int = 0,
                              start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> PerFrameBoxes:
        """Track every instance matching the text prompts through [start_frame, end_frame)."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        self._validate_frame_range(start_frame, end_frame)
        out = self._video_post({"video": self._video_source, "start_frame": start_frame,
                                            "end_frame": end_frame, "prompt_frame_idx": prompt_frame_idx,
                                            "prompts": prompts})
        return self._video_result(out, list(labels) if labels else prompts)

    def segment_video_by_points(self, points_per_object, point_labels_per_object, labels, prompt_frame_idx: int = 0,
                                start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> PerFrameBoxes:
        """Track the objects clicked (pixel points) on the prompt frame."""
        points_per_object = _normalize_points_per_object(points_per_object)
        if point_labels_per_object and isinstance(point_labels_per_object[0], (int, float)):
            point_labels_per_object = [list(point_labels_per_object)]
        labels = [labels] if isinstance(labels, str) else list(labels)
        self._validate_frame_range(start_frame, end_frame)
        norm = [[coords.active().to_unit([x, y], self._W, self._H) for x, y in pts] for pts in points_per_object]
        out = self._video_post({"video": self._video_source, "start_frame": start_frame,
                                            "end_frame": end_frame, "prompt_frame_idx": prompt_frame_idx,
                                            "points_per_object": norm,
                                            "point_labels_per_object": point_labels_per_object})
        return self._video_result(out, labels)

    def segment_video_by_box(self, boxes, labels, prompt_frame_idx: int = 0,
                             start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> PerFrameBoxes:
        """Track the objects inside pixel ``boxes`` ([x1, y1, x2, y2] each) on the prompt frame."""
        boxes = _normalize_boxes(boxes)
        labels = [labels] if isinstance(labels, str) else list(labels)
        if len(boxes) != len(labels):
            raise ValueError("`boxes` and `labels` must have the same length.")
        self._validate_frame_range(start_frame, end_frame)
        unit = [coords.active().to_unit(b, self._W, self._H) for b in boxes]
        norm = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in unit]
        out = self._video_post({"video": self._video_source, "start_frame": start_frame,
                                            "end_frame": end_frame, "prompt_frame_idx": prompt_frame_idx,
                                            "boxes_per_object": norm})
        return self._video_result(out, labels)

    def __repr__(self) -> str:
        return ("SAM3Tool(segment_image_by_text, segment_image_by_points, segment_image_by_box, "
                "segment_video_by_text, segment_video_by_points, segment_video_by_box, is_object_exist)")
