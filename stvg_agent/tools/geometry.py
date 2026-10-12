"""``tools.Geometry``: 2D helpers for grounding.

``normalized_to_pixel`` is identical to SpatialClaw's ``GeometryUtils.normalized_to_pixel``
(tools/geometry_utils.py); the 3D helpers there are not needed for grounding, so
2D box utilities are provided instead.
"""


class GeometryUtils:
    @staticmethod
    def normalized_to_pixel(coords, width: int, height: int) -> list:
        """Convert 0-1000 normalized coordinates (x, y pairs) to pixel coordinates."""
        if any(v > 1000 for v in coords):
            print(
                f"[WARNING] normalized_to_pixel: coordinates {coords} exceed 1000 — "
                f"these may already be pixel coordinates. Normalized coords should be in 0-1000 scale. "
                f"If these are already pixels, use them directly without normalized_to_pixel()."
            )
        dims = [width, height] * (len(coords) // 2)
        if len(dims) < len(coords):
            dims.extend([width] * (len(coords) - len(dims)))
        return [v / 1000.0 * d for v, d in zip(coords, dims)]

    @staticmethod
    def box_iou(a, b) -> float:
        """IoU of two [x1, y1, x2, y2] boxes."""
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def box_center(box) -> list:
        """Center (x, y) of an [x1, y1, x2, y2] box."""
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    @staticmethod
    def frame_to_time(frame_index: int, fps: float) -> float:
        """Seconds from the start of the video for an absolute frame index."""
        return frame_index / fps

    @staticmethod
    def time_to_frame(seconds: float, fps: float) -> int:
        """Nearest absolute frame index for a time in seconds."""
        return int(round(seconds * fps))

    def __repr__(self) -> str:
        return "GeometryUtils(static methods: normalized_to_pixel, box_iou, box_center, frame_to_time, time_to_frame)"
