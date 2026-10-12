"""Coordinate system the agent's code works in (``AgentConfig.coord_format``).

Tools keep pixel boxes internally; this module converts at the boundary with the agent:
  "pixel"    pixel coordinates of the image (SpatialClaw's convention; default, no conversion at all)
  "norm1000" 0-1000 relative coordinates (Qwen3-VL's native grounding scale, same as vlm.locate)
  "norm1"    0-1 relative coordinates (the scale AnyGroundBench's ST prompt asks Qwen for)
"""

from typing import List, Sequence

SCALES = {"norm1000": 1000.0, "norm1": 1.0}


class CoordSpace:
    def __init__(self, fmt: str = "pixel"):
        if fmt not in ("pixel", "norm1000", "norm1"):
            raise ValueError(f"unknown coord_format {fmt!r}")
        self.fmt = fmt

    @property
    def is_pixel(self) -> bool:
        return self.fmt == "pixel"

    def to_unit(self, values: Sequence[float], W: float, H: float) -> List[float]:
        """Agent coordinates (x, y, x, y, ...) -> fractions of the image size (0-1)."""
        if self.is_pixel:
            return [float(v) / (W if i % 2 == 0 else H) for i, v in enumerate(values)]
        s = SCALES[self.fmt]
        return [float(v) / s for v in values]

    def to_pixel(self, values: Sequence[float], W: float, H: float) -> List[float]:
        """Agent coordinates -> pixels (identity in pixel mode)."""
        if self.is_pixel:
            return [float(v) for v in values]
        return [u * (W if i % 2 == 0 else H) for i, u in enumerate(self.to_unit(values, W, H))]

    def from_pixel(self, values: Sequence[float], W: float, H: float) -> List[float]:
        """Pixels -> agent coordinates (identity in pixel mode)."""
        if self.is_pixel:
            return [float(v) for v in values]
        s = SCALES[self.fmt]
        return [round(float(v) / (W if i % 2 == 0 else H) * s, 4 if s == 1.0 else 1) for i, v in enumerate(values)]

    @property
    def unit_name(self) -> str:
        return {"pixel": "px", "norm1000": "(0-1000)", "norm1": "(0-1)"}[self.fmt]


ACTIVE = CoordSpace("pixel")


def set_active(fmt: str) -> CoordSpace:
    global ACTIVE
    ACTIVE = CoordSpace(fmt)
    return ACTIVE


def active() -> CoordSpace:
    return ACTIVE
