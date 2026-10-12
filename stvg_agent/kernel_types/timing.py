"""Wall-time log of tool / VLM calls inside the kernel (for finding the bottleneck).

``Timed`` wraps an object so that every public method call is recorded as
``{"name": "SAM3.segment_video_by_text", "sec": ...}``; ``record`` adds entries
from inside a tool (e.g. the tool server's queue wait and processing time).
The workflow collects and clears the log after every step (``drain``).
"""

import functools
import json
import time
from typing import Any, Dict, List

_LOG: List[Dict[str, Any]] = []


def record(name: str, sec: float, **extra) -> None:
    _LOG.append({"name": name, "sec": round(float(sec), 3), **extra})


def drain_json() -> str:
    """Return the log as JSON and clear it (called by the workflow through the kernel)."""
    out = json.dumps(_LOG)
    _LOG.clear()
    return out


class Timed:
    """Proxy that times each public method call of ``obj`` under ``prefix.method``."""

    def __init__(self, obj, prefix: str):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if name.startswith("_") or not callable(attr):
            return attr

        @functools.wraps(attr)
        def call(*args, **kwargs):
            t0 = time.time()
            try:
                return attr(*args, **kwargs)
            finally:
                record(f"{self._prefix}.{name}", time.time() - t0)
        return call

    def __setattr__(self, name, value):
        setattr(self._obj, name, value)

    def __repr__(self) -> str:
        return repr(self._obj)

    def __dir__(self):
        return dir(self._obj)
