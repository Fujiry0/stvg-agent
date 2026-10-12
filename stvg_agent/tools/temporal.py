"""``tools.Temporal``: interval refinement and verification (not in SpatialClaw).

Two steps of the interval-labeling cascade of TimeLens2 (Generalist Video Temporal
Grounding with Multimodal LLM), which publishes no prompts, so the prompts here
are our own:
  - ``refine_boundary``: local change-point detection — the VLM sees a ±3 s
    neighborhood of a candidate boundary (dense frames, each labeled with its
    time) and predicts the transition point.
  - ``relevance``: text–video cosine similarity from Qwen3-VL-Embedding
    (TimeLens2 keeps clips with similarity >= 0.5) via the embedding server.
TimeLens2's multi-model consensus and merging of multi-interval labels are not
used: AnyGroundBench ST queries have a single interval, and the agent has one model.
"""

import json
import re
from typing import Dict, Optional

import time

from stvg_agent.kernel_types import timing
from stvg_agent.kernel_types.video_clip import annotate_frame
from stvg_agent.tools import server_wait

_REFINE_QUESTION = """These {n} images are consecutive frames of one video, {fps:g} frames per second, from t={t0:.2f}s to t={t1:.2f}s; each frame shows its time in the top-left.
Query: "{query}"
A candidate {side} of the event described by the query is at t={t:.2f}s, in the middle of this window.
Find the exact transition: {definition}
Reply with ONLY a JSON object:
{{"time": <seconds of that frame, or null>, "outside": null | "before_window" | "after_window" | "not_visible", "evidence": "<what changes between the frames just before and after>"}}
Use "outside" when the transition is not inside this window (it happens earlier or later) or the event is not visible here."""

_DEFINITIONS = {
    "start": "the time of the FIRST frame in which the event described by the query is happening "
             "(the frame before it must not show the event yet).",
    "end": "the time of the LAST frame in which the event described by the query is still happening "
           "(the frame after it must no longer show the event).",
}


class TemporalTool:
    def __init__(self, input_images, video_fps: float, llm_client, role_params, embed_url: Optional[str] = None,
                 video_path: Optional[str] = None, timeout_sec: float = 1100.0):
        self._ii = input_images
        self._fps = video_fps
        self._n = len(input_images)
        self._llm = llm_client
        self._role_params = role_params
        self._embed_url = embed_url.rstrip("/") if embed_url else None
        self._video_path = video_path
        self._timeout = timeout_sec
    def refine_boundary(self, t: float, side: str, query: str, window_sec: float = 3.0, fps: float = 5.0,
                        max_side: int = 512) -> Dict:
        """Refine a candidate event boundary at ``t`` seconds (``side`` = "start" or "end").

        Shows the VLM the ±``window_sec`` neighborhood at ``fps`` (dense still frames, time-labeled)
        and asks for the transition frame. Returns {"time", "frame", "outside", "evidence", "raw"};
        ``time`` is None when the VLM says the transition lies outside the window.
        """
        if side not in _DEFINITIONS:
            raise ValueError('side must be "start" or "end"')
        t0 = max(0.0, t - window_sec)
        t1 = min((self._n - 1) / self._fps, t + window_sec)
        n = int((t1 - t0) * fps) + 1
        if n > 64:
            raise ValueError(f"{n} frames exceed the 64-image limit; lower fps or window_sec")
        times = [t0 + i / fps for i in range(n)]
        frames = []
        for tt in times:
            img = self._ii[min(self._n - 1, int(round(tt * self._fps)))].image
            scale = max_side / max(img.size)
            if scale < 1:
                img = img.resize((round(img.width * scale), round(img.height * scale)))
            frames.append(annotate_frame(img, tt))
        question = _REFINE_QUESTION.format(n=n, fps=fps, t0=t0, t1=t1, query=query, side=side, t=t,
                                           definition=_DEFINITIONS[side])
        for _attempt in range(2):  # retry once on an empty / unparsable reply
            raw = self._llm.generate_vision_query(frames, question, system_prompt="", role_params=self._role_params,
                                                  tag="refine_boundary")
            parsed = _parse_json(raw) or {}
            if parsed.get("time") is not None or parsed.get("outside"):
                break
        time = parsed.get("time")
        try:
            time = None if time is None else float(time)
        except (TypeError, ValueError):
            time = None
        result = {"time": time, "frame": None if time is None else int(round(time * self._fps)),
                  "outside": parsed.get("outside"), "evidence": parsed.get("evidence"), "raw": raw}
        print(f"[Temporal.refine_boundary] {side} near t={t:.2f}s -> time={time} outside={result['outside']} "
              f"| {result['evidence']}")
        return result

    def relevance(self, start: float, end: float, query: str, fps: float = 1.0, max_frames: int = 64) -> float:
        """Cosine similarity between the video segment [start, end] s and ``query`` (Qwen3-VL-Embedding).

        TimeLens2 keeps a segment as relevant when the similarity is at least 0.5; here scores are
        best used to compare candidate segments.
        """
        if not self._embed_url:
            raise RuntimeError("relevance() needs the embedding server (embed_url is not configured)")
        t0 = time.time()
        resp = server_wait.post(self._embed_url, "/similarity", timeout=self._timeout, json={
            "video": self._video_path, "start": float(start), "end": float(end), "text": query,
            "fps": fps, "max_frames": max_frames})
        out = resp.json()
        timing.record("server/similarity", time.time() - t0, queued=out.get("queued_sec"))
        if resp.status_code != 200:
            raise RuntimeError(f"embedding server failed: {out.get('error')}")
        print(f"[Temporal.relevance] [{start:.1f}, {end:.1f}]s vs {query!r}: {out['similarity']:.3f}")
        return float(out["similarity"])

    def __repr__(self) -> str:
        return "TemporalTool(refine_boundary, relevance)"


def _parse_json(text: str) -> Optional[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(r"(\{.*\})", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
