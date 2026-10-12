"""The whole-video "when?" estimate.

One call over the whole video (at most 1 fps and ``max_frames`` frames, each frame labeled with its time) asks a VLM
during which interval the query holds. The estimate is given to the agent as a hint after the plan; it can be wrong.
"""

import json
import re
from typing import List, Optional

from stvg_agent.kernel_types.frame_image import VideoSource
from stvg_agent.kernel_types.video_clip import annotate_frame

WHEN_QUESTION = """These {n} images are frames of one video in time order, {fps:g} frames per second, covering 0-{dur:.1f}s. Each frame shows its time in the top-left.
Query: "{query}"
During which time interval does the content described by the query happen? Give the whole time it holds.
Answer with JSON only: {{"start": seconds, "end": seconds}}"""


def _parse_json(s: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", s or "", flags=re.S), re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def when(llm, role_params, video_path: str, query: str, max_frames: int = 64, max_side: int = 384,
         session_id: Optional[str] = None) -> Optional[List[float]]:
    """[start, end] seconds of the interval in which ``query`` holds, or None if the reply is unreadable."""
    src = VideoSource(video_path)
    fps_video = src.reader.get_avg_fps()
    n = len(src.reader)
    dur = (n - 1) / fps_video
    if dur <= 0:
        raise ValueError(f"video too short: {dur:.1f} s")
    fps = min(1.0, max_frames / dur)
    imgs = []
    for i in range(int(dur * fps) + 1):
        t = i / fps
        img = src.load(min(n - 1, int(round(t * fps_video))))
        scale = max_side / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)))
        imgs.append(annotate_frame(img, t))
    q = WHEN_QUESTION.format(n=len(imgs), fps=round(fps, 3), dur=dur, query=query)
    js = _parse_json(llm.generate_vision_query(imgs, q, system_prompt="", role_params=role_params,
                                               usage_session_id=session_id, tag="when")) or {}
    try:
        a, b = sorted([float(js["start"]), float(js["end"])])
    except (KeyError, TypeError, ValueError):
        return None
    a, b = max(0.0, a), min(dur, b)
    return [a, b] if b >= a else None
