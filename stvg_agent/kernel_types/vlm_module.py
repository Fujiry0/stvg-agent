"""VLMModule: ``vlm.locate()`` and ``vlm.ask_with_thinking()`` inside the kernel.

Synchronous port of SpatialClaw's ``kernel_types/vlm_module.py``: each call is an
isolated VLM session (only the given images + question, no conversation
history); ``locate`` prepends the presence-check preamble and allows 8 images,
``ask_with_thinking`` allows 64. Queries are printed and recorded so the
feedback shows them. Extension: ``ask_with_thinking`` also accepts VideoClips
(sent as video input), which temporal questions need.
"""

import os
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from stvg_agent.kernel_types.frame_image import FrameImage
from stvg_agent.kernel_types.video_clip import VideoClip
from stvg_agent.kernel_types.visual_feedback import VisualFeedback

_LOCATE_MAX_IMAGES = 8
_THINKING_MAX_IMAGES = 64

_LOCATE_VERIFY_PREAMBLE = (
    "Before answering, first check whether what the question describes is "
    "clearly visible in the provided image(s). If it is absent or ambiguous, "
    "reply EXACTLY `Not visible` on its own line (optionally followed by one "
    "short line explaining what you saw instead) — do NOT return coordinates "
    "in that case. Only if the requested target is clearly and unambiguously "
    "present, answer the request below.\n\n"
    "Request: "
)


class VLMModule:
    """Injected into the kernel as ``vlm``."""

    def __init__(self, llm_client, locate_system_prompt: str, thinking_system_prompt: str, session_dir: str,
                 session_id: str = "", locate_role_params=None, thinking_role_params=None):
        self._llm_client = llm_client
        self._locate_prompt = locate_system_prompt
        self._thinking_prompt = thinking_system_prompt
        self._session_dir = session_dir
        self._session_id = session_id
        self._locate_role_params = locate_role_params
        self._thinking_role_params = thinking_role_params
        self._queries: List[Dict[str, Any]] = []
        self._query_counter = 0

    def locate(self, visual_input: Union[VisualFeedback, Image.Image, List], question: str) -> str:
        """Grounding VLM: short answer (typically 0-1000 coordinates) or ``Not visible``."""
        return self._dispatch(visual_input, _LOCATE_VERIFY_PREAMBLE + question, "locate",
                              self._locate_prompt, self._locate_role_params, _LOCATE_MAX_IMAGES)

    def ask_with_thinking(self, visual_input: Union[VisualFeedback, Image.Image, VideoClip, List], question: str) -> str:
        """Thinking VLM: text answer about the provided frames (up to 64) and/or VideoClips."""
        return self._dispatch(visual_input, question, "thinking",
                              self._thinking_prompt, self._thinking_role_params, _THINKING_MAX_IMAGES)

    def _dispatch(self, visual_input, question, query_type, system_prompt, role_params, max_images) -> str:
        self._query_counter += 1
        query_id = f"vlm_q_{query_type}_{self._query_counter:04d}"
        items = visual_input if isinstance(visual_input, list) else [visual_input]
        videos = [v for v in items if isinstance(v, VideoClip)]
        if videos and query_type == "locate":
            raise TypeError("vlm.locate takes images only; pass a frame such as InputImages[i]. "
                            "Use vlm.ask_with_thinking for video clips.")
        rest = [v for v in items if not isinstance(v, VideoClip)]
        images = self._resolve_images(rest, label=query_type == "thinking") if rest else []
        truncated = len(images) > max_images
        images = images[:max_images]
        self._save_query_images(query_id, query_type, images)
        try:
            answer = self._llm_client.generate_vision_query(
                images=images, question=question, system_prompt=system_prompt,
                role_params=role_params, usage_session_id=self._session_id, videos=videos,
                tag=f"vlm.{query_type}")
        except Exception as exc:
            answer = f"[VLM Error] {exc}"
        source = str(visual_input) if not isinstance(visual_input, list) else f"list of {len(visual_input)} images"
        self._queries.append({"query_id": query_id, "query_type": query_type, "question": question,
                              "answer": answer, "num_images": len(images), "num_videos": len(videos),
                              "videos": [v.path for v in videos], "source": source})
        if truncated:
            print(f"[vlm.{query_type}] Showing first {max_images} images; remaining were truncated.")
        print(f"[VLM Q | {query_type}] {question}")
        print(f"[VLM A | {query_type}] {answer}")
        return answer

    def _resolve_images(self, visual_input, label: bool = False) -> List[Image.Image]:
        if isinstance(visual_input, list) or (
            hasattr(visual_input, "__iter__")
            and not isinstance(visual_input, (str, np.ndarray, Image.Image, VisualFeedback, FrameImage))
        ):
            return [self._to_pil(v, label) for v in visual_input]
        return [self._to_pil(visual_input, label)]

    @staticmethod
    def _to_pil(obj: Any, label: bool = False) -> Image.Image:
        if isinstance(obj, FrameImage):
            from stvg_agent.kernel_types import frame_image
            return frame_image.labeled_image(obj) if label else obj.image
        if isinstance(obj, VisualFeedback):
            return obj.image
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
            return Image.fromarray(obj)
        raise TypeError(f"Cannot convert {type(obj).__name__} to PIL.Image. "
                        f"Accepted: FrameImage, VisualFeedback, PIL.Image, uint8 ndarray.")

    def _save_query_images(self, query_id: str, query_type: str, images: List[Image.Image]) -> None:
        img_dir = os.path.join(self._session_dir, "vlm_queries", query_type)
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(img_dir, f"{query_id}_img_{i}.png"))

    def get_and_clear_queries(self) -> List[Dict[str, Any]]:
        queries = self._queries.copy()
        self._queries.clear()
        return queries
