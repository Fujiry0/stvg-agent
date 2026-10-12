"""OpenAI-compatible LLM client for a vLLM server.

Synchronous port of SpatialClaw's ``spatial_agent/llm/client.py`` (LLMClient):
``generate`` for the main agent / planner and ``generate_vision_query`` for the
isolated VLM sessions. Role params map to request kwargs as in
``LLMClient._build_api_kwargs`` (vLLM-only params go in ``extra_body``).
Endpoint discovery and load balancing are dropped (single server).
"""

import base64
import io
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI
from PIL import Image

from stvg_agent.config import AgentConfig, LLMRoleParams
from stvg_agent.kernel_types import timing

_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, ConnectionError, OSError)


def image_to_base64_url(img: Image.Image, max_size: int = 1024) -> str:
    """Encode a PIL image as a base64 data URI, resizing if needed (as SpatialClaw)."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def video_parts(clip) -> List[Dict[str, Any]]:
    """A VideoClip as OpenAI content parts: its description, then the video (file URL for vLLM).

    vLLM must be started with ``--allowed-local-media-path`` covering the clip directory.
    """
    return [{"type": "text", "text": clip.describe()},
            {"type": "video_url", "video_url": {"url": "file://" + os.path.abspath(clip.path)}}]


class LLMClient:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key or "EMPTY",
                              timeout=config.vlm_query_timeout_sec)
        self._usage_lock = threading.Lock()
        self._session_usage: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    def _build_api_kwargs(self, params: LLMRoleParams) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"max_tokens": params.max_tokens, "temperature": params.temperature}
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.presence_penalty is not None:
            kwargs["presence_penalty"] = params.presence_penalty
        extra_body: Dict[str, Any] = {}
        if params.top_k is not None:
            extra_body["top_k"] = params.top_k
        if params.min_p is not None:
            extra_body["min_p"] = params.min_p
        if params.repetition_penalty is not None:
            extra_body["repetition_penalty"] = params.repetition_penalty
        if params.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": params.enable_thinking}
        if params.thinking_token_budget is not None and params.enable_thinking:
            extra_body["thinking_token_budget"] = params.thinking_token_budget
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _record_usage(self, usage_session_id: Optional[str], response) -> None:
        if not usage_session_id or response.usage is None:
            return
        u = response.usage
        details = getattr(u, "completion_tokens_details", None)
        with self._usage_lock:
            s = self._session_usage.setdefault(usage_session_id, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                "max_prompt_tokens": 0})
            s["calls"] += 1
            s["prompt_tokens"] += u.prompt_tokens or 0
            s["completion_tokens"] += u.completion_tokens or 0
            s["reasoning_tokens"] += (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            s["max_prompt_tokens"] = max(s["max_prompt_tokens"], u.prompt_tokens or 0)

    def pop_session_usage(self, usage_session_id: str) -> Dict[str, int]:
        with self._usage_lock:
            return self._session_usage.pop(usage_session_id, {})

    # ------------------------------------------------------------------
    def generate(self, messages: List[Dict[str, Any]], role_params: Optional[LLMRoleParams] = None,
                 max_retries: int = 5, usage_session_id: Optional[str] = None,
                 tag: str = "llm") -> Tuple[str, Optional[str]]:
        """Chat completion; returns ``(content, reasoning_content)``."""
        params = role_params or self.config.main_params
        kwargs = self._build_api_kwargs(params)
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                response = self._client.chat.completions.create(
                    model=self.config.llm_model, messages=messages, **kwargs)
                self._record_usage(usage_session_id, response)
                u = response.usage
                details = getattr(u, "completion_tokens_details", None) if u else None
                # one entry per LLM call (tag = caller) for the bottleneck log
                timing.record(f"llm:{tag}", time.time() - t0, prompt=getattr(u, "prompt_tokens", None),
                              completion=getattr(u, "completion_tokens", None),
                              reasoning=getattr(details, "reasoning_tokens", None) if details else None,
                              finish=response.choices[0].finish_reason, max_tokens=kwargs.get("max_tokens"))
                msg = response.choices[0].message
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                return msg.content or "", reasoning
            except _RETRYABLE as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_exc}") from last_exc

    def generate_vision_query(self, images: List[Image.Image], question: str, system_prompt: str,
                              role_params: Optional[LLMRoleParams] = None,
                              usage_session_id: Optional[str] = None, videos: Optional[List] = None,
                              tag: str = "vision") -> str:
        """Isolated VLM call: images (+ VideoClips) + question -> text answer."""
        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_to_base64_url(img)}} for img in images]
        for clip in videos or []:
            content += video_parts(clip)
        content.append({"type": "text", "text": question})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        answer, _ = self.generate(messages, role_params=role_params or self.config.vlm_params,
                                  usage_session_id=usage_session_id, tag=tag)
        return answer.strip()
