"""Agent loop for one sample: init -> plan -> (LLM step -> execute -> feedback)* -> answer.

Sequential port of SpatialClaw's LangGraph workflow (``spatial_agent/workflow.py`` and
``nodes/{init,plan,llm_step,execute,feedback}_node.py``, ``nodes/router.py``):
  - init: persistent Jupyter kernel (reused across samples via ``%reset``), stdlib init,
    per-sample injection (InputImages, Metadata, feedback/show, vlm, tools, ReturnAnswer),
    first user message = key frames + mapping + instruction.
  - plan: isolated blind planner; its plan + the transition prompt are appended.
  - llm_step: response must parse into Purpose / Reasoning / Next Goal / Code.
  - execute: AST safety check, run the cell, collect ``show()`` images from stdout markers.
  - feedback: variable diff, rollback of errored steps (condense_errors), ReturnAnswer
    sentinel, show() budget, last-step warning; errored assistant turns are condensed.
  - router / force_terminate: stop on answer, max_steps or max_failures; fall back to the
    last submitted answer, else a single VLM call over the key frames.
Reflection (off by default in SpatialClaw) is not implemented.
"""

import ast
import collections
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from PIL import Image

from stvg_agent.config import AgentConfig
from stvg_agent.kernel.feedback_collector import (
    FeedbackCollector,
    _compact_error,
    _extract_error_line_number,
    _extract_error_snippet,
    _search_code_for_pattern,
    _truncate_code_at_error,
)
from stvg_agent.kernel.manager import JupyterKernelManager
from stvg_agent.kernel.safety import SecuritySandbox
from stvg_agent.kernel.variable_tracker import VariableTracker
from stvg_agent.kernel_types import timing as host_timing
from stvg_agent.kernel_types.frame_image import VideoSource
from stvg_agent.kernel_types.video_clip import render_clip, sample_clip_frames
from stvg_agent.llm.client import LLMClient, image_to_base64_url, video_parts
from stvg_agent.llm.planning_prompt import build_planning_system_prompt, build_planning_user_message
from stvg_agent.llm.response_schema import LLMResponseValidator
from stvg_agent.llm.system_prompt import build_system_prompt, key_frame_mapping_text
from stvg_agent import temporal_select

TRANSITION_PROMPT = (
    "Good plan. Now execute it step by step. Each response MUST use the required markdown format "
    "(**Purpose**, **Reasoning**, **Next Goal**, **Code** with a ```python fenced block). "
    "Start with step 1 of your plan."
)

INSTRUCTION = (
    "Spatio-temporal video grounding.\n"
    "Given the query \"{query}\", identify the spatio-temporal instances where the described visual content "
    "occurs in the video. For each timestamp, detect and localize the object(s). Only include timestamps where "
    "the visual content explicitly matches the query. If the content is not present in a frame, skip that "
    "timestamp. Ensure temporal continuity for dynamic actions (e.g., tracking a moving object). "
    "Submit the result with ReturnAnswer(...)."
)

# The single whole-video "when?" estimate given to the agent as evidence; the agent decides what to do with it.
ANCHOR_HINT = (
    "[Whole-video estimate] A separate single look at the whole video (1 frame per second, each frame labeled with its "
    "time) estimated that the query holds at about {s:.0f}-{e:.0f} s. It is one estimate and can be wrong. A single "
    "look at the whole video is often better at finding WHICH moment than a search that starts from one narrow guess, "
    "so check this interval (e.g. a clip of it) before settling on another one; if you choose a different interval, "
    "have clear visual evidence for it."
)

FALLBACK_PROMPT = (
    "The images are {n} frames of a video, at these times (seconds): {times}.\n"
    "Query: \"{query}\"\n"
    "Find when the query holds and where the object it refers to is. Think step-by-step, then output "
    "the answer as a JSON array inside a ```json block: "
    "[{{\"timestamp\": <seconds>, \"bbox_2d\": [x1, y1, x2, y2]}}, ...] with coordinates in 0-1000 "
    "normalized scale, one entry per whole second of the interval."
)

_SHOW_RE = re.compile(r"^\[SHOW:(\{.*\})\]$", re.MULTILINE)


# ----------------------------------------------------------------------
# show() plumbing (execute_node / feedback_node helpers)
# ----------------------------------------------------------------------
def _parse_show_markers(stdout: str) -> List[Dict[str, Any]]:
    results = []
    for m in _SHOW_RE.finditer(stdout):
        try:
            meta = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        images = []
        for p in meta.get("paths", []):
            try:
                images.append(Image.open(p).copy())
            except Exception:
                pass
        videos = meta.get("videos", [])
        if images or videos:
            results.append({"show_id": meta.get("show_id", ""), "label": meta.get("label", ""),
                            "images": images, "_paths": meta.get("paths", []), "videos": videos})
    return results


class _ClipRef:
    """Minimal stand-in for VideoClip on the host side (path + description from the marker)."""

    def __init__(self, path: str, describe: str):
        self.path, self._describe = path, describe

    def describe(self) -> str:
        return self._describe


def _extract_show_labels(code: str) -> List[str]:
    labels = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return labels
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_show = (isinstance(func, ast.Name) and func.id == "show") or (
            isinstance(func, ast.Attribute) and func.attr == "show"
            and isinstance(func.value, ast.Name) and func.value.id == "feedback")
        if is_show and node.args:
            labels.append(ast.get_source_segment(code, node.args[0]) or "")
    return labels


def _add_label(image: Image.Image, label: str, padding: int = 32) -> Image.Image:
    """White band above the image with a word-wrapped centered label (SpatialClaw feedback_node)."""
    from PIL import ImageDraw, ImageFont

    w, h = image.size
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    words, lines = label.split(), []
    if words:
        current = words[0]
        for word in words[1:]:
            cand = f"{current} {word}"
            box = draw.textbbox((0, 0), cand, font=font)
            if box[2] - box[0] <= w - 16:
                current = cand
            else:
                lines.append(current)
                current = word
        lines.append(current)
    else:
        lines = [label]
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + 4
    pad = max(padding, line_h * len(lines) + 8)
    out = Image.new("RGB", (w, h + pad), (255, 255, 255))
    out.paste(image.convert("RGB"), (0, pad))
    draw = ImageDraw.Draw(out)
    y = (pad - line_h * len(lines)) // 2
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        draw.text(((w - (box[2] - box[0])) // 2, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    return out


def _multimodal_content(text: str, show_images: List[Dict[str, Any]], add_labels: bool) -> List[Dict[str, Any]]:
    parts = [{"type": "text", "text": text}]
    for entry in show_images:
        base = f"Visualization of {entry['label']}" if entry.get("label") else "Visualization"
        imgs = entry["images"]
        for i, img in enumerate(imgs):
            if add_labels:
                img = _add_label(img, f"{base} ({i + 1}/{len(imgs)})" if len(imgs) > 1 else base)
            parts.append({"type": "image_url", "image_url": {"url": image_to_base64_url(img)}})
        for video in entry.get("videos", []):
            parts += video_parts(_ClipRef(video["path"], video["describe"]))
    return parts


def _trim_images(show_images: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    kept = []
    for entry in show_images:
        if budget <= 0:
            break
        imgs = entry["images"]
        if len(imgs) <= budget:
            kept.append(entry)
            budget -= len(imgs)
        else:
            kept.append({**entry, "images": imgs[:budget], "_paths": entry["_paths"][:budget]})
            budget = 0
    return kept


def _condensed_ai_message(parsed: Dict[str, str], code: str, error: str) -> str:
    truncated = _truncate_code_at_error(code, error)
    if truncated is not None:
        code_section = truncated
    else:
        snippet = _extract_error_snippet(code, error)
        compact = _compact_error(error)
        code_section = f"# [ERROR] {compact}\n{snippet}" if snippet else f"# [ERROR] {compact}"
    return (f"**Purpose**: {parsed.get('purpose', 'unknown')}\n"
            f"**Reasoning**: [errored — condensed]\n"
            f"**Next Goal**: [errored — condensed]\n"
            f"**Code**:\n```python\n{code_section}\n```")


# ----------------------------------------------------------------------
# Kernel init / injection (init_node)
# ----------------------------------------------------------------------
INIT_CODE = """
# === STVG Agent Kernel Initialization ===
import numpy as np
import math
import collections
import itertools
import functools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import scipy
    from scipy import ndimage, spatial, signal, optimize
except ImportError:
    pass
import stvg_agent.kernel_types.frame_image
import stvg_agent.kernel_types.per_frame_boxes
import stvg_agent.tools
print("[Kernel] stdlib init complete.")
"""

SHOW_INJECTION = """
show = feedback.show
import matplotlib.pyplot as _plt_module
def _patched_plt_show(*args, **kwargs):
    import io as _io
    from PIL import Image as _PILImage
    try:
        fig_nums = _plt_module.get_fignums()
        if not fig_nums:
            print("[plt.show patch] No open figures.")
            return
        for fig in [_plt_module.figure(n) for n in fig_nums]:
            title = ""
            for ax in fig.get_axes():
                if ax.get_title():
                    title = ax.get_title()
                    break
            buf = _io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = _PILImage.open(buf).copy()
            buf.close()
            feedback.show(img, label=title or "matplotlib figure")
        _plt_module.close('all')
    except Exception as _e:
        print(f"[plt.show patch] Error: {_e}")
_plt_module.show = _patched_plt_show
"""


def build_injection_code(config: AgentConfig, video_path: str, session_dir: str, session_id: str,
                         meta: Dict[str, Any], input_indices: List[int]) -> str:
    payload = json.dumps({"config": config.to_dict(), "video_path": video_path, "session_dir": session_dir,
                          "session_id": session_id, "meta": meta, "input_indices": input_indices})
    return f"""
import json as _json
_p = _json.loads({payload!r})
from stvg_agent.config import AgentConfig as _AC, LLMRoleParams as _RP
_c = _p["config"]
for _k in [k for k in _c if k.endswith("_params")]:
    _c[_k] = _RP(**_c[_k])
_config = _AC(**_c)
from stvg_agent.kernel_types.frame_image import VideoSource as _VS, InputImages as _II
from stvg_agent.kernel_types.metadata import Metadata as _MD
from stvg_agent.kernel_types.feedback_module import FeedbackModule as _FM
from stvg_agent.kernel_types.vlm_module import VLMModule as _VM
from stvg_agent.kernel_types.return_answer import ReturnAnswer
from stvg_agent.llm.client import LLMClient as _LC
from stvg_agent.llm.vision_prompt import LOCATE_SYSTEM_PROMPT as _LSP, THINKING_SYSTEM_PROMPT as _TSP
from stvg_agent.tools import ToolsModule as _TM
from stvg_agent.tools.sam3_tool import SAM3Tool as _ST
from stvg_agent.tools.video_tools import VideoTool as _VT, DrawTool as _DT
from stvg_agent.tools.temporal import TemporalTool as _TT
_m = _p["meta"]
_src = _VS(_p["video_path"], _config.image_max_long_edge)
InputImages = _II.from_video(_src, _p["input_indices"])          # what the agent sees (video_max_fps)
import stvg_agent.kernel_types.frame_image as _fimod
_fimod.LABEL_FPS = _m["fps"] if _config.time_label_stills else None
_all_frames = _II.from_video(_src, list(range(_m["total_frames"])))  # every native frame, for the tools
Metadata = _MD(is_video=True, fps=_m["fps"], total_frames=_m["total_frames"], duration_sec=_m["duration_sec"],
               num_images=_m["num_images"], video_source=_p["video_path"], videos=None, num_videos=1,
               width=_m["width"], height=_m["height"])
feedback = _FM(session_dir=_p["session_dir"], enable_sighted_feedback=_config.enable_sighted_feedback,
               session_id=_p["session_id"])
_llm = _LC(_config)
from stvg_agent.kernel_types.timing import Timed as _Timed
vlm = _Timed(_VM(_llm, _LSP, _TSP, session_dir=_p["session_dir"], session_id=_p["session_id"],
                 locate_role_params=_config.vlm_grounding_params, thinking_role_params=_config.vlm_params), "vlm")
import os as _os
_clip_dir = _os.path.join(_p["session_dir"], "clips")
tools = _TM(_ST(_config.sam3_url, _all_frames, _p["video_path"], _m["fps"],
                max_video_frames=_config.sam3_max_video_frames, max_retries=_config.gpu_tool_max_retries,
                frame_stride=_config.sam3_frame_stride, interpolate=_config.sam3_interpolate),
            _VT(_all_frames, _m["fps"], _clip_dir), _DT(_all_frames, _m["fps"], _clip_dir),
            _TT(_all_frames, _m["fps"], _llm, _config.vlm_params, embed_url=_config.embed_url,
                video_path=_p["video_path"]))
for _name in tools.enabled:
    globals()[_name] = getattr(tools, _name)
ReturnAnswer.fps, ReturnAnswer.width, ReturnAnswer.height = _m["fps"], _m["width"], _m["height"]
ReturnAnswer.num_frames = _m["total_frames"]
ReturnAnswer.num_inputs, ReturnAnswer.index_warning = len(InputImages), _config.index_warning
ReturnAnswer.index_warned = False
{SHOW_INJECTION}
del _Timed, _json, _p, _c, _k, _AC, _RP, _VS, _II, _src, _all_frames, _MD, _FM, _VM, _LC, _LSP, _TSP, _TM, _ST, _VT, _DT, _TT, _llm, _m, _os, _clip_dir, _name
print("[Kernel] Injected: feedback, vlm, InputImages, Metadata, tools, ReturnAnswer; tools:", tools.enabled)
"""


# ----------------------------------------------------------------------
# Workflow
# ----------------------------------------------------------------------
class STVGAgentWorkflow:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_client = LLMClient(config)
        self.km = JupyterKernelManager(timeout_sec=config.timeout_sec)

    def shutdown(self) -> None:
        self.km.shutdown()

    def _prepare_kernel(self) -> None:
        if self.config.fresh_kernel_per_query and self.km.is_running:
            # %reset does not give memory back: a reused kernel keeps growing and several workers can fill the
            # machine's RAM. Start each query afresh.
            self.km.shutdown()
        if self.km.is_running and self.km._init_code:
            try:
                self.km.reset_namespace()
                return
            except Exception as exc:
                print(f"[workflow] reset_namespace failed ({exc}); restarting kernel")
                self.km.shutdown()
        if not self.km.is_running:
            self.km.start()
        self.km.set_init_code(INIT_CODE)
        result = self.km.execute(INIT_CODE, timeout=120)
        if result.error:
            raise RuntimeError(f"Kernel initialization failed: {result.error}")

    def _answer_check(self, answer: Dict[str, Any], done: set) -> Optional[str]:
        """Check on a submitted tube: a tube of at most 2 s is questioned once. Returns a message or None."""
        secs = sorted(int(s) for s in (answer.get("raw_value") or {}))
        if not secs:
            return None
        a, b = secs[0], secs[-1]
        if self.config.confirm_short and "short" not in done and b - a + 1 <= 2:
            done.add("short")
            return (f"[CHECK — not rejected for good] Your tube covers only {b - a + 1} s ({a}-{b} s). Check whether "
                    f"the query's content also holds in the seconds just before {a} s and after {b} s (e.g. a clip "
                    f"around each end). If it does, extend the tube to the whole time it holds; if the short interval "
                    f"is right, submit the same answer again.")
        return None

    def _drain_tool_timings(self) -> List[Dict[str, Any]]:
        """Tool / VLM call times logged in the kernel during the last cell (stvg_agent.kernel_types.timing)."""
        r = self.km.execute("import stvg_agent.kernel_types.timing as _tm\nprint(_tm.drain_json())\ndel _tm", timeout=10)
        try:
            return json.loads(r.stdout.strip().split("\n")[-1])
        except (json.JSONDecodeError, IndexError):
            return []

    def _may_reject(self, step_count: int, total_answer_attempts: int) -> bool:
        """SpatialClaw's guard: never reject when the budget is too low or after too many attempts."""
        cfg = self.config
        return (cfg.max_steps - step_count > cfg.min_budget_for_answer_reject
                and total_answer_attempts < cfg.max_total_answer_attempts)

    # ------------------------------------------------------------------
    def run(self, video_path: str, query: str, session_id: str, session_dir: str,
            query_id: Optional[str] = None) -> Dict[str, Any]:
        cfg = self.config
        t_start = time.time()
        os.makedirs(session_dir, exist_ok=True)
        instruction = INSTRUCTION.format(query=query)

        # --- video metadata and key frames ---
        source = VideoSource(video_path, cfg.image_max_long_edge)
        total = len(source.reader)
        fps = float(source.reader.get_avg_fps())
        first = source.load(0)
        meta = {"fps": round(fps, 4), "total_frames": total, "duration_sec": total / fps, "num_images": total,
                "width": first.width, "height": first.height}
        # InputImages: the video sampled at video_max_fps (SpatialClaw's extract_video_frames: frame i of the
        # output is native frame round(i * fps / out_fps)); tools keep access to every native frame.
        if cfg.video_max_fps and fps > cfg.video_max_fps:
            step = fps / cfg.video_max_fps
            input_indices = []
            while int(round(len(input_indices) * step)) < total:
                input_indices.append(int(round(len(input_indices) * step)))
        else:
            input_indices = list(range(total))
        num_inputs = len(input_indices)
        meta["num_images"] = num_inputs
        # Key frames: a subset of InputImages (list positions kf_li, absolute video frames kf_list).
        n = min(cfg.num_key_frames, num_inputs)
        if n >= num_inputs:
            kf_li = list(range(num_inputs))
        else:
            kf_li = [round(i * (num_inputs - 1) / max(1, n - 1)) for i in range(n)]
        kf_list = [input_indices[j] for j in kf_li]
        key_frames = [source.load(i) for i in kf_list]
        initial_clip = None
        if cfg.initial_video:
            indices, clip_fps = sample_clip_frames(0, total - 1, fps, cfg.initial_video_fps,
                                                   cfg.initial_video_max_frames)
            initial_clip = render_clip(source.load, indices, fps, clip_fps,
                                       os.path.join(session_dir, "clips", "full_video.mp4"), label="whole video")

        # --- init_node ---
        self._prepare_kernel()
        injection = build_injection_code(cfg, video_path, session_dir, session_id, meta, input_indices)
        self.km.set_injection_code(injection)
        result = self.km.execute(injection, timeout=300)
        if result.error:
            raise RuntimeError(f"Object injection failed: {result.error}")

        system_prompt = build_system_prompt(cfg, meta, kf_list, kf_li)
        if kf_list:
            header = (f"Here are {len(key_frames)} key frames — a visual overview subset of InputImages "
                      f"({num_inputs} total frames).\nKey frame mapping (context position → variable[index] → video frame):\n"
                      + key_frame_mapping_text(kf_li, kf_list, fps))
        else:
            header = f"Here is the whole video as a clip ({num_inputs} frames in InputImages)."
        first_content = [{"type": "text", "text": header}]
        first_content += [{"type": "image_url", "image_url": {"url": image_to_base64_url(img)}} for img in key_frames]
        if initial_clip is not None:
            first_content += video_parts(initial_clip)
        first_content.append({"type": "text", "text": instruction})
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt},
                                          {"role": "user", "content": first_content}]
        variable_registry = self.km.get_variables()
        log: Dict[str, Any] = {"session_id": session_id, "query": query, "video_path": video_path, "meta": meta,
                               "instruction": instruction, "key_frames": kf_list, "system_prompt": system_prompt,
                               "plan": None, "steps": []}

        # --- plan_node ---
        timing: Dict[str, float] = collections.defaultdict(float)  # wall seconds per phase (bottleneck log)
        host_timing.drain_json()  # LLM calls made in this process (main, plan, ...) are logged there too
        timing["init"] = time.time() - t_start
        # The whole-video "when?" estimate runs in the background while the agent plans.
        pool = anchor_future = None
        if cfg.anchor_hint:
            pool = ThreadPoolExecutor(1)
            anchor_future = pool.submit(temporal_select.when, self.llm_client, cfg.vlm_params, video_path, query,
                                        max_frames=cfg.when_max_frames, session_id=session_id)
        if cfg.enable_planning and cfg.max_steps > 1:
            t_plan = time.time()
            try:
                raw_plan, plan_reasoning = self.llm_client.generate(tag="plan", messages=
                    [{"role": "system", "content": build_planning_system_prompt(cfg, meta)},
                     build_planning_user_message(
                         instruction, kf_list,
                         key_frames=key_frames if cfg.planning_sighted else None,
                         video=initial_clip if cfg.planning_sighted else None,
                         mapping_text=key_frame_mapping_text(kf_li, kf_list, fps),
                         num_total_images=num_inputs)],
                    role_params=cfg.planning_params, usage_session_id=session_id)
                plan_text = LLMResponseValidator._strip_thinking(raw_plan)
                messages += [{"role": "user", "content": f"Here is your execution plan:\n\n{plan_text}"},
                             {"role": "user", "content": TRANSITION_PROMPT}]
                log["plan"] = plan_text
                log["plan_reasoning"] = plan_reasoning
            except Exception as exc:
                log["plan_error"] = f"{type(exc).__name__}: {exc}"
            timing["plan"] += time.time() - t_plan

        # --- anchor hint (extension): tell the agent the whole-video estimate; nothing is overridden ---
        if cfg.anchor_hint:
            t_a = time.time()
            try:
                anchor = anchor_future.result()
                log["anchor_hint"] = anchor
                if anchor:
                    messages.append({"role": "user", "content": ANCHOR_HINT.format(s=anchor[0], e=anchor[1])})
            except Exception as exc:
                log["anchor_hint_error"] = f"{type(exc).__name__}: {exc}"
            timing["anchor"] += time.time() - t_a

        step_count = failure_count = total_show = total_tool_calls = total_videos = 0
        final_answer: Optional[Dict[str, Any]] = None
        last_submitted: Optional[Dict[str, Any]] = None
        total_answer_attempts = 0
        checks_done: set = set()      # answer checks already applied (each once)
        termination = None

        while True:
            # --- router ---
            if final_answer is not None:
                termination = "completed"
                break
            if step_count >= cfg.max_steps:
                termination = "max_steps"
                break
            if failure_count >= cfg.max_failures:
                termination = "max_failures"
                break

            step_log: Dict[str, Any] = {"step": step_count}
            # --- llm_step_node ---
            parsed = None
            t_llm = time.time()
            try:
                raw, reasoning = self.llm_client.generate(messages, role_params=cfg.main_params, tag="main",
                                                          usage_session_id=session_id)
                step_log["llm_sec"] = round(time.time() - t_llm, 2)
                timing["llm_main"] += time.time() - t_llm
                step_log.update(raw_response=raw, reasoning=reasoning)
                try:
                    resp = LLMResponseValidator.validate(raw)
                    parsed = {"purpose": resp.purpose, "reasoning": resp.reasoning,
                              "next_goal": resp.next_goal, "code": resp.code}
                    messages.append({"role": "assistant", "content": (
                        f"**Purpose**: {resp.purpose}\n**Reasoning**: {resp.reasoning}\n"
                        f"**Next Goal**: {resp.next_goal}\n**Code**:\n```python\n{resp.code}\n```")})
                    failure_count = 0
                    skip_reason = None
                except ValueError as exc:
                    messages += [{"role": "assistant", "content": f"[Format Error] {exc}"},
                                 {"role": "user", "content": "Re-read the required format in the system prompt and try again."}]
                    failure_count += 1
                    skip_reason = "Skipped: LLM response was not in the required format."
            except Exception as exc:
                messages.append({"role": "assistant", "content": f"[LLM Error] LLM call failed: {exc}"})
                failure_count += 1
                skip_reason = "Skipped: LLM call failed (check logs for details)."
                step_log["llm_error"] = str(exc)

            # --- execute_node ---
            if parsed is None:
                step_result = {"step_index": step_count, "code": "", "stdout": "", "stderr": "",
                               "error": skip_reason, "show_images": [], "tool_call_count": 0,
                               "execution_time_sec": 0.0}
            else:
                code = parsed["code"]
                safety_error = SecuritySandbox.check(code)
                if safety_error:
                    step_result = {"step_index": step_count, "code": code, "stdout": "", "stderr": "",
                                   "error": f"Security violation: {safety_error}", "show_images": [],
                                   "tool_call_count": 0, "execution_time_sec": 0.0}
                else:
                    er = self.km.execute(code, timeout=cfg.timeout_sec)
                    timing["exec"] += er.execution_time_sec
                    tool_log = self._drain_tool_timings()
                    step_log["tool_timings"] = tool_log
                    for entry in tool_log:
                        timing["tool:" + entry["name"]] += entry["sec"]
                        if entry.get("queued"):
                            timing["tool_queue_wait"] += entry["queued"]
                    show_images, show_warning = [], ""
                    if cfg.enable_sighted_feedback and er.stdout:
                        show_images = _parse_show_markers(er.stdout)
                        labels = _extract_show_labels(code)
                        for i, entry in enumerate(show_images):
                            if not entry["label"] and i < len(labels) and labels[i]:
                                entry["label"] = labels[i]
                        if cfg.max_show_images_per_step >= 0:
                            count = sum(len(e["images"]) for e in show_images)
                            if count > cfg.max_show_images_per_step:
                                show_images = _trim_images(show_images, cfg.max_show_images_per_step)
                                show_warning = (f"{count} images requested but only {cfg.max_show_images_per_step} "
                                                f"can be shown per step. Use fewer show() calls or fewer images per call.")
                    stdout = _SHOW_RE.sub("", er.stdout).strip()
                    # Only the kernel's own cell timeout (SpatialClaw matches "timed out" anywhere, which also caught
                    # HTTP errors such as "Connection ... timed out" and wiped the agent's variables for nothing).
                    if er.error and er.error.startswith("Cell execution timed out"):
                        try:
                            self.km.reset_namespace()
                            self.km.execute(injection, timeout=300)
                        except Exception:
                            self.km.restart()
                        # SpatialClaw resets silently; say so, or the agent keeps using variables that are gone.
                        er.error += ("\n[Kernel reset] The kernel was reset after the timeout: every variable you "
                                     "created in earlier steps is gone (InputImages, tools, vlm, show, ReturnAnswer "
                                     "are available again). Recompute what you still need.")
                    tool_calls = len(re.findall(r"tools\.\w+\.\w+\(", code)) + len(re.findall(r"vlm\.\w+\(", code))
                    step_result = {"step_index": step_count, "code": code, "stdout": stdout, "stderr": er.stderr,
                                   "error": er.error, "show_images": show_images,
                                   "show_truncation_warning": show_warning, "tool_call_count": tool_calls,
                                   "execution_time_sec": er.execution_time_sec}
                    total_tool_calls += tool_calls

            # --- feedback_node ---
            current_vars = self.km.get_variables()
            new_vars, changed_vars = VariableTracker.diff(variable_registry, current_vars)
            is_error = bool(step_result.get("error"))
            is_condensed = is_error and cfg.condense_errors and parsed is not None
            new_names = set(current_vars) - set(variable_registry) if is_condensed else set()
            has_survivors = False
            if new_names:
                line = _extract_error_line_number(step_result["error"])
                if line is None:
                    line = _search_code_for_pattern(step_result["code"].splitlines(), step_result["error"])
                has_survivors = line is not None
            full_rollback = is_condensed and not has_survivors
            if full_rollback and new_names:
                self.km.execute("\n".join(f"try:\n    del {n}\nexcept NameError:\n    pass" for n in sorted(new_names)),
                                timeout=10)
            large_warnings = [] if full_rollback else VariableTracker.check_large_variables(
                current_vars, max_size_mb=cfg.max_variable_size_mb)
            if full_rollback:
                var_summaries = {}
            elif has_survivors:
                var_summaries = {k: VariableTracker.format_summary(k, v) for k, v in new_vars.items()}
            else:
                var_summaries = {k: VariableTracker.format_summary(k, v) for k, v in {**new_vars, **changed_vars}.items()}

            answer = self.km.check_sentinel("_return_answer_result")
            if answer is not None and "text" in answer:
                self.km.clear_sentinel("_return_answer_result")
                final_answer = last_submitted = answer
                total_answer_attempts += 1
            feedback_text = FeedbackCollector.build_feedback(
                step_result, var_summaries, large_warnings, final_answer=final_answer,
                condensed=is_condensed, has_survivors=has_survivors)
            if final_answer is None and cfg.max_steps - (step_count + 1) <= 1:
                feedback_text += ("\n\n[LAST STEP] This is your final step. The session will be "
                                  "force-terminated after this. You MUST call ReturnAnswer(...) now "
                                  "to submit your best answer.")
            show_images = [] if full_rollback else step_result.get("show_images", [])
            if show_images and cfg.max_show_images_per_session >= 0:
                remaining = cfg.max_show_images_per_session - total_show
                if remaining <= 0:
                    show_images = []
                    feedback_text += (f"\n\n[show() budget] EXHAUSTED — 0/{cfg.max_show_images_per_session} remaining. "
                                      f"Further show() calls will not produce inline images.")
                else:
                    show_images = _trim_images(show_images, remaining)
                n_imgs = sum(len(e["images"]) for e in show_images)
                if n_imgs and final_answer is None:
                    feedback_text += (f"\n[show() budget] {cfg.max_show_images_per_session - total_show - n_imgs}"
                                      f"/{cfg.max_show_images_per_session} images remaining.")
                total_show += n_imgs
            n_videos = sum(len(e.get("videos", [])) for e in show_images)
            if n_videos:
                allowed = max(0, cfg.max_show_videos_per_session - total_videos)
                if n_videos > allowed:
                    dropped = n_videos - allowed
                    for e in reversed(show_images):
                        while e.get("videos") and dropped > 0:
                            e["videos"].pop()
                            dropped -= 1
                    feedback_text += (f"\n[show() video budget] only {allowed} of {n_videos} clip(s) attached; "
                                      f"{cfg.max_show_videos_per_session} clips per session.")
                n_videos = sum(len(e.get("videos", [])) for e in show_images)
                if n_videos:
                    feedback_text += (f"\n[Inline Videos] {n_videos} clip(s) attached below "
                                      f"({cfg.max_show_videos_per_session - total_videos - n_videos} remaining).")
                total_videos += n_videos
            if is_condensed:
                messages[-1] = {"role": "assistant",
                                "content": _condensed_ai_message(parsed, step_result["code"], step_result["error"])}
            if any(e["images"] or e.get("videos") for e in show_images):
                messages.append({"role": "user",
                                 "content": _multimodal_content(feedback_text, show_images, cfg.show_image_labels)})
            else:
                messages.append({"role": "user", "content": feedback_text})
            if not full_rollback:
                variable_registry = current_vars
            step_log.update(parsed=parsed, code=step_result.get("code"), stdout=step_result.get("stdout"),
                            error=step_result.get("error"), feedback=feedback_text,
                            shown=[p for e in show_images for p in e["_paths"]]
                            + [v["path"] for e in show_images for v in e.get("videos", [])],
                            execution_time_sec=step_result.get("execution_time_sec"))
            log["steps"].append(step_log)
            step_count += 1

            # --- answer check (extension; at most once) ---
            if final_answer is not None and self._may_reject(step_count, total_answer_attempts):
                msg = self._answer_check(final_answer, checks_done)
                if msg:
                    step_log["answer_check"] = msg
                    messages.append({"role": "user", "content": msg})
                    final_answer = None

        # --- force_terminate ---
        fallback = None
        if final_answer is None and last_submitted is not None:
            final_answer, fallback = last_submitted, "last_submitted"
        if final_answer is None:
            fallback = "none"
            t_fb = time.time()
            try:
                times = [i / fps for i in kf_list]
                question = FALLBACK_PROMPT.format(n=len(kf_list), times=", ".join(f"{t:.1f}" for t in times),
                                                  query=query)
                if not kf_list:  # video-only input
                    question = question.split("\n", 1)[1]
                text = self.llm_client.generate_vision_query(key_frames, question, system_prompt="", tag="fallback",
                                                             role_params=cfg.general_params,
                                                             usage_session_id=session_id,
                                                             videos=[initial_clip] if initial_clip else None)
                log["fallback_response"] = text
                tube = _parse_fallback_tube(text)
                if tube:
                    final_answer = {"text": f"fallback tube with {len(tube)} boxes", "raw_value": tube}
                    fallback = "cot_vlm"
            except Exception as exc:
                log["fallback_error"] = f"{type(exc).__name__}: {exc}"
            timing["fallback"] += time.time() - t_fb

        log.update(final_answer=final_answer, termination_reason=termination, fallback=fallback,
                   step_count=step_count, total_tool_calls=total_tool_calls, total_show_images=total_show,
                   total_show_videos=total_videos, total_answer_attempts=total_answer_attempts,
                   usage=self.llm_client.pop_session_usage(session_id), seconds=round(time.time() - t_start, 1))
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        phases = ("init", "plan", "llm_main", "exec", "anchor", "fallback")
        timing["other"] = log["seconds"] - sum(timing[k] for k in phases)  # feedback building, variable listing...
        log["timing"] = {k: round(v, 2) for k, v in sorted(timing.items())}
        log["host_llm_calls"] = json.loads(host_timing.drain_json())
        with open(os.path.join(session_dir, "trajectory.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False, default=str)
        return log


def _parse_fallback_tube(text: str) -> Dict[str, List[float]]:
    """Parse the fallback VLM's JSON into {whole second: normalized box}."""
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL) or re.search(r"(\[\s*\{.*\}\s*\])", text, re.DOTALL)
    if not m:
        return {}
    try:
        rows = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    tube = {}
    for row in rows:
        try:
            t = int(round(float(row["timestamp"])))
            x1, y1, x2, y2 = (min(max(float(v) / 1000, 0.0), 1.0) for v in row["bbox_2d"])
        except (KeyError, TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            tube[str(t)] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
    return tube
