"""System prompt for the main agent (sighted mode).

Same section structure and order as SpatialClaw's ``spatial_agent/llm/system_prompt.py``
(header, response_format, show_api, vlm_api, available_tools, coordinate_system,
robust_computation, evidence_hierarchy, return_answer, code_rules, workflow +
budget, session_input); response_format and workflow are verbatim.
"""

import re
from typing import Any, Dict, List

from stvg_agent.config import AgentConfig
from stvg_agent.llm.prompt_common import (
    build_input_description,
    code_rules_section,
    coordinate_system_section,
    evidence_hierarchy_section,
    return_answer_section,
    robust_computation_section,
    sam3_tool_section,
    show_api_section,
    vlm_api_section,
)

RESPONSE_FORMAT = (
    "## Response Format (MANDATORY)\n\n"
    "Every response MUST use this exact markdown format:\n\n"
    "**Purpose**: What this code step accomplishes\n\n"
    "**Reasoning**: Your chain-of-thought reasoning\n\n"
    "**Next Goal**: What you plan to do next\n\n"
    "**Code**:\n"
    "```python\n"
    "# Python code to execute — written verbatim, no escaping needed\n"
    "print(\"hello\")\n"
    "```\n\n"
    "All four sections are required. Code must be inside a fenced ```python block."
)


def key_frame_mapping_text(key_frame_list_indices: List[int], key_frame_indices: List[int], fps: float) -> str:
    """Compact key-frame mapping (as SpatialClaw), with each frame's time added."""
    def line(i: int) -> str:
        li, ai = key_frame_list_indices[i], key_frame_indices[i]
        return f"  Key frame #{i + 1} → InputImages[{li}] (video frame {ai}, t={ai / fps:.1f}s)"

    n = len(key_frame_indices)
    if n <= 16:
        return "\n".join(line(i) for i in range(n))
    return "\n".join([line(i) for i in range(5)] + [f"  ... ({n - 10} more) ..."] + [line(i) for i in range(n - 5, n)])


def build_system_prompt(config: AgentConfig, metadata: Dict[str, Any], key_frame_indices: List[int],
                        key_frame_list_indices: List[int]) -> str:
    num_images = metadata["num_images"]
    fps = metadata["fps"]
    duration = metadata.get("duration_sec")
    max_steps = config.max_steps

    header = (
        "# Spatio-Temporal Video Grounding Agent\n\n"
        "You solve spatio-temporal video grounding tasks by writing Python code in a Jupyter kernel.\n"
        "Your input is `InputImages` (a list of `FrameImage` objects) and `Metadata`. "
        "Full details in \"This Session's Input\" below."
    )
    available_tools = "## Available Tools\n\nAccess via `tools.X.method(...)` (each tool is also bound by its bare name, e.g. `SAM3`).\n\n" + sam3_tool_section(
        config.sam3_max_video_frames)

    if max_steps <= 1:
        workflow = (
            "## Workflow — SINGLE-TURN MODE\n"
            "You have **exactly ONE code execution step**. There is NO second chance.\n"
            "Your code block MUST:\n"
            "1. Gather all needed evidence (visual queries, tool calls, computation)\n"
            "2. Call `ReturnAnswer(...)` at the end to submit your final answer\n\n"
            "If you do not call `ReturnAnswer`, the system will force-terminate and guess.\n"
            "Do NOT leave analysis for a \"next step\" — there is none."
        )
    else:
        workflow = (
            "## Workflow\n"
            "An execution plan has been prepared for you. **Follow the plan faithfully**, writing code for each step.\n"
            "- If the plan specifies tool calls, execute them. Do not skip planned steps.\n"
            "- **Cross-validate**: Before answering, ensure your conclusion is supported by at least two independent lines of evidence. If visual perception and geometric computation disagree, diagnose why before concluding.\n"
            "- When quantitative methods are available, prefer them over qualitative visual judgments — but verify that the pipeline's inputs (masks, coordinates) are correct."
            f"\n\n## Budget: {max_steps} steps."
        )

    num_kf = len(key_frame_indices)
    session_input = (
        "## This Session's Input\n\n"
        + (f"Your first message contains {num_kf} **key frames** — a visual overview subset of InputImages"
           + (" — and " if config.initial_video else "") if num_kf else "Your first message contains ")
        + (f"the **whole video as a clip** ({config.initial_video_fps:g} fps, at most "
           f"{config.initial_video_max_frames} frames, each labeled with its time)" if config.initial_video else "")
        + ". Use `show(image)` to inspect additional frames or intermediate results inline. "
        "Use `show(tools.Video.clip(...))` to watch stretches of the video.\n\n"
        f"{build_input_description(metadata)}\n\n"
        + (f"- `InputImages` — ALL {num_images} frames of the video (the full set, not just key frames). "
           if num_images == metadata["total_frames"] else
           f"- `InputImages` — {num_images} frames sampled from the video at {config.video_max_fps:g} fps (the full "
           "set, not just key frames); tools (SAM3, clips) still read every native frame of the video. ")
        + "Use `show()` to see any of them.\n"
        "  - `InputImages[i]` → `FrameImage` (PIL-compatible, has `.frame_index` attribute)\n"
        + (f"  - **`i` is a list index (0 to {num_images}-1).** Here every video frame is included, so `InputImages[i].frame_index == i`.\n"
           if num_images == metadata["total_frames"] else
           f"  - **`i` is a list index (0 to {num_images}-1), NOT a video frame number.** Use `img.frame_index` to get the absolute video frame number.\n"
           "  - Example: if `InputImages.frame_indices == [0, 30, 60]`, then `InputImages[1]` is the frame at video position 30.\n")
        + "  - `InputImages[start:end]` → `InputImages` subset (preserves frame indices)\n"
        "  - `InputImages.frame_indices` — list of absolute video frame indices\n"
        + ("  - **Key frames in your first message are a SUBSET of InputImages.** Not all InputImages entries are shown as key frames.\n"
           f"- **Key frame → variable mapping:**\n{key_frame_mapping_text(key_frame_list_indices, key_frame_indices, fps)}\n"
           if num_kf else "")
        + 
        f"- `Metadata` — `is_video=True`, `fps={fps}`, `total_frames={metadata['total_frames']}`, `num_images={num_images}`"
        + (f", `duration_sec={duration:.1f}`" if duration else "")
        + f", `width={metadata['width']}`, `height={metadata['height']}` (InputImages resolution)\n"
        "  - `Metadata.video_source` — path to original video file"
    )

    sections = [
        header,
        RESPONSE_FORMAT,
        show_api_section(config.max_show_images_per_step, config.max_show_images_per_session),
        vlm_api_section(),
        available_tools,
        coordinate_system_section(),
        robust_computation_section(),
        evidence_hierarchy_section(),
        return_answer_section(),
        code_rules_section(),
        workflow,
        session_input,
    ]
    prompt = "\n\n".join(s for s in sections if s)
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()
