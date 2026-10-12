"""Planning prompt for the isolated (blind) planning session.

Same structure as SpatialClaw's ``spatial_agent/llm/planning_prompt.py``: the planner
gets the tool docs and frame metadata but never the images, must not pre-conclude
the answer, and writes a plan (task analysis, information needs, computation plan,
verification checklist, verification, fallbacks). The tool-selection guidance is
rewritten for grounding; header, rules and checklist are SpatialClaw's wording
(with "spatial reasoning" replaced by the grounding task).
"""

import re
from typing import Any, Dict, List, Optional

from PIL import Image

from stvg_agent.config import AgentConfig
from stvg_agent.llm.client import image_to_base64_url, video_parts
from stvg_agent.llm.prompt_common import (
    build_input_description,
    coordinate_system_section,
    evidence_hierarchy_section,
    robust_computation_section,
    sam3_tool_section,
    show_api_section,
    vlm_api_section,
)


def build_planning_system_prompt(config: AgentConfig, metadata: Dict[str, Any]) -> str:
    sighted = config.planning_sighted
    if sighted:
        header = (
            "# Spatio-Temporal Grounding Planner\n\n"
            "You are a planning-only assistant. You will be given a spatio-temporal video grounding task "
            "and a sparse set of key frames from the video. Use the key frames only to tailor the plan "
            "(which objects are present, whether similar objects may be confused, which parts of the video "
            "deserve a closer look). Your job is to create a concrete execution plan that a separate agent "
            "(which can inspect every frame and run tools) will follow."
        )
    else:
        header = (
            "# Spatio-Temporal Grounding Planner\n\n"
            "You are a planning-only assistant. You will be given a spatio-temporal video grounding task "
            "and metadata about the input frames. You do NOT see the images. Your job is to "
            "create a concrete execution plan that a separate agent (which CAN see images) will follow."
        )
    available_tools = "## Available Tools\n\n" + sam3_tool_section(config.sam3_max_video_frames)
    tool_decision = (
        "## Planning Strategy — Tool Selection\n\n"
        "For each step, choose the tool whose evidence shape matches what the task needs AND "
        "that is expected to produce a reliable result on this input:\n"
        "- `vlm.locate`             — pixel coordinates of an object describable in words.\n"
        "- `tools.SAM3`             — per-frame boxes given a point/box or text prompt; video methods track through a frame window.\n"
        "- `tools.Geometry`         — numeric ops on coordinates / boxes / frame-time conversion.\n"
        "- `tools.Video.clip`       — a clip of any frame window (≤ 2 fps) to watch or to ask `vlm.ask_with_thinking` about.\n"
        "- `tools.Draw.tube`        — the predicted tube drawn on a clip, to verify before submitting.\n"
        "- `tools.Temporal`         — refine a boundary in its ±3 s neighborhood; text–video relevance of a segment.\n"
        "- `show()`                 — the executing agent looks at frames and decides.\n"
        "- `vlm.ask_with_thinking`  — a separate visual reasoner returns a text answer about the provided frames.\n\n"
        "`vlm.ask_with_thinking` is appropriate when:\n"
        "  (a) the task needs a kind of visual judgment that none of the specialized tools above produces "
        "(e.g. whether / when an action is happening), or\n"
        "  (b) a specialized tool was the right shape but its output on this input does not actually answer the "
        "question (failed, ambiguous, or self-inconsistent).\n"
        "You can plan multiple `vlm.ask_with_thinking` calls — different frame selections or rephrasings — when "
        "one call is not sufficient.\n\n"
        "### Temporal localization\n"
        "Plan how to find WHEN the query holds from the video itself: motion is only visible across "
        "consecutive frames, so inspect dense consecutive frames around the candidate start and end, and confirm "
        "with more than one frame selection or phrasing. "
        "Use the whole-video clip and short `tools.Video.clip(..., fps=2)` windows around the boundaries for this. "
        "Then refine each boundary locally (`tools.Temporal.refine_boundary`). "
        "Check that the segment is relevant to the query (`tools.Temporal.relevance`).\n\n"
        "### Spatial localization and tracking\n"
        "Plan how to find WHICH object the query refers to (text detection, or `vlm.locate` when several "
        "similar objects exist), then track it over the interval with SAM3 video methods (windows of at most "
        f"{config.sam3_max_video_frames} frames).\n\n"
        "### Verification\n"
        "- Plan to verify tool inputs (prompts, boxes, frame windows) before relying on their outputs.\n"
        "- Plan to watch the final tube with `tools.Draw.tube(...)` before submitting (boundaries and identity).\n"
        "- When two evidence sources may disagree, name the diagnostic step.\n"
        "- Multiple consistent `vlm.ask_with_thinking` answers across different frame selections or phrasings "
        "are stronger evidence than a single call.\n\n"
        "**NEVER pre-conclude the answer.** Your job is to plan an investigation, not to answer the question. "
        "Do NOT write \"the answer is likely X\" or \"I expect the answer is Y\" anywhere in your plan. "
        "If you catch yourself forming a hypothesis from the task text, that hypothesis is EXACTLY what tools "
        "must verify — it is not evidence.\n\n"
        + evidence_hierarchy_section() + "\n\n" + robust_computation_section()
    )
    input_desc = (
        "## This Session's Input\n\n"
        f"{build_input_description(metadata)}\n"
        + ("You see only sparse key frames; boundaries and object identity must still be established with "
           "tools on the full frame set. Do NOT read the answer off the key frames — plan an investigation.\n\n"
           if sighted else
           "You will be told the frame count and key frame indices, but you cannot see the actual images. "
           "Do NOT guess or infer answers from the task text — plan an investigation using tools.\n\n")
        + ("**IMPORTANT:** `InputImages` is a list indexed 0 to N-1; here it contains every video frame, so "
           "`InputImages[i].frame_index == i`. Frame f is at f / fps seconds."
           if metadata["num_images"] == metadata["total_frames"] else
           # SpatialClaw's wording for sampled video frames
           "**IMPORTANT:** `InputImages` is a list indexed 0 to N-1 (NOT by video frame numbers). "
           "The absolute video frame indices are in `InputImages.frame_indices`. In your plan, use list "
           "indices like `InputImages[0]`, `InputImages[-1]`, `InputImages[len(InputImages)//2]` — "
           "NOT absolute frame numbers. Tools (SAM3 windows, clips) and ReturnAnswer take absolute frame "
           "indices; frame f is at f / fps seconds.")
    )
    task = (
        "## Your Task\n\n"
        "Create a plan that covers:\n\n"
        "1. **Task Analysis** — Which object does the query refer to (category, distinguishing attributes)? "
        "What event or state defines the time interval? Could several similar objects be present?\n\n"
        "2. **Information Needs** — What do you need to localize the interval and the object reliably? "
        "Refer to the tool selection heuristic above.\n\n"
        "3. **Computation Plan** — Ordered list of concrete steps. For each step: the tool call or "
        "VLM query, what data it produces, how it feeds into the next step. **Boxes must come from tools "
        "(SAM3 / vlm.locate), not from eyeballing.**\n\n"
        "4. **Verification Checklist** — See the CHECKLIST section below. You MUST include this "
        "section with a JSON array of verification items. Generate this IMMEDIATELY after your "
        "computation plan, before writing anything else.\n\n"
        "5. **Verification** — Cross-validate before answering:\n"
        "   - Every conclusion must be supported by at least two independent pieces of evidence.\n"
        "   - Visually verify tool inputs and outputs before using them for the final answer.\n"
        "   - Name concrete verification steps (not conditional \"if needed\" phrases).\n"
        "   - If two evidence sources disagree, include a diagnostic step.\n\n"
        "6. **Fallbacks** — What could go wrong? Alternative approach?"
    )
    rules = (
        "## CRITICAL RULES\n"
        "- Respond with your plan in **plain text only**.\n"
        "- Do NOT write JSON objects (except for the CHECKLIST section).\n"
        "- Do NOT write executable Python code.\n"
        "- Do NOT produce an answer or pre-conclude what the answer might be.\n"
        "- Do NOT write phrases like \"the answer is likely X\", \"therefore the answer is\", "
        + ("\"I expect the answer is\". Key frames are too sparse to answer from — anything you notice in them "
           "is a hypothesis for the agent to verify with tools.\n" if sighted else
           "\"I expect the answer is\". You are BLIND — you cannot answer the question.\n")
        +
        "- Your ONLY job is to PLAN. A separate agent will execute the plan later.\n"
        "- Use pseudocode sketches (e.g., `tools.SAM3.segment_video_by_box(...)`) to describe steps, "
        "but do NOT write full code."
    )
    checklist = (
        "## Verification Checklist\n\n"
        "After your plan, add a `### CHECKLIST` section with a JSON array of verification items. "
        "Each item has a `priority` (\"HIGH\", \"MEDIUM\", or \"LOW\") and a `description`:\n\n"
        "```\n"
        "### CHECKLIST\n"
        "```json\n"
        "[\n"
        '  {"priority": "HIGH", "description": "Verify the tracked object is the one the query refers to"},\n'
        '  {"priority": "HIGH", "description": "Confirm the interval boundaries by inspecting frames around the start and end"},\n'
        '  {"priority": "MEDIUM", "description": "Cross-check the interval using an independent method or frame selection"},\n'
        '  {"priority": "LOW", "description": "Check the track for missing frames or identity switches"}\n'
        "]\n"
        "```\n\n"
        "**Priority guide:**\n"
        "- **HIGH**: If wrong, the answer is almost certainly wrong (e.g., wrong object identity, wrong interval).\n"
        "- **MEDIUM**: Could affect accuracy but has fallback (e.g., VLM identification needs cross-check).\n"
        "- **LOW**: Nice to verify but unlikely to cause errors.\n\n"
        "HIGH-priority items MUST be verified before submitting the final answer."
    )
    sections = [header, available_tools,
                show_api_section(config.max_show_images_per_step, config.max_show_images_per_session),
                vlm_api_section(), coordinate_system_section(), tool_decision, input_desc, task, rules, checklist]
    prompt = "\n\n".join(s for s in sections if s)
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()


def build_planning_user_message(instruction: str, key_frame_indices: List[int],
                                 key_frames: Optional[List[Image.Image]] = None,
                                 mapping_text: str = "", num_total_images: int = 0, video=None) -> dict:
    """Planner input, as SpatialClaw's ``build_planning_user_message``.

    Blind (``key_frames=None``, SpatialClaw's plan_node default): frame indices only.
    Sighted: header + key-frame mapping + images + question.
    """
    if key_frames or video is not None:
        header = (f"Here are {len(key_frames)} key frames — a visual overview subset of "
                  f"InputImages ({num_total_images} total frames)." if key_frames else
                  f"Here is the whole video as a clip ({num_total_images} total frames in InputImages).")
        if mapping_text:
            header += f"\nKey frame mapping (context position → InputImages index → video frame):\n{mapping_text}"
        content = [{"type": "text", "text": header}]
        content += [{"type": "image_url", "image_url": {"url": image_to_base64_url(img)}} for img in key_frames or []]
        if video is not None:
            content += video_parts(video)
        content.append({"type": "text", "text": f"Question: {instruction}"})
        return {"role": "user", "content": content}
    text = f"Question: {instruction}"
    # SpatialClaw says "The input has N frames (frame indices: ...)" with N = the key frames, which made the
    # planner treat the key-frame list as the whole input. Say what InputImages holds.
    if key_frame_indices:
        text = (f"InputImages holds {num_total_images} frames of the video. The executing agent's first message "
                f"shows {len(key_frame_indices)} of them as key frames (absolute video frame indices: "
                f"{key_frame_indices}).\n\n" + text)
    elif num_total_images:
        text = f"InputImages holds {num_total_images} frames of the video.\n\n" + text
    return {"role": "user", "content": text}
