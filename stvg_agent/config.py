"""Agent configuration.

Field names and defaults follow SpatialClaw's ``spatial_agent/config.py``
(SpatialAgentConfig). Fields for tools we do not use (Reconstruct, reflection,
SLURM/vLLM discovery) are omitted. The defaults are the best configuration.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Optional


@dataclass
class LLMRoleParams:
    """Sampling parameters for one LLM role (as in SpatialClaw)."""

    max_tokens: int = 16384
    temperature: float = 0.6
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    enable_thinking: Optional[bool] = None
    # vLLM thinking_token_budget: reasoning is force-ended after this many tokens and the answer follows
    # (not in SpatialClaw; bounds runaway reasoning). None = no budget.
    thinking_token_budget: Optional[int] = None


ROLE_NAMES = ("main", "planning", "general", "vlm", "vlm_grounding")


@dataclass
class AgentConfig:
    # --- LLM ---
    llm_model: str = "qwen3.5-9b"
    llm_base_url: str = "http://127.0.0.1:8100/v1"
    llm_api_key: str = "EMPTY"
    main_params: LLMRoleParams = field(default_factory=LLMRoleParams)
    planning_params: LLMRoleParams = field(default_factory=LLMRoleParams)
    general_params: LLMRoleParams = field(default_factory=LLMRoleParams)
    vlm_params: LLMRoleParams = field(default_factory=lambda: LLMRoleParams(max_tokens=32768))
    vlm_grounding_params: LLMRoleParams = field(default_factory=lambda: LLMRoleParams(max_tokens=16384))

    # --- Agent loop ---
    max_steps: int = 30
    max_failures: int = 30
    # per Jupyter cell execution. SpatialClaw: 600. Raised because the 3 workers share one tool server that runs
    # one request at a time, so a SAM3 call can wait several minutes behind the others' calls.
    timeout_sec: int = 1200
    # New Jupyter kernel for every query (SpatialClaw reuses one via %reset, which does not free memory).
    fresh_kernel_per_query: bool = True

    # --- Tools ---
    sam3_url: str = "http://127.0.0.1:8101"
    embed_url: str = "http://127.0.0.1:8101"  # tools_server also serves /similarity (Qwen3-VL-Embedding)
    sam3_max_video_frames: int = 1000
    gpu_tool_max_retries: int = 3

    # --- VLM queries ---
    vlm_query_timeout_sec: int = 600

    # --- Video ---
    image_max_long_edge: Optional[int] = 768
    num_key_frames: int = 32
    # InputImages = the video sampled at this fps, with absolute frame indices (SpatialClaw's video benchmark
    # configs, e.g. Video-MME, use video_max_fps=1; its default None keeps every frame). Tools (SAM3, clips,
    # Draw, Temporal) still read every native frame. None = every frame in InputImages.
    video_max_fps: Optional[float] = 1.0
    # SAM3 video methods track on every k-th native frame (1 = every frame, as SpatialClaw); 3 is about 10 fps on
    # 30 fps video. The skipped frames are filled by linear interpolation (the agent sees a dense track).
    sam3_frame_stride: int = 3
    sam3_interpolate: bool = True
    # Extension (not in SpatialClaw): the whole video as a clip in the first message, sampled like
    # the single-pass baseline (1 fps, <= 120 frames, <= 384*384 pixels), and a video show() budget.
    initial_video: bool = True
    initial_video_fps: float = 1.0
    initial_video_max_frames: int = 120
    max_show_videos_per_session: int = 30

    # --- Planning ---
    enable_planning: bool = True
    # The planner also sees the key frames and the whole-video clip (SpatialClaw's blind planner gets frame indices
    # only), still told not to answer.
    planning_sighted: bool = True

    # SpatialClaw's guard for answer checks: never reject when the remaining budget is at most this many steps
    # or after this many submissions.
    max_total_answer_attempts: int = 5
    min_budget_for_answer_reject: int = 3

    # --- Answer checks and temporal estimate (not in SpatialClaw) ---
    # The whole-video "when?" estimate (one call over the video at 1 fps, each frame labeled with its time) runs in the
    # background and is given to the agent as a message after the plan (evidence only, never overriding the answer).
    anchor_hint: bool = True
    # Frames of that call (1 fps, at most this many; a longer video is sampled at N / duration fps).
    when_max_frames: int = 64
    confirm_short: bool = True       # tube of <= 2 s: ask once to check the seconds before / after first
    index_warning: bool = True       # ReturnAnswer keys that look like InputImages list positions -> error once
    # Draw the time on still frames given to vlm.ask_with_thinking or show() (video clips always carry it).
    time_label_stills: bool = True

    # --- Sighted feedback ---
    enable_sighted_feedback: bool = True
    max_show_images_per_step: int = -1
    max_show_images_per_session: int = 250
    show_image_labels: bool = True

    # --- Context management ---
    condense_errors: bool = True

    # --- Resource guardrails ---
    max_variable_size_mb: int = 500

    def to_dict(self) -> dict:
        return asdict(self)


def load_model_config(path: str, config: AgentConfig) -> AgentConfig:
    """Apply a SpatialClaw-style model JSON ({llm_model, roles: {main: {...}, ...}})."""
    with open(path) as f:
        data = json.load(f)
    for key in ("llm_model", "llm_base_url", "llm_api_key"):
        if key in data:
            setattr(config, key, data[key])
    valid = {f.name for f in fields(LLMRoleParams)}
    for role, params in data.get("roles", {}).items():
        if role not in ROLE_NAMES:
            raise ValueError(f"unknown role '{role}' in {path}")
        unknown = set(params) - valid
        if unknown:
            raise ValueError(f"unknown params {sorted(unknown)} for role '{role}' in {path}")
        setattr(config, f"{role}_params", LLMRoleParams(**params))
    return config
