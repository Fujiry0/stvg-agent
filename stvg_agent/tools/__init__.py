"""Tools exposed to the agent as ``tools.X``.

SpatialClaw's ToolsModule, grounding subset (SAM3, Geometry), plus Video / Draw for
clips and tube visualization and Temporal for interval refinement / verification.
"""

from stvg_agent.tools.geometry import GeometryUtils
from stvg_agent.tools.sam3_tool import SAM3Tool
from stvg_agent.tools.temporal import TemporalTool
from stvg_agent.tools.video_tools import DrawTool, VideoTool
from stvg_agent.kernel_types.timing import Timed


TOOL_NAMES = ("SAM3", "Geometry", "Video", "Draw", "Temporal")


class ToolsModule:
    TOOL_NAMES = TOOL_NAMES

    def __init__(self, sam3: SAM3Tool, video: VideoTool, draw: DrawTool, temporal: TemporalTool):
        available = {"SAM3": sam3, "Geometry": GeometryUtils(), "Video": video, "Draw": draw, "Temporal": temporal}
        self.enabled = list(self.TOOL_NAMES)
        for name in self.enabled:  # each call's wall time goes to the timing log
            setattr(self, name, Timed(available[name], f"tools.{name}"))

    def __repr__(self) -> str:
        return f"ToolsModule({', '.join(self.enabled)})"
