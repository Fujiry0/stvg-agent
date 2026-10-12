"""ReturnAnswer for spatio-temporal grounding.

Same mechanism as SpatialClaw's ``kernel_types/return_answer.py`` (a valid answer
is stored in a builtins sentinel that the agent loop detects), but the answer is
a *tube*: pixel boxes keyed by absolute video frame index. It is converted to
whole-second timestamps (nearest frame within 0.5 s) and normalized
``[x0, y0, x1, y1]`` boxes, which is what AnyGroundBench evaluates.
"""

import math

SENTINEL_NAME = "_return_answer_result"


class ReturnAnswer:
    """Submit the final tube and terminate the agent.

    Usage::

        ReturnAnswer(boxes, object=0)          # PerFrameBoxes from tools.SAM3 (one object)
        ReturnAnswer({120: [x1, y1, x2, y2],   # dict: absolute frame index -> pixel box
                      150: [x1, y1, x2, y2]})
        ReturnAnswer(boxes, object=0, start_frame=120, end_frame=390)  # trim to the event

    Include only frames where the query holds (the ground-truth interval); frames
    outside it should have no box.
    """

    # Set at injection time for the current video / InputImages resolution.
    fps: float = 30.0
    width: int = 1
    height: int = 1
    num_frames: int = 0
    num_inputs: int = 0          # len(InputImages)
    index_warning: bool = False  # AgentConfig.index_warning
    index_warned: bool = False   # the warning fires once per query; the same call again is accepted

    def __init__(self, tube, object=0, start_frame=None, end_frame=None, frames_are_video_indices=False):
        from_boxes = hasattr(tube, "get_track")
        if from_boxes:
            tube = tube.get_track(object)
        if not isinstance(tube, dict) or not tube:
            raise TypeError("ReturnAnswer expects a non-empty dict {frame_index: [x1, y1, x2, y2]} "
                            "or a PerFrameBoxes (with object=...).")
        keys = [int(k) for k in tube]
        if (self.index_warning and not type(self).index_warned and not from_boxes and not frames_are_video_indices
                and len(keys) > 1 and self.num_inputs < self.num_frames and max(keys) < self.num_inputs):
            type(self).index_warned = True
            raise ValueError(
                f"The dict keys {sorted(keys)[:6]}... are all below len(InputImages)={self.num_inputs}: they look like "
                f"InputImages list positions, but ReturnAnswer needs absolute video frame indices (use "
                f"InputImages[i].frame_index, e.g. InputImages[{max(keys)}].frame_index). If they really are video "
                f"frames, run the same ReturnAnswer again (it is accepted the second time).")
        frames = {}
        for fi, box in tube.items():
            fi = int(fi)
            if box is None:
                continue
            if len(box) != 4 or any(v is None or (isinstance(v, float) and math.isnan(v)) for v in box):
                raise ValueError(f"Invalid box at frame {fi}: {box}. Expected [x1, y1, x2, y2] in pixels.")
            if start_frame is not None and fi < start_frame:
                continue
            if end_frame is not None and fi > end_frame:
                continue
            from stvg_agent.kernel_types import coords
            c = coords.active()
            x1, y1, x2, y2 = c.to_pixel(box, self.width, self.height)
            if c.is_pixel and max(x1, y1, x2, y2) <= 1.0 and max(self.width, self.height) > 10:
                print(f"[ReturnAnswer] WARNING: box {box} at frame {fi} looks normalized; "
                      f"boxes must be in PIXELS of InputImages ({self.width}x{self.height}).")
            x1, x2 = sorted((min(max(x1, 0.0), self.width), min(max(x2, 0.0), self.width)))
            y1, y2 = sorted((min(max(y1, 0.0), self.height), min(max(y2, 0.0), self.height)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            frames[fi] = [x1 / self.width, y1 / self.height, x2 / self.width, y2 / self.height]
        if not frames:
            raise ValueError("No valid boxes to submit (after trimming to start_frame/end_frame).")

        # Resample to whole seconds with the ground truth's convention (AnyGroundBench fps1 tubes quantized
        # to 1 s by the evaluator): the first second is the tube start rounded half-up, the last second is
        # the tube end rounded DOWN (a tube ending at 24.5 s has no box at 25 s). Each second takes the
        # nearest submitted frame within 0.5 s.
        indices = sorted(frames)
        t_first, t_last = indices[0] / self.fps, indices[-1] / self.fps
        first_sec = math.floor(t_first + 0.5)
        last_sec = math.floor(t_last + 1e-6)
        if last_sec < first_sec:  # tube shorter than a second and not spanning an integer second
            first_sec = last_sec = math.floor((t_first + t_last) / 2 + 0.5)
        seconds = {}
        for s in range(first_sec, last_sec + 1):
            nearest = min(indices, key=lambda fi: abs(fi / self.fps - s))
            if abs(nearest / self.fps - s) <= 0.5:
                seconds[s] = [round(v, 4) for v in frames[nearest]]
        if not seconds:
            raise ValueError("Submitted frames do not cover any whole-second timestamp.")

        import builtins

        text = (f"tube with {len(seconds)} boxes over {min(seconds)}-{max(seconds)} s "
                f"(frames {indices[0]}-{indices[-1]})")
        result = {"text": text, "raw_value": {str(k): v for k, v in seconds.items()},
                  "frames": {str(fi): [round(v, 4) for v in frames[fi]] for fi in indices}}
        setattr(builtins, SENTINEL_NAME, result)
        self._result = result
        print(f"[ReturnAnswer] Answer submitted: {text}")

    def __repr__(self) -> str:
        return f"ReturnAnswer(text='{self._result['text']}')"
