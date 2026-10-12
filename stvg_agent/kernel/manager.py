"""Persistent Jupyter (IPython) kernel for agent code execution.

Synchronous port of SpatialClaw's ``spatial_agent/kernel/manager.py``
(JupyterKernelManager): timeout with interrupt, namespace reset, variable
introspection, and the builtins sentinel used by ``ReturnAnswer``.
"""

import json
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil
from jupyter_client import KernelManager

# Resident-memory cap per kernel. One runaway cell (e.g. all frames of a long video decoded into RAM) reached 27 GB;
# with several workers that exhausted the host's RAM, and the swapping stalled the tool server.
KERNEL_MAX_RSS_GB = float(os.environ.get("STVG_KERNEL_MAX_RSS_GB", "24"))

# Names injected by the agent runtime; excluded from variable introspection.
INJECTED_NAMES = [
    "InputImages", "Metadata", "tools", "feedback", "ReturnAnswer", "show", "vlm",
    "SAM3", "Geometry", "Video", "Draw", "Temporal", "np", "numpy", "math", "matplotlib", "plt", "scipy",
    "collections", "itertools", "functools", "ndimage", "spatial", "signal", "optimize",
]


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None  # None on success; traceback on error
    display_data: List[Any] = field(default_factory=list)
    execution_time_sec: float = 0.0


class JupyterKernelManager:
    def __init__(self, timeout_sec: int = 600, kernel_name: str = "python3"):
        self.timeout_sec = timeout_sec
        self.kernel_name = kernel_name
        self._km: Optional[KernelManager] = None
        self._kc = None
        self._init_code = ""
        self._injection_code = ""

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._km is not None and self._kc is not None and self._km.is_alive()

    def start(self) -> str:
        self._km = KernelManager(kernel_name=self.kernel_name)
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=60)
        return str(self._km.kernel_id)

    def _rss_gb(self) -> float:
        try:
            proc = psutil.Process(self._km.provisioner.process.pid)
            return sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)]) / 1e9
        except Exception:
            return 0.0

    def set_init_code(self, code: str) -> None:
        self._init_code = code

    def set_injection_code(self, code: str) -> None:
        self._injection_code = code

    def reset_namespace(self) -> None:
        """Clear user variables and re-run the stdlib init code (for kernel reuse)."""
        self.execute("%reset -f", timeout=10)
        self.clear_sentinel()
        if self._init_code:
            result = self.execute(self._init_code, timeout=60)
            if result.error:
                raise RuntimeError(f"Namespace reset init failed: {result.error}")

    def interrupt(self) -> None:
        if self._km is not None:
            try:
                self._km.interrupt_kernel()
            except Exception:
                pass

    def restart(self) -> None:
        """Hard restart, then re-run init + injection code."""
        self.shutdown()
        self.start()
        if self._init_code:
            self.execute(self._init_code, timeout=120)
        if self._injection_code:
            self.execute(self._injection_code, timeout=120)

    def shutdown(self) -> None:
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
        if self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass
        self._km = None
        self._kc = None

    # ------------------------------------------------------------------
    def execute(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """Execute ``code``; on timeout the kernel is interrupted (the caller decides on reset)."""
        if self._kc is None:
            raise RuntimeError("Kernel not started or has been shut down.")
        timeout = timeout or self.timeout_sec
        t0 = time.monotonic()
        msg_id = self._kc.execute(code, store_history=True)
        stdout: List[str] = []
        stderr: List[str] = []
        display: List[Any] = []
        error: Optional[str] = None
        while True:
            remaining = timeout - (time.monotonic() - t0)
            if remaining <= 0:
                self.interrupt()
                self._drain_until_idle(msg_id, 10)
                elapsed = time.monotonic() - t0
                return ExecutionResult("".join(stdout), "".join(stderr),
                                       f"Cell execution timed out after {elapsed:.1f}s (limit {timeout}s).",
                                       display, elapsed)
            try:
                msg = self._kc.get_iopub_msg(timeout=min(1.0, remaining))
            except queue.Empty:
                rss = self._rss_gb()
                if rss > KERNEL_MAX_RSS_GB:
                    return self._memory_exceeded(msg_id, rss, stdout, stderr, display, t0)
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type, content = msg.get("msg_type", ""), msg.get("content", {})
            if msg_type == "stream":
                (stdout if content.get("name") == "stdout" else stderr).append(content.get("text", ""))
            elif msg_type == "error":
                error = "\n".join(content.get("traceback", []))
            elif msg_type in ("execute_result", "display_data"):
                display.append(content.get("data", {}))
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        return ExecutionResult("".join(stdout), "".join(stderr), error, display, time.monotonic() - t0)

    def _memory_exceeded(self, msg_id, rss, stdout, stderr, display, t0) -> ExecutionResult:
        """Interrupt the cell; if the memory is still held afterwards, restart the kernel (variables are lost)."""
        self.interrupt()
        self._drain_until_idle(msg_id, 10)
        note = f"Cell stopped: the kernel used {rss:.0f} GB of memory (limit {KERNEL_MAX_RSS_GB:.0f} GB). "
        if self._rss_gb() > KERNEL_MAX_RSS_GB:
            self.restart()
            note += "The kernel was restarted, so all variables are gone. "
        note += "Load fewer frames at a time (lower resolution, a shorter window or a larger stride)."
        elapsed = time.monotonic() - t0
        return ExecutionResult("".join(stdout), "".join(stderr), note, display, elapsed)

    def _drain_until_idle(self, msg_id: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = self._kc.get_iopub_msg(timeout=0.5)
            except queue.Empty:
                continue
            if (msg.get("parent_header", {}).get("msg_id") == msg_id and msg.get("msg_type") == "status"
                    and msg.get("content", {}).get("execution_state") == "idle"):
                return

    # ------------------------------------------------------------------
    def get_variables(self) -> Dict[str, Dict[str, Any]]:
        """Kernel namespace summary: ``name -> {type, shape?, dtype?, len?, keys?, ...}``."""
        code = r'''
import json as _json_mod
_SKIP = set(%s) | {'In', 'Out', 'get_ipython', 'exit', 'quit'}
_var_info = {}
for _name in list(globals().keys()):
    if _name.startswith('_') or _name in _SKIP:
        continue
    _val = globals()[_name]
    if callable(_val) and not hasattr(_val, '__dataclass_fields__'):
        continue
    _info = {"type": type(_val).__name__}
    try:
        if hasattr(_val, 'shape'):
            _info["shape"] = str(_val.shape)
        if hasattr(_val, 'dtype'):
            _info["dtype"] = str(_val.dtype)
        if hasattr(_val, 'frame_indices'):
            _info["frame_indices"] = list(_val.frame_indices)
        if isinstance(_val, (list, tuple)):
            _info["len"] = len(_val)
        if isinstance(_val, dict):
            _info["len"] = len(_val)
            _info["keys"] = [str(k) for k in list(_val.keys())[:10]]
        if hasattr(_val, 'nbytes'):
            _info["size_mb"] = round(_val.nbytes / 1e6, 2)
    except Exception:
        pass
    _var_info[_name] = _info
print(_json_mod.dumps(_var_info))
del _json_mod, _var_info, _SKIP
''' % repr(INJECTED_NAMES)
        result = self.execute(code, timeout=10)
        if result.error:
            return {}
        try:
            return json.loads(result.stdout.strip().split("\n")[-1])
        except (json.JSONDecodeError, IndexError):
            return {}

    def check_sentinel(self, sentinel_name: str = "_return_answer_result") -> Optional[Dict]:
        code = f"""
import json as _j, builtins as _b
_val = getattr(_b, '{sentinel_name}', None)
print(_j.dumps(_val) if _val is not None else '__NONE__')
del _j, _b, _val
"""
        result = self.execute(code, timeout=10)
        if result.error or not result.stdout.strip():
            return None
        text = result.stdout.strip().split("\n")[-1]
        if text == "__NONE__":
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def clear_sentinel(self, sentinel_name: str = "_return_answer_result") -> None:
        self.execute(f"import builtins as _b\nif hasattr(_b, '{sentinel_name}'):\n    delattr(_b, '{sentinel_name}')\ndel _b",
                     timeout=10)
