"""Ride out a tool-server outage instead of answering without the tool.

When the tool server cannot be reached (refused / connect timeout), ``post`` waits for ``/health`` to come
back (scripts/tools_watchdog.sh restarts a dead server within a few minutes) and sends the request again.
If it stays down, the error carries TOOL_SERVER_DOWN so the run leaves the query out of predictions.jsonl
(``--resume`` then runs it again) instead of recording an answer made without the tool.
"""

import time

import requests

DOWN_MARK = "TOOL_SERVER_DOWN"
MAX_WAIT_SEC = 600  # keeps a whole outage inside one cell (cell timeout 1200 s)


def wait_for_health(base_url: str, max_wait: float = MAX_WAIT_SEC) -> bool:
    """Poll ``base_url``/health until it answers (True) or ``max_wait`` seconds pass (False)."""
    t_end = time.time() + max_wait
    while time.time() < t_end:
        try:
            if requests.get(f"{base_url}/health", timeout=30).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(10)
    return False


def post(base_url: str, path: str, json: dict, timeout: float) -> requests.Response:
    """requests.post that, if the server is unreachable, waits for it to come back and tries once more."""
    try:
        return requests.post(f"{base_url}{path}", json=json, timeout=timeout)
    except requests.ConnectionError as exc:  # refused / connect timeout / connection dropped mid-request
        print(f"[tool server] {path} unreachable ({type(exc).__name__}); waiting for it to come back", flush=True)
        last = exc
    if wait_for_health(base_url):
        try:
            return requests.post(f"{base_url}{path}", json=json, timeout=timeout)
        except requests.ConnectionError as exc:
            last = exc
    raise RuntimeError(f"{DOWN_MARK}: tool server at {base_url} is unreachable ({last}). "
                       f"This is an infrastructure failure, not a problem with your code.")
