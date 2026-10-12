#!/usr/bin/env bash
# Tool server on :8101 — SAM3 (tools.SAM3) + Qwen3-VL-Embedding-8B (tools.Temporal.relevance).
# The embedding model stays on CPU and is moved to the GPU only during /similarity calls,
# so it can share the GPU with vLLM; SAM3 is also moved to CPU during those calls (--swap_sam3), since
# vLLM (27 GiB) + embedder (15 GiB) + SAM3 (3.5 GiB) left no room for activations.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

# Few malloc arenas: every request runs in its own thread, and per-thread arenas kept freed memory (RSS growth).
export MALLOC_ARENA_MAX=2
PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} exec envs/tools/.venv/bin/python -m stvg_agent.tools_server \
  --port 8101 --tmp_root "${STVG_TMP_ROOT:-/tmp/stvg_agent}/sam3_frames" --swap_sam3 ${STVG_TOOLS_ARGS:-} "$@"
