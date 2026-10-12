#!/usr/bin/env bash
# Run the STVG agent (needs scripts/serve_vllm.sh and scripts/serve_tools.sh running).
# Usage: scripts/run_agent.sh --run_name demo --max_samples 5 [--domain animal] [--concurrency 3] [other stvg_agent.run args]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

exec envs/agent/.venv/bin/python -m stvg_agent.run --domain animal "$@"
