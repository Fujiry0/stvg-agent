#!/usr/bin/env bash
# Restart the tool server with extra args (e.g. --offload_video) and wait until it is healthy.
cd "$(dirname "$0")/.."
pkill -f '[s]tvg_agent.tools_server'; sleep 5
# The wrapper logs how the server ended (exit code; 128+N = killed by signal N, e.g. 137 = SIGKILL / OOM killer).
setsid nohup bash -c 'scripts/serve_tools.sh "$@"; rc=$?; echo "$(date "+%F %T") tool server exited with code $rc" >> logs/tools_server_exit.log' \
  _ "$@" > logs/tools_server.log 2>&1 < /dev/null &
for i in $(seq 120); do sleep 5; curl -sf http://127.0.0.1:8101/health > /dev/null && exit 0; done
echo "tool server did not come up" >&2; exit 1
