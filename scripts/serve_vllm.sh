#!/usr/bin/env bash
# Serve Qwen3.5-9B with vLLM (OpenAI API on :8100). GPU chosen via CUDA_VISIBLE_DEVICES (PCI order).
# Leaves room on the GPU for the SAM3 server (see --gpu-memory-utilization).
# Context 262144 + fp8 KV cache, as SpatialClaw's Qwen3.5/3.6 servers (launch_managers/vllm_manager/models.json);
# with main max_tokens 131072 this leaves ~131k input tokens per call.
# 300 images/prompt: 32 key frames + SpatialClaw's show() budget of 250 per session.
# 32 videos/prompt: the whole-video clip + a budget of 30 show() clips; clips are read as local files.
# --reasoning-parser splits Qwen3.5 thinking into reasoning_content (roles use enable_thinking=true).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
# FlashInfer (chosen for the fp8 KV cache) JIT-compiles kernels and needs CUDA >= 12.9 for SM 12.x
# (Blackwell); the system default /usr/local/cuda is 12.8, so use CUDA 13.0 (matches vLLM's cu130 torch).
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$CUDA_HOME/bin:$PATH

exec envs/vllm/.venv/bin/vllm serve Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b \
  --host 127.0.0.1 --port 8100 \
  --gpu-memory-utilization ${VLLM_GPU_UTIL:-0.55} \
  --max-model-len ${VLLM_MAX_LEN:-262144} \
  --kv-cache-dtype fp8 \
  --limit-mm-per-prompt '{"image": 300, "video": 32}' \
  --allowed-local-media-path "$PWD" \
  --reasoning-parser qwen3 \
  "$@"
