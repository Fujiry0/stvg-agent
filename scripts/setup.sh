#!/usr/bin/env bash
# Create the four virtual environments under envs/ and download the model weights.
# Usage: bash scripts/setup.sh [--agent] [--vllm] [--tools] [--eval] [--weights]     (default: everything)
# Needs: uv, git, and for --weights a Hugging Face login with access to facebook/sam3 (huggingface-cli login).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(--agent --vllm --tools --eval --weights)
want() { local s; for s in "${steps[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
venv() { uv venv "envs/$1/.venv" --python 3.12; }
pip()  { uv pip install --python "envs/$1/.venv/bin/python" "${@:2}"; }

mkdir -p logs

if want --agent; then
  echo "== agent environment"; venv agent
  pip agent -e .
fi

if want --vllm; then
  echo "== vllm environment"; venv vllm
  pip vllm vllm==0.30.0
fi

if want --tools; then
  echo "== tools environment (SAM 3 + embedding model)"; venv tools
  pip tools torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  pip tools transformers==4.57.6 accelerate qwen-vl-utils==0.0.14 opencv-python-headless \
      decord av numpy scipy pandas matplotlib psutil requests tqdm pillow
  [ -d third_party/sam3 ] || git clone https://github.com/facebookresearch/sam3.git third_party/sam3
  pip tools -e third_party/sam3
fi

if want --eval; then
  echo "== eval environment"; venv eval
  pip eval numpy pandas decord
fi

if want --weights; then
  echo "== model weights (HF_HUB_CACHE=$HF_HUB_CACHE)"
  HF="uvx --from huggingface_hub hf"
  $HF download Qwen/Qwen3.5-9B
  $HF download Qwen/Qwen3-VL-Embedding-8B
  $HF download facebook/sam3 sam3.pt   # gated: accept the license on the model page and log in first
fi

echo "done. Next: put AnyGroundBench under data/ (see README), then start the servers."
