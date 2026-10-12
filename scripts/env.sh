# Shared environment for stvg_agent scripts. Source this file; do not execute it.
# Per-host overrides go in scripts/env.local.sh (gitignored).
_STVG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$_STVG_ROOT/scripts/env.local.sh" ] && source "$_STVG_ROOT/scripts/env.local.sh"
export ANYGROUND_ROOT=${ANYGROUND_ROOT:-$_STVG_ROOT/third_party/anygroundbench}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}
export STVG_TMP_ROOT=${STVG_TMP_ROOT:-/tmp/stvg_agent}
# CUDA's default order differs from nvidia-smi; with PCI order, index 0 is the first GPU in nvidia-smi.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EVAL_PY=${EVAL_PY:-$_STVG_ROOT/envs/eval/.venv/bin/python}
