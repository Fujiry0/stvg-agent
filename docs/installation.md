# Installation

Prerequisites, environment setup and model weights for STVG Agent.

> ⬅ [Main README](../README.md) &nbsp;·&nbsp; Next: [Architecture »](architecture.md)

---

## Prerequisites

* Linux with an NVIDIA GPU. One GPU of about **48 GB** runs everything: vLLM (Qwen3.5-9B with an fp8 KV cache, about
  27 GB at `VLLM_GPU_UTIL=0.55`) and the tool server (SAM 3, and Qwen3-VL-Embedding-8B which is moved to the GPU only
  during similarity calls, about 4–18 GB). With a smaller GPU lower `VLLM_MAX_LEN` and `VLLM_GPU_UTIL`.
* An NVIDIA driver with CUDA 12.8 or newer. On Blackwell GPUs (SM 12.x) vLLM's fp8 KV cache selects FlashInfer, which
  JIT-compiles kernels and needs a **CUDA toolkit >= 12.9** (`scripts/serve_vllm.sh` uses `CUDA_HOME=/usr/local/cuda-13.0`;
  override with the `CUDA_HOME` environment variable).
* Python 3.12 and [`uv`](https://docs.astral.sh/uv/) (`pip install uv`), `git`, `ffmpeg`.
* About 125 GB of host RAM is comfortable for 5 parallel workers (each query has its own Jupyter kernel; a kernel is
  capped at 24 GB, see `STVG_KERNEL_MAX_RSS_GB`).
* A Hugging Face account with access to the gated [`facebook/sam3`](https://huggingface.co/facebook/sam3) weights.
* Network access to huggingface.co for the initial model downloads.

---

## Install

```bash
git clone https://github.com/Fujiry0/stvg-agent.git && cd stvg-agent
uvx --from huggingface_hub hf auth login     # a token whose account accepted the facebook/sam3 license
bash scripts/setup.sh
```

`scripts/setup.sh` runs everything below. To run a single step: `bash scripts/setup.sh --agent` (or `--vllm`,
`--tools`, `--eval`, `--weights`).

| Step | What it does |
|---|---|
| `--agent` | `envs/agent/.venv`: the agent runtime and the Jupyter kernels (`uv pip install -e .`) |
| `--vllm` | `envs/vllm/.venv`: vLLM 0.30.0 serving Qwen3.5-9B |
| `--tools` | `envs/tools/.venv`: torch 2.8 (cu128), transformers 4.57, SAM 3 (cloned to `third_party/sam3`), qwen-vl-utils |
| `--eval` | `envs/eval/.venv`: numpy, pandas, decord for the AnyGroundBench evaluation |
| `--weights` | downloads Qwen3.5-9B, Qwen3-VL-Embedding-8B and `facebook/sam3` (`sam3.pt`) into `HF_HUB_CACHE` |

The environments are separate because their PyTorch / transformers versions differ. Any vLLM release that serves
`Qwen/Qwen3.5-9B` with `--reasoning-parser qwen3` should work in place of 0.30.0.

Verify:

```bash
envs/agent/.venv/bin/python -c "import stvg_agent, jupyter_client, openai, cv2; print('agent OK')"
envs/vllm/.venv/bin/vllm --version
envs/tools/.venv/bin/python -c "import torch, transformers, sam3; print('tools OK', torch.cuda.is_available())"
envs/eval/.venv/bin/python -c "import numpy, pandas, decord; print('eval OK')"
```

### Model weights

| Weights | Used by | Where it is read |
|---|---|---|
| `Qwen/Qwen3.5-9B` | vLLM (all LLM roles) | `scripts/serve_vllm.sh` |
| `Qwen/Qwen3-VL-Embedding-8B` | `tools.Temporal.relevance` | `stvg_agent/tools_server.py` (`HF_HUB_CACHE`) |
| `facebook/sam3` (`sam3.pt`) | `tools.SAM3` | `stvg_agent/tools_server.py` (`$HF_HUB_CACHE/models--facebook--sam3/snapshots/*/sam3.pt`) |

`HF_HUB_CACHE` defaults to `~/.cache/huggingface/hub` (set in `scripts/env.sh`).

---

## Data

Put AnyGroundBench under **`data/` in the repository root** (a symlink to wherever you keep it works). The agent and the
evaluation code both read `data/<domain>/...` relative to the repository root. The domains are `animal`, `industry`,
`sports`, `surgery` and `safety` (Public Security).

```text
data/<domain>/
├── meta-data/        st_test.json, st_train.json, ...   (in the Hugging Face release)
├── simrank/          ...                                (in the Hugging Face release)
├── videos/clips/     <video_name>.mp4                   (reconstructed from the source datasets)
└── vidi_eval/        created by the evaluation script
```

Download the release from [Hugging Face](https://huggingface.co/datasets/rinost081/AnyGroundBench) for `meta-data/`.
The videos must be reconstructed from the original source datasets with AnyGroundBench's preparation scripts; follow
the README of [rinost081/anygroundbench](https://github.com/rinost081/anygroundbench) (a copy is in
`third_party/anygroundbench/README.upstream.md`). The agent runs on `meta-data/st_test.json` and reads the videos from
`videos/clips/`. `--data_root` reads the data from another location (evaluation always uses `data/` in the repository
root). `ANYGROUND_ROOT` (default `third_party/anygroundbench`) is where the evaluation code is, not the data.

---

## Host settings

`scripts/env.sh` sets the paths and the GPU (`CUDA_VISIBLE_DEVICES`, default `0`; `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the
index matches `nvidia-smi`). Override anything for your host in `scripts/env.local.sh` (git-ignored), e.g.

```bash
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_CACHE=/data/hf_cache/hub
export STVG_TMP_ROOT=/data/tmp/stvg_agent
```

Next, start the servers and run the agent: see the [Quickstart](../README.md#quickstart).
