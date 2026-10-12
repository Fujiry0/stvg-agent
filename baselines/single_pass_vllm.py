"""Single-pass zero-shot baseline through the vLLM server (the engine the agent uses).

The same data, prompt and output format as AnyGroundBench's own inference (``test.py`` with HF ``generate``), but the
generation is a request to the vLLM OpenAI server, so many queries run concurrently. AnyGroundBench's classes are used only
for data and prompts: the model loader is replaced by a stub.

Video: sampled at ``--fps`` (1) with at most ``--query_max_frames_num`` frames, frames resized to <= 384*384 pixels
(as AnyGroundBench's Qwen message builder asks), written as an mp4 whose frame rate keeps the real time axis,
and sent as ``file://`` (vLLM needs --allowed-local-media-path). Thinking off, greedy, max 4096 new tokens.

Usage (vLLM on :8100 up):
  envs/agent/.venv/bin/python baselines/single_pass_vllm.py --domain_name animal --query_max_frames_num 32 --concurrency 8
Outputs ``results/baselines/<tag>/spatio_temporal/<time>_<domain>.json(l)``, evaluated with scripts/eval_st.sh.
Or run ``scripts/run_zeroshot.sh``, which also evaluates and splits the result per query.
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

ANYGROUND_ROOT = Path(os.environ.get("ANYGROUND_ROOT", str(Path(__file__).resolve().parents[1] / "third_party" / "anygroundbench")))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANYGROUND_ROOT))

import av  # noqa: E402
import numpy as np  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
from openai import OpenAI  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

MAX_PIXELS = 384 * 384


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain_name", default="animal")
    p.add_argument("--tag", default="qwen3.5-9b-vllm")
    p.add_argument("--query_max_frames_num", type=int, default=32)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--base_url", default="http://127.0.0.1:8100/v1")
    p.add_argument("--model", default="qwen3.5-9b")
    p.add_argument("--target_query_ids", nargs="*", default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--clip_dir", default=str(PROJECT_ROOT / "results" / "tmp" / "zs_clips"))
    return p.parse_args()


def make_icl(args):
    """AnyGroundBench's SpatioTemporalVideoICL without loading a model."""
    try:
        import torch  # noqa: F401
    except ImportError:  # vLLM client venv has no torch; base.py only needs it for local generation
        from unittest.mock import MagicMock
        for name in ("torch", "torch.nn", "torch.nn.functional"):
            sys.modules[name] = MagicMock()
    import src.anygroundbench.base as base
    base.load_model_and_processor = lambda *a, **k: (None, None)
    from src.anygroundbench.spatio_temporal import SpatioTemporalVideoICL
    ns = argparse.Namespace(
        task="spatio_temporal", domain_name=args.domain_name, model_name="qwen3_5", model_id="Qwen/Qwen3.5-9B",
        tag=None, n_shot=0, alpha=None, support_max_frames_num=args.query_max_frames_num,
        query_max_frames_num=args.query_max_frames_num, fps=int(args.fps), max_test_samples=args.max_test_samples,
        target_query_ids=args.target_query_ids, device="cpu", seed=42, non_interleave_text=False, overlay=None,
        add_prompt=[], prompt_variant="base", text_only_demonstration=False, resize=None,
        remove_max_support_frames=False, max_retries=5)
    from src.utils.validate_args import validate_args
    validate_args(ns, argparse.ArgumentParser())
    return SpatioTemporalVideoICL(ns)


def resize_max_pixels(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w * h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (w * h))
        w, h = round(w * s), round(h * s)
    return img.resize((max(2, w // 2 * 2), max(2, h // 2 * 2)), Image.LANCZOS)


def render_video(path: str, fps: float, max_frames: int, out_path: str) -> str:
    """Frames at ``fps`` (<= ``max_frames``, evenly spaced) as an mp4 that keeps the real time axis."""
    vr = VideoReader(path, ctx=cpu(0), num_threads=2)
    total, vfps = len(vr), vr.get_avg_fps()
    duration = total / vfps
    n = max(2, min(max_frames, int(math.floor(duration * fps)) + 1))
    idx = sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))
    clip_fps = (len(idx) - 1) / max(1e-6, (idx[-1] - idx[0]) / vfps) if len(idx) > 1 else fps
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames = [resize_max_pixels(Image.fromarray(vr[i].asnumpy())) for i in idx]
    with av.open(out_path, "w") as c:
        s = c.add_stream("libx264", rate=Fraction(max(1, round(clip_fps * 1000)), 1000))
        s.width, s.height, s.pix_fmt = frames[0].width, frames[0].height, "yuv420p"
        s.options = {"crf": "12"}
        for im in frames:
            for pkt in s.encode(av.VideoFrame.from_image(im)):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return out_path


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    icl = make_icl(args)
    _, test_videos = icl.get_data()
    out_dir = PROJECT_ROOT / "results" / "baselines" / args.tag / "spatio_temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = Path(args.resume_from) if args.resume_from else out_dir / f"{icl.now}_{args.domain_name}.jsonl"
    done = {}
    if jsonl.exists():
        done = {r["query_id"]: r for r in map(json.loads, filter(str.strip, open(jsonl, encoding="utf-8")))}
    todo = [q for q in test_videos if q not in done]
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    def run(qid):
        messages = icl.prepare_messages([], qid)
        content = messages[-1]["content"]
        video = next(c for c in content if c["type"] == "video")
        text = next(c for c in content if c["type"] == "text")["text"]
        clip = render_video(video["video"], args.fps, args.query_max_frames_num,
                            os.path.join(args.clip_dir, args.domain_name, f"{qid.replace('/', '_')}.mp4"))
        t0 = time.time()
        resp = client.chat.completions.create(
            model=args.model, temperature=0.0, max_tokens=args.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": "file://" + os.path.abspath(clip)}},
                {"type": "text", "text": text}]}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        pred = resp.choices[0].message.content or ""
        entry = icl.test_video_to_entries[qid]
        record = {
            "query_id": qid, "query": entry["video_name"], "demonstrations": [],
            "gt": {"temporal_range": entry["gt_time_range"],
                   "boxes": icl._build_mllm_sampled_boxes(entry["sampled_boxes"], icl.spatio_temporal_key_name,
                                                          icl.spatio_temporal_bbox_name,
                                                          icl.spatio_temporal_use_mmss)},
            "raw_pred": pred, "domain_name": icl.domain, "model_id": icl.model_id, "n_shot": 0,
            "text_only_demonstration": False, "query_video_path": entry["video_name"],
            "engine": {"name": "vllm", "sec": round(time.time() - t0, 1),
                       "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                       "finish_reason": resp.choices[0].finish_reason}}
        return record

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, open(jsonl, "a", encoding="utf-8") as f:
        for rec in tqdm(pool.map(run, todo), total=len(todo)):
            done[rec["query_id"]] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    records = [done[q] for q in test_videos if q in done]
    out = jsonl.with_suffix(".json")
    json.dump(records, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} records: {out}")


if __name__ == "__main__":
    main()
