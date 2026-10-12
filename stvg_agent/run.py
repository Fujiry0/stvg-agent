"""Run the STVG agent on AnyGroundBench spatio-temporal test queries.

Outputs under ``results/agent/<run_name>/``:
  predictions.jsonl                 one record per finished query (enables --resume)
  spatio_temporal/<run_name>_<domain>.json
                                    all records in AnyGroundBench's format, for scripts/eval_st.sh
  traj/<query_id>/                  trajectory.json, show() images and VLM query images

Usage:
  python -m stvg_agent.run --domain animal --max_samples 5 --run_name debug
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from stvg_agent.config import AgentConfig, load_model_config
from stvg_agent.tools.server_wait import DOWN_MARK
from stvg_agent.workflow import STVGAgentWorkflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_st_test(data_root: Path, domain: str, split: str = "test") -> dict:
    """AnyGroundBench spatio-temporal test entries keyed by query_id, skipping empty tubes (as the benchmark does)."""
    with open(data_root / domain / "meta-data" / f"st_{split}.json", encoding="utf-8") as f:
        data = json.load(f)
    return {qid: e for qid, e in data.items() if e["fps1_timestamp_0_1000_st_demonstration"]["tubes"]}


def ground_truth(entry: dict) -> dict:
    """GT in the same shape AnyGroundBench stores for Qwen-family models (for reference only)."""
    start, end = (float(v) for v in entry["temporal_range"].split())
    boxes = []
    for tube in entry["fps1_timestamp_0_1000_st_demonstration"]["tubes"]:
        for t in sorted(tube["bbox"], key=float):
            boxes.append({"timestamp": float(t), "bbox_2d": tube["bbox"][t]})
    return {"temporal_range": [start, end], "boxes": boxes}


def to_raw_pred(submission: dict) -> str:
    """Submission {sec: [x0, y0, x1, y1] in 0-1} -> JSON parsed by AnyGroundBench's evaluator."""
    rows = [{"timestamp": float(t), "bbox_2d": [round(v, 4) for v in box]}
            for t, box in sorted(submission.items(), key=lambda kv: float(kv[0]))]
    return json.dumps(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="animal")
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--run_name", default=None)
    p.add_argument("--out_root", default=str(PROJECT_ROOT / "results" / "agent"))
    p.add_argument("--query_ids", nargs="*", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Skip query_ids already in predictions.jsonl.")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Queries run in parallel worker processes (each with its own Jupyter kernel).")
    p.add_argument("--model_config", default=str(PROJECT_ROOT / "configs" / "model" / "qwen3.5-9b.json"))
    p.add_argument("--sam3_url", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--tag", default="", help="Free-text description stored in config.json.")
    return p.parse_args()


_WORKER: dict = {}


def _init_worker(config: AgentConfig) -> None:
    """Per-process workflow (Jupyter kernel + LLM client), reused across that process's queries."""
    import atexit

    _WORKER["workflow"] = STVGAgentWorkflow(config)
    _WORKER["model_id"] = f"agent/{config.llm_model}"
    atexit.register(_WORKER["workflow"].shutdown)


def _run_one(qid: str, entry: dict, video_path: str, session_dir: str, domain: str) -> dict:
    """Run one query and return its prediction record (AnyGroundBench format + agent info)."""
    t0 = time.time()
    try:
        result = _WORKER["workflow"].run(video_path, entry["text"].strip(), session_id=Path(session_dir).name, session_dir=session_dir,
                                          query_id=qid)
        answer = result["final_answer"] or {}
        submission = answer.get("raw_value") or {}
        termination, error = result["termination_reason"], None
        num_steps, fallback = result["step_count"], result["fallback"]
        timing = result.get("timing")
    except Exception as exc:
        traceback.print_exc()
        submission, termination, num_steps, fallback, timing = {}, "exception", 0, None, None
        error = f"{type(exc).__name__}: {exc}"
    traj = Path(session_dir) / "trajectory.json"
    infra_failure = bool(error and DOWN_MARK in error) or (traj.exists() and DOWN_MARK in traj.read_text())
    return {
        "query_id": qid,
        "query": f"{entry['video_name']}.mp4",
        "text": entry["text"].strip(),
        "demonstrations": [],
        "gt": ground_truth(entry),
        "raw_pred": to_raw_pred(submission),
        "domain_name": domain,
        "model_id": _WORKER["model_id"],
        "n_shot": 0,
        "query_video_path": f"{entry['video_name']}.mp4",
        "agent": {"termination": termination, "fallback": fallback, "num_steps": num_steps, "error": error,
                  "num_boxes": len(submission), "seconds": round(time.time() - t0, 1), "timing": timing,
                  "infra_failure": infra_failure},
    }


def record_code_version(out_dir: Path, args) -> None:
    """Append this launch's code version to code_versions.jsonl (a --resume run may use newer code) and save the
    uncommitted diff, so results can be traced to the exact code that produced them."""
    root = Path(__file__).resolve().parents[1]
    git = lambda *c: subprocess.run(["git", "-C", str(root), *c], capture_output=True, text=True).stdout
    diff = git("diff", "HEAD", "--", "stvg_agent", "scripts")
    src = hashlib.sha256()
    for f in sorted((root / "stvg_agent").rglob("*.py")):
        src.update(f.read_bytes())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rec = {"time": stamp, "commit": git("rev-parse", "HEAD").strip(), "dirty": bool(diff),
           "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16], "src_sha256": src.hexdigest()[:16],
           "argv": vars(args)}
    if diff:
        (out_dir / f"code_diff_{stamp}.patch").write_text(diff)
    with open(out_dir / "code_versions.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    args = parse_args()
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    entries = load_st_test(data_root, args.domain)
    qids = list(entries)
    if args.query_ids:
        qids = [q for q in qids if q in set(args.query_ids)]
    if args.max_samples is not None:
        qids = qids[:args.max_samples]

    pred_path = out_dir / "predictions.jsonl"
    done = {}
    if args.resume and pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["query_id"]] = rec
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    record_code_version(out_dir, args)

    config = load_model_config(args.model_config, AgentConfig())
    if args.sam3_url:
        config.sam3_url = args.sam3_url
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    with open(out_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    jobs = [(q, entries[q], str(data_root / args.domain / "videos" / "clips" / f"{entries[q]['video_name']}.mp4"),
             str(out_dir / "traj" / q), args.domain) for q in qids if q not in done]

    def record_done(record: dict) -> None:
        a = record["agent"]
        if a.get("infra_failure"):  # the tool server stayed down: keep it out of the predictions so --resume reruns it
            with open(out_dir / "predictions_infra_failed.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[infra failure] {record['query_id']}: tool server down, not recorded (rerun with --resume)",
                  flush=True)
            return
        done[record["query_id"]] = record
        with open(pred_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{len(done)}/{len(qids)}] {record['query_id']} {a['termination']} steps={a['num_steps']} "
              f"fallback={a['fallback']} boxes={a['num_boxes']} {a['seconds']}s", flush=True)

    if args.concurrency <= 1:
        _init_worker(config)
        try:
            for job in jobs:
                record_done(_run_one(*job))
        finally:
            _WORKER["workflow"].shutdown()
    else:
        ctx = mp.get_context("spawn")  # fresh processes: each gets its own kernel manager and clients
        with ProcessPoolExecutor(max_workers=args.concurrency, mp_context=ctx,
                                 initializer=_init_worker, initargs=(config,)) as pool:
            futures = [pool.submit(_run_one, *job) for job in jobs]
            for fut in as_completed(futures):
                record_done(fut.result())

    final_dir = out_dir / "spatio_temporal"
    final_dir.mkdir(exist_ok=True)
    records = [done[q] for q in qids if q in done]
    final_path = final_dir / f"{run_name}_{args.domain}.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} records: {final_path}")


if __name__ == "__main__":
    main()
