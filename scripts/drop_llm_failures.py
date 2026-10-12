"""Take queries whose LLM calls failed (vLLM down) out of a run so `--resume` runs them again.
Records go to predictions_llm_down.jsonl, trajectories to traj_llm_down/ (kept, not deleted).
Usage: python3 scripts/drop_llm_failures.py <run_name>"""
import json, shutil, sys
from pathlib import Path

run = Path(__file__).resolve().parents[1] / "results/agent" / sys.argv[1]
bad = sorted(p.parent.name for p in (run / "traj").glob("*/trajectory.json") if "LLM call failed after" in p.read_text())
keep, drop = [], []
for line in open(run / "predictions.jsonl"):
    if line.strip():
        (drop if json.loads(line)["query_id"] in bad else keep).append(line)
with open(run / "predictions_llm_down.jsonl", "a") as f:
    f.writelines(drop)
(run / "predictions.jsonl").write_text("".join(keep))
(run / "traj_llm_down").mkdir(exist_ok=True)
for q in bad:
    dst = run / "traj_llm_down" / q
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(run / "traj" / q), str(dst))
print(f"{sys.argv[1]}: {len(bad)} queries with LLM failures dropped, kept {len(keep)}")
