"""Take queries that hit a down tool server out of a run so `--resume` runs them again.

Trajectories containing a connection error to the tool server are moved to traj_toolserver_down/ and their
records to predictions_toolserver_down.jsonl (kept, not deleted).
Usage: python3 scripts/drop_toolserver_failures.py <run_name>
"""
import json, re, shutil, sys
from pathlib import Path

PAT = re.compile(r"TOOL_SERVER_DOWN|Connection refused|NewConnectionError|Max retries exceeded")
run = Path(__file__).resolve().parents[1] / "results/agent" / sys.argv[1]
bad = sorted(p.parent.name for p in (run / "traj").glob("*/trajectory.json") if PAT.search(p.read_text()))
if bad:
    keep, drop = [], []
    for line in open(run / "predictions.jsonl"):
        if line.strip():
            (drop if json.loads(line)["query_id"] in bad else keep).append(line)
    with open(run / "predictions_toolserver_down.jsonl", "a") as f:
        f.writelines(drop)
    (run / "predictions.jsonl").write_text("".join(keep))
    (run / "traj_toolserver_down").mkdir(exist_ok=True)
    for q in bad:
        dst = run / "traj_toolserver_down" / q
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(run / "traj" / q), str(dst))
print(f"{sys.argv[1]}: {len(bad)} queries hit a down tool server: {' '.join(bad)}")
