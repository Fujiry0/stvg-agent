import json,shutil,sys
from pathlib import Path
run=Path(__file__).resolve().parents[1]/"results/agent"/sys.argv[1]
keep,drop=[],[]
for l in open(run/"predictions.jsonl"):
    if not l.strip(): continue
    x=json.loads(l); a=x["agent"]
    (drop if a.get("termination")=="exception" or a.get("infra_failure") else keep).append((l,x["query_id"]))
open(run/"predictions_exception.jsonl","a").writelines(l for l,_ in drop)
(run/"predictions.jsonl").write_text("".join(l for l,_ in keep))
(run/"traj_exception_down").mkdir(exist_ok=True)
for _,q in drop:
    s=run/"traj"/q
    if s.exists(): shutil.move(str(s),str(run/"traj_exception_down"/q))
print(f"{sys.argv[1]}: dropped {len(drop)} exception queries, kept {len(keep)}")
