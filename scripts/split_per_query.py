"""Write one JSON file per query (prediction + metrics) from an ST evaluation.

Inputs are the evaluator's outputs:
  --per_query   *_with_per_query_metrics_spatio_temporal.json (from scripts/eval_st.sh)
  --tubes       data/<domain>/vidi_eval/results/<model>_spatio_temporal/tubes.csv — the predicted boxes
                exactly as scored (normalized xyxy per second, after the model-specific conversion)
Each <out_dir>/<query_id>.json holds: query, video, ground truth, raw model output, scored
prediction boxes, parse status and metrics (fps1 = the paper's numbers, and full_fps).

Usage:
  python scripts/split_per_query.py --per_query <json> --tubes <tubes.csv> --out_dir <dir>
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_query", required=True)
    p.add_argument("--tubes", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    scored = defaultdict(list)
    with open(args.tubes, newline="") as f:
        for row in csv.DictReader(f):
            scored[row["query_id"]].append({"time_sec": int(row["time_ms"]) / 1000,
                                            "box_xyxy": [round(float(row[k]), 4) for k in ("x0", "y0", "x1", "y1")]})
    failures = set()
    fail_csv = Path(args.tubes).with_name("tubes_parse_failures.csv")
    if fail_csv.exists():
        with open(fail_csv, newline="") as f:
            failures = {row["query_id"] for row in csv.DictReader(f)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = json.load(open(args.per_query))
    for r in records:
        qid = str(r["query_id"])
        boxes = scored.get(qid, [])
        parse_failed = qid in failures
        record = {
            "query_id": qid,
            "text": r.get("text") or r.get("text_query"),
            "video": r.get("query_video_path") or r.get("query"),
            "model_id": r.get("model_id"),
            "n_shot": r.get("n_shot"),
            "gt": r.get("gt"),
            "raw_pred": r.get("raw_pred"),
            "parse_failed": parse_failed,
            "pred_scored": [] if parse_failed else boxes,
            "pred_interval_sec": None if parse_failed or not boxes else
            [min(b["time_sec"] for b in boxes), max(b["time_sec"] for b in boxes)],
            "metrics": r.get("metric"),
        }
        with open(out_dir / f"{qid}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(records)} files to {out_dir} ({len(failures)} parse failures)")


if __name__ == "__main__":
    main()
