"""Draw ground-truth and predicted tubes on the video, one mp4 per query.

Ground truth (green): the evaluator's full-frame GT tubes
(data/<domain>/vidi_eval/vue-stg-benchmark/tubes.csv). Prediction (red): the boxes that were
actually scored (per-query files from scripts/split_per_query.py, one box per second), each shown
for ±0.5 s around its second. A header shows the time, query, both intervals and fps1 metrics.

Usage:
  python scripts/visualize_st.py --per_query_dir results/eval/qwen3.5-9b_0shot_animal/per_query_files \
      --query_ids test_85 test_8 --out_dir results/eval/qwen3.5-9b_0shot_animal/vis
"""

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_COLOR, PRED_COLOR = (40, 220, 40), (255, 50, 50)


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def load_gt(domain: str, qids: set) -> dict:
    gt = defaultdict(list)
    with open(PROJECT_ROOT / "data" / domain / "vidi_eval" / "vue-stg-benchmark" / "tubes.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["query_id"] in qids:
                gt[row["query_id"]].append((int(row["time_ms"]) / 1000, [float(row[k]) for k in ("x0", "y0", "x1", "y1")]))
    return {q: sorted(v) for q, v in gt.items()}


def box_at(entries, t, tol):
    """Box of the entry nearest to t (within tol seconds), else None."""
    if not entries:
        return None
    times = np.array([e[0] for e in entries])
    i = int(np.argmin(np.abs(times - t)))
    return entries[i][1] if abs(times[i] - t) <= tol else None


def draw_frame(img, t, gt_box, pred_box, header_lines, status):
    w, h = img.size
    header_h = 22 * len(header_lines) + 30
    canvas = Image.new("RGB", (w, h + header_h), (20, 20, 20))
    canvas.paste(img, (0, header_h))
    d = ImageDraw.Draw(canvas)
    f, fs = font(15), font(17)
    for i, line in enumerate(header_lines):
        d.text((8, 6 + 22 * i), line, fill=(235, 235, 235), font=f)
    d.text((8, 6 + 22 * len(header_lines)), status, fill=(255, 220, 90), font=fs)
    lw = max(2, h // 150)
    for box, color, tag in ((gt_box, GT_COLOR, "GT"), (pred_box, PRED_COLOR, "pred")):
        if box is None:
            continue
        x0, y0, x1, y1 = box[0] * w, box[1] * h + header_h, box[2] * w, box[3] * h + header_h
        d.rectangle([x0, y0, x1, y1], outline=color, width=lw)
        d.text((x0 + 4, y0 + 2), tag, fill=color, font=fs)
    return canvas


def render(record, gt_entries, video_path, out_path, out_fps):
    cap = cv2.VideoCapture(str(video_path))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(vfps / out_fps)))
    pred_entries = [(p["time_sec"], p["box_xyxy"]) for p in record["pred_scored"]]
    gs, ge = record["gt"]["temporal_range"]
    pi = record["pred_interval_sec"]
    m = record["metrics"]["fps1"]
    header = [
        f"{record['query_id']}  \"{record['text']}\"",
        f"GT (green) {gs:.1f}-{ge:.1f}s   |   pred (red) " + (f"{pi[0]:.1f}-{pi[1]:.1f}s" if pi else "none"),
        f"fps1: t_IoU={m['t_IoU']:.2f}  v_IoU={m['v_IoU']:.2f}  v_IoU_Int={(m['v_IoU_Int'] or 0):.2f}",
    ]
    # half the spacing of the ground-truth annotations (Surgery annotates one box every 1-2 s)
    gt_times = sorted({e[0] for e in gt_entries})
    gaps = [b - a for a, b in zip(gt_times, gt_times[1:]) if b > a]
    gt_tol = max(0.6 / vfps, (min(gaps) / 2 + 1e-6) if gaps else 0.0)
    tmp = out_path.with_suffix(".frames.mp4")
    writer = None
    fi = 0
    while fi < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / vfps
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if img.width > 960:
            img = img.resize((960, round(img.height * 960 / img.width)))
        gt_box = box_at(gt_entries, t, gt_tol)
        pred_box = box_at(pred_entries, t, 0.5)
        status = f"t={t:6.2f}s   " + ("IN GT interval" if gs <= t <= ge else "outside GT") + \
                 ("   | pred box" if pred_box is not None else "   | no pred box")
        out = draw_frame(img, t, gt_box, pred_box, header, status)
        if writer is None:
            size = (out.width // 2 * 2, out.height // 2 * 2)
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), vfps / step, size)
        writer.write(cv2.cvtColor(np.asarray(out.crop((0, 0, *size))), cv2.COLOR_RGB2BGR))
        fi += step
    writer.release()
    cap.release()
    # Re-encode to H.264 so browsers / editors can play it.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", str(out_path)], check=True)
    tmp.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_query_dir", required=True)
    p.add_argument("--query_ids", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--domain", default="animal")
    p.add_argument("--fps", type=float, default=10.0, help="Output frame rate (frames are subsampled).")
    args = p.parse_args()

    gt = load_gt(args.domain, set(args.query_ids))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for qid in args.query_ids:
        record = json.load(open(Path(args.per_query_dir) / f"{qid}.json"))
        video = PROJECT_ROOT / "data" / args.domain / "videos" / "clips" / record["video"]
        out = out_dir / f"{qid}.mp4"
        render(record, gt.get(qid, []), video, out, args.fps)
        print(f"{qid}: {out}")


if __name__ == "__main__":
    main()
