"""Side-by-side figure: ground truth (green) and predictions (red) of several runs on the same query.

Usage:
  python scripts/compare_figure.py --domain animal --query_ids test_85 test_28 --out_dir docs/assets \
      --run "Qwen3.5-9B single pass=results/eval/<run>/per_query_files" \
      --run "stvg-agent=results/eval/<run>/per_query_files"
One image per query: a row per run, four frames spread over the ground-truth interval (a frame is drawn with the
prediction of the second it falls in; no red box = the run predicted nothing there).
"""

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from visualize_st import GT_COLOR, PRED_COLOR, PROJECT_ROOT, box_at, font, load_gt

TILE_W = 360


def gt_tolerance(entries, vfps):
    """Half the spacing of the ground-truth annotations (Surgery annotates one box every 1-2 s, other domains 10-30 per
    second), so that a frame between two annotated times shows the nearest annotation."""
    times = sorted({e[0] for e in entries})
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    return max(0.6 / vfps, (min(gaps) / 2 + 1e-6) if gaps else 0.0)
FRACTIONS = (0.1, 0.4, 0.7, 0.95)


def frame_at(cap, t, vfps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * vfps)))
    ok, frame = cap.read()
    if not ok:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return img.resize((TILE_W, round(img.height * TILE_W / img.width)))


def box_on(d, box, color, tag, x0, y0, w, h, lw=3):
    """Draw a normalized box in the tile whose top-left corner is (x0, y0)."""
    if box is None:
        return
    d.rectangle([x0 + box[0] * w, y0 + box[1] * h, x0 + box[2] * w, y0 + box[3] * h], outline=color, width=lw)
    d.text((x0 + box[0] * w + 3, y0 + box[1] * h + 1), tag, fill=color, font=font(13))


def make(qid, runs, gt_entries, video_path, out_path):
    records = [(name, json.load(open(Path(d) / f"{qid}.json"))) for name, d in runs]
    gs, ge = records[0][1]["gt"]["temporal_range"]
    times = [gs + (ge - gs) * f for f in FRACTIONS]
    cap = cv2.VideoCapture(str(video_path))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    tiles = [frame_at(cap, t, vfps) for t in times]
    gt_tol = gt_tolerance(gt_entries, vfps)
    cap.release()
    tw, th = tiles[0].size
    head_h, row_h = 58, th + 26
    canvas = Image.new("RGB", (tw * len(times), head_h + row_h * len(records)), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), f"{qid}: \"{records[0][1]['text']}\"", fill=(0, 0, 0), font=font(17))
    d.text((8, 31), f"ground truth (green) {gs:.1f}-{ge:.1f} s    prediction (red)", fill=(60, 60, 60), font=font(15))
    for r, (name, rec) in enumerate(records):
        y = head_h + r * row_h
        m = rec["metrics"]["fps1"]
        pi = rec["pred_interval_sec"]
        label = (f"{name}: {pi[0]:.0f}-{pi[1]:.0f} s" if pi else f"{name}: no answer") + \
                f"   t_IoU {m['t_IoU']:.2f}   v_IoU {m['v_IoU']:.2f}"
        d.text((8, y + 3), label, fill=(0, 0, 0), font=font(15))
        pred = [(p["time_sec"], p["box_xyxy"]) for p in rec["pred_scored"]]
        for c, (t, tile) in enumerate(zip(times, tiles)):
            x = c * tw
            canvas.paste(tile, (x, y + 24))
            td = ImageDraw.Draw(canvas)
            box_on(td, box_at(gt_entries, t, gt_tol), GT_COLOR, "GT", x, y + 24, tw, th)
            pb = box_at(pred, t, 0.5)
            box_on(td, pb, PRED_COLOR, "pred", x, y + 24, tw, th)
            td.text((x + 5, y + 24 + th - 18), f"t={t:.1f}s" + ("" if pb is not None else "  (no pred)"),
                    fill=(255, 255, 255), font=font(13))
    canvas.save(out_path, quality=85)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="animal")
    p.add_argument("--query_ids", nargs="+", required=True)
    p.add_argument("--run", action="append", required=True, help='"label=per_query_files_dir"')
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    runs = [tuple(r.split("=", 1)) for r in args.run]
    gt = load_gt(args.domain, set(args.query_ids))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for qid in args.query_ids:
        rec = json.load(open(Path(runs[0][1]) / f"{qid}.json"))
        video = PROJECT_ROOT / "data" / args.domain / "videos" / "clips" / rec["video"]
        out = out_dir / f"compare_{args.domain}_{qid}.jpg"
        make(qid, runs, gt.get(qid, []), video, out)
        print(out)


if __name__ == "__main__":
    main()
