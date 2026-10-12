"""Where does the time go? Summarize the per-query timing log of agent runs.

Reads results/agent/<run>/traj/*/trajectory.json ("timing": wall seconds per phase, "tool:<name>" per tool /
VLM method, "tool:server/<path>" for tool-server round trips and "tool_queue_wait" for the time spent waiting
behind other workers' tool-server calls) and prints a table per run.

Usage: python3 scripts/timing_report.py timing_full3 timing_6fps3 [--out results/timing_report.md]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = ["init", "plan", "llm_main", "exec", "anchor", "fallback", "other"]


def load(run):
    out = {}
    for f in sorted((ROOT / "results/agent" / run / "traj").glob("*/trajectory.json")):
        t = json.load(open(f))
        if t.get("timing"):
            out[f.parent.name] = t
    return out


def report(run):
    trajs = load(run)
    if not trajs:
        return f"## {run}\n\n(no trajectories with a timing log)\n"
    lines = [f"## {run}", ""]
    head = ["query", "total", "steps"] + PHASES + ["tool queue wait"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tools = defaultdict(float)
    calls = defaultdict(int)
    totals = defaultdict(float)
    for q, t in trajs.items():
        tm = t["timing"]
        lines.append(f"| {q} | {t['seconds']:.0f} | {t['step_count']} | "
                     + " | ".join(f"{tm.get(k, 0):.0f}" for k in PHASES) + f" | {tm.get('tool_queue_wait', 0):.0f} |")
        totals["total"] += t["seconds"]
        for k in PHASES + ["tool_queue_wait"]:
            totals[k] += tm.get(k, 0)
        for s in t["steps"]:
            for e in s.get("tool_timings") or []:
                tools[e["name"]] += e["sec"]
                calls[e["name"]] += 1
    lines.append(f"| **sum** | {totals['total']:.0f} | | " + " | ".join(f"{totals[k]:.0f}" for k in PHASES)
                 + f" | {totals['tool_queue_wait']:.0f} |")
    lines += ["", "Share of total: " + ", ".join(f"{k} {100 * totals[k] / totals['total']:.0f}%" for k in PHASES
                                                 if totals[k] > 0), ""]
    lines += ["| tool / VLM call | calls | seconds | s/call | share of exec |", "|---|---|---|---|---|"]
    for name, sec in sorted(tools.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {calls[name]} | {sec:.0f} | {sec / calls[name]:.1f} | "
                     f"{100 * sec / max(totals['exec'], 1e-9):.0f}% |")
    llm = defaultdict(list)
    for t in trajs.values():
        entries = list(t.get("host_llm_calls") or [])
        for s in t["steps"]:
            entries += [e for e in s.get("tool_timings") or [] if e["name"].startswith("llm:")]
        for e in entries:
            if e["name"].startswith("llm:"):
                llm[e["name"][4:]].append(e)
    if llm:
        lines += ["", "| LLM call (caller) | calls | seconds | s/call | completion mean / max | hit max_tokens |",
                  "|---|---|---|---|---|---|"]
        for tag, es in sorted(llm.items(), key=lambda kv: -sum(e["sec"] for e in kv[1])):
            comp = [e.get("completion") or 0 for e in es]
            lines.append(f"| {tag} | {len(es)} | {sum(e['sec'] for e in es):.0f} | {sum(e['sec'] for e in es) / len(es):.1f} | "
                         f"{sum(comp) / len(comp):.0f} / {max(comp)} | {sum(e.get('finish') == 'length' for e in es)} |")
    lines.append("")
    lines.append("(`tools.X.method` / `vlm.method` = whole call inside the kernel; `server/...` = the HTTP round trip "
                 "to the tool server within it, including the queue wait.)")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    text = "\n".join(report(r) for r in args.runs)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
