# Limitations

What the agent does not do well, with numbers from our runs of the configuration in this repository (one run per domain,
fps1 evaluation, all queries of each domain). The failure modes themselves, counted over all 955 queries and with an example
of each, are in [examples.md, section 5](examples.md#5-where-it-fails); this page explains the causes behind them.

> ⬅ [Main README](../README.md) &nbsp;·&nbsp; [Examples](examples.md) &nbsp;·&nbsp; [Architecture](architecture.md)

---

## 1. Time: it is slow

An answer takes several minutes, against seconds for a single pass of the same model. Wall-clock time per query (median
and 90th percentile), measured with 2–5 queries running in parallel on one shared 48 GB GPU:

| Domain | Parallel queries | Median | 90th percentile |
|---|---|---|---|
| Animal | 5 | 11 min | 23 min |
| Industry | 2 | 15 min | 57 min |
| Sports | 2 | 11 min | 25 min |
| Surgery | 2 | 15 min | 37 min |
| Public Security | 2 | 7 min | 18 min |

Median time per phase (Animal and Sports, in seconds): planning 275–310, code execution 210–230 (SAM 3, VLM calls, clips),
the main VLM's own generation 90–130, waiting in the tool server's queue 5–35. The planner, which thinks over the key frames
and the whole-video clip before the first step, is the single largest part.

- The numbers depend on the hardware and on how many queries run at once: the tool server runs one request at a time, so
  more parallel queries mostly add queue time.

## 2. Time: short events

Most AnyGroundBench queries are short, and short intervals are where the temporal estimate is worst. The answer is a
tube with one box per second, so an interval is placed to the nearest second; a ±0.5 s error is a large part of a 1-second
event.

| Ground-truth interval | Queries | Mean t_IoU | Mean v_IoU |
|---|---|---|---|
| < 2 s | 401 | 0.22 | 0.08 |
| 2–5 s | 257 | 0.37 | 0.19 |
| 5–10 s | 97 | 0.35 | 0.21 |
| ≥ 10 s | 200 | 0.30 | 0.14 |

The median ground-truth interval is 0.8 s in Sports and 1.0 s in Industry, against 20.5 s in Surgery.

- **Finding the interval is the bottleneck.** Only 29% (Animal), 23% (Industry), 25% (Sports), 15% (Surgery) and 49%
  (Public Security) of the queries reach t_IoU ≥ 0.5.
- **Long procedural segments are hard to delimit.** In Surgery the median interval is 20 s, and the mean t_IoU is still 0.25
  (0.01–0.18 for the few intervals under 10 s).
- **The whole-video estimate is right in only 22% of the queries** (IoU ≥ 0.5 with the ground truth), and the agent does not
  always recover from a wrong one or keep a right one: see [examples.md, 5.1 and 5.2](examples.md#5-where-it-fails).

## 3. Space: the target is tracked, but not always the right one

When the interval is right (t_IoU ≥ 0.5), the mean v_IoU is 0.63 in Animal, 0.40 in Public Security, 0.38 in Industry, 0.21
in Surgery and **0.03 in Sports**.

- **Crowded scenes with small targets** (team sports: 93% of Sports queries have v_IoU below 0.05). SAM 3 returns many
  players, and the agent often tracks the ball or a teammate instead of the actor, or answers a team-level query with the box
  of one person.
- **Fine-grained instance choice** (Surgery: "own hands right", "other hands left", a given instrument): there are several
  similar objects and the agent has to pick one from a text description.
- **Text prompts that SAM 3 cannot resolve** (113 queries) make the agent fall back to a single box from `vlm.locate`; a
  loose box gives a loose track ([examples.md, 5.5](examples.md#5-where-it-fails)).

## 4. The agent itself

- **A 9B model writes the code**, and one in five steps ends in an error (wrong indexing of tracks and frames, wrong argument
  types for the tools, parsing a VLM's text). Each error costs a step, and the 30-step budget is not always enough.
- **When tracking is awkward it can give up on tracking** and build the answer from boxes it wrote by hand, or loop (re-watch
  clips, doubt the query) until the step limit.
- **One tube per query.** A query that describes several instances or several separate intervals is answered with one tube.

Counts, rates and examples of these: [examples.md, 5.3–5.5](examples.md#5-where-it-fails).
