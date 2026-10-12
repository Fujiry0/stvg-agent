#!/usr/bin/env bash
# Zero-shot Qwen3.5-9B (a single pass, no tools) through the vLLM server, then score it with the benchmark.
# 1 fps, at most 32 frames, thinking off, greedy. Needs vLLM on :8100 (scripts/serve_vllm.sh).
# Usage: scripts/run_zeroshot.sh [domain ...]   (default: animal)
#        scripts/run_zeroshot.sh animal industry sports surgery safety
#        QUERY_IDS="test_28 test_121" scripts/run_zeroshot.sh animal    # only these queries (scored with --subset)
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p logs
domains=("$@"); [ ${#domains[@]} -eq 0 ] && domains=(animal)
for d in "${domains[@]}"; do
  envs/agent/.venv/bin/python baselines/single_pass_vllm.py --domain_name "$d" --query_max_frames_num 32 \
    --concurrency 8 ${QUERY_IDS:+--target_query_ids $QUERY_IDS} > "logs/zeroshot_$d.log" 2>&1
  pred=$(ls -t results/baselines/qwen3.5-9b-vllm/spatio_temporal/*_"$d".json 2>/dev/null | head -1)
  m=zeroshot_$d
  [ -n "$pred" ] && scripts/eval_st.sh ${QUERY_IDS:+--subset} "$m" "$pred" > "logs/eval_zeroshot_$d.log" 2>&1
  pq=$(ls results/eval/$m/per_query/*with_per_query_metrics_spatio_temporal.json 2>/dev/null | head -1)
  [ -n "$pq" ] && python3 scripts/split_per_query.py --per_query "$pq" \
    --tubes "data/$d/vidi_eval/results/${m}_spatio_temporal/tubes.csv" \
    --out_dir "results/eval/$m/per_query_files" >> "logs/eval_zeroshot_$d.log" 2>&1
  echo "$(date '+%F %T') $d done: results/eval/$m"
done
