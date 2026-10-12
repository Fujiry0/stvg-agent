#!/usr/bin/env bash
# Run the agent on the other domains (industry, sports, safety, surgery) and evaluate each.
# The tool server runs with the embedding model (Temporal.relevance uses it).
# Needs vLLM on :8100. Usage: nohup scripts/run_domains.sh > logs/run_domains.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
CONC=${CONC:-2}
ts() { date '+%F %T'; }
vllm_up() { curl -sf -m 5 http://127.0.0.1:8100/v1/models > /dev/null; }

start_tools() {  # restart the tool server and start a fresh watchdog
  rm -f logs/tools_watchdog.pause
  pgrep -f "[t]ools_watchdog.sh" | xargs -r kill
  scripts/restart_tools.sh || { echo "$(ts) tool server failed; stop"; exit 1; }
  setsid nohup scripts/tools_watchdog.sh >> logs/tools_watchdog.log 2>&1 < /dev/null &
}
run_agent_domain() {  # <domain>
  local d=$1 N=$1 out out2 out3
  python3 scripts/drop_llm_failures.py $N; python3 scripts/drop_exceptions.py $N
  echo "$(ts) $d agent start ($N, concurrency $CONC)"
  for i in 1 2 3; do
    vllm_up || { echo "$(ts) vLLM is down; stop"; exit 1; }
    scripts/run_agent.sh --run_name $N --resume --concurrency $CONC --domain $d >> logs/agent_$N.log 2>&1
    out=$(python3 scripts/drop_toolserver_failures.py $N); out2=$(python3 scripts/drop_exceptions.py $N); out3=$(python3 scripts/drop_llm_failures.py $N)
    echo "$(ts) $out | $out2 | $out3" >> logs/agent_$N.log
    echo "$out" | grep -q ": 0 queries" && echo "$out2" | grep -q "dropped 0 " && echo "$out3" | grep -q ": 0 queries" && break
  done
  scripts/eval_st.sh $N results/agent/$N/spatio_temporal/${N}_$d.json > logs/eval_$N.log 2>&1
  python3 scripts/split_per_query.py \
    --per_query results/eval/$N/per_query/${N}_${d}_with_per_query_metrics_spatio_temporal.json \
    --tubes data/$d/vidi_eval/results/${N}_spatio_temporal/tubes.csv \
    --out_dir results/eval/$N/per_query_files >> logs/eval_$N.log 2>&1
  echo "$(ts) $d agent done"
}

vllm_up || { echo "$(ts) vLLM is down; stop"; exit 1; }
start_tools
for d in industry sports safety surgery; do run_agent_domain $d; done
echo "$(ts) all done"
