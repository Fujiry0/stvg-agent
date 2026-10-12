#!/usr/bin/env bash
# Evaluate spatio-temporal prediction JSONs with AnyGroundBench's Vidi-style evaluator.
# Usage: scripts/eval_st.sh [--subset] <model_label> <pred.json> [<pred.json> ...]
#   Filenames must end with _<domain>.json or _<domain>_<N>shot.json.
#   --subset: evaluate only the queries present in the predictions (e.g. a 5-query agent run);
#             without it, missing queries count as failures (full-benchmark numbers).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

extra=()
if [[ "${1:-}" == "--subset" ]]; then extra+=(--ignore-missing-pred); shift; fi
model=$1; shift
out_dir=results/eval/$model
$EVAL_PY "$ANYGROUND_ROOT/scripts/eval/run_spatio_temporal_grounding_bundle.py" \
  --task spatio_temporal \
  --model "$model" \
  --pred-jsons "$@" \
  --summary-csv "$out_dir/spatio_temporal_summary.csv" \
  --per-query-output-dir "$out_dir/per_query" \
  "${extra[@]}"
