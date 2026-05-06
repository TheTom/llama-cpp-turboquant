#!/usr/bin/env bash
# Aggregated throughput benchmark across N parallel /v1/chat/completions requests.
# Usage: PARALLEL=4 N_PREDICT=200 ./scripts/bench-parallel.sh
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
PARALLEL="${PARALLEL:-4}"
N_PREDICT="${N_PREDICT:-200}"
PROMPT="${PROMPT:-Write a long article about the history of artificial intelligence and its impact on modern society. Include detailed sections on early pioneers, key milestones, and future predictions. Be thorough and at least 200 words.}"

REQ_BODY=$(cat <<EOF
{
  "model": "gemma-4",
  "messages": [{"role":"user","content":"$PROMPT"}],
  "max_tokens": $N_PREDICT,
  "temperature": 0,
  "stream": false
}
EOF
)

echo "info: launching $PARALLEL parallel requests, max_tokens=$N_PREDICT"

t0=$(python3 -c 'import time; print(time.time())')

PIDS=()
TMPDIR=$(mktemp -d)
for i in $(seq 1 "$PARALLEL"); do
  curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$REQ_BODY" > "$TMPDIR/r${i}.json" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

t1=$(python3 -c 'import time; print(time.time())')
WALL_S=$(python3 -c "print(${t1} - ${t0})")

TOTAL_TOKENS=0
for i in $(seq 1 "$PARALLEL"); do
  TOK=$(python3 -c "import json,sys; d=json.load(open('$TMPDIR/r${i}.json')); print(d.get('usage',{}).get('completion_tokens', 0))")
  echo "  req $i: completion_tokens=$TOK"
  TOTAL_TOKENS=$((TOTAL_TOKENS + TOK))
done

AGG_TPS=$(python3 -c "print(f'{${TOTAL_TOKENS} / ${WALL_S}:.2f}')")
PER_SEQ=$(python3 -c "print(f'{${TOTAL_TOKENS} / ${WALL_S} / ${PARALLEL}:.2f}')")

echo
echo "==== AGGREGATED RESULTS ===="
echo "  parallel:        $PARALLEL"
echo "  wall_time_s:     $WALL_S"
echo "  total_tokens:    $TOTAL_TOKENS"
echo "  aggregated_tps:  $AGG_TPS"
echo "  per_seq_tps:     $PER_SEQ"
echo
rm -rf "$TMPDIR"
