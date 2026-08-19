#!/usr/bin/env bash
# Full-stack verification: health, one real generation, acceptance counters.
# Run after 04 from the repo root. Adjust NODE1/PORT as needed.
set -euo pipefail
NODE1="${NODE1:-192.168.1.205}"
PORT="${PORT:-8004}"
BASE_URL="http://$NODE1:$PORT/v1"
MODEL="qwen3.8-27b-dflash2"

echo "== health =="
curl -s -o /dev/null -w 'health: %{http_code}\n' --max-time 5 "http://$NODE1:$PORT/health"

echo "== generation =="
curl -s --max-time 120 "$BASE_URL/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with exactly: DFLASH2 OK"}],"max_tokens":50,"temperature":0}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("reply:", (m.get("content") or m.get("reasoning_content",""))[:120])'

echo "== bench (single run) =="
ssh -o ConnectTimeout=8 "$NODE1" "cd $BASE && python3 /tmp/edit_bench.py $BASE_URL $MODEL" 2>/dev/null || \
  python3 bench/edit_bench.py "$BASE_URL" "$MODEL"
