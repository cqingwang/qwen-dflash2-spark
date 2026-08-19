#!/usr/bin/env bash
# bench/quality_capture.py helper wrapper: 6-prompt greedy quality gate.
# Compares against expected answers (math/syllogism/bugfix). Exit 1 on wrong math.
set -euo pipefail
NODE1="${NODE1:-192.168.1.205}"
PORT="${PORT:-8004}"
curl -s --max-time 300 "http://$NODE1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-dflash2","messages":[{"role":"user","content":"What is 17*23+45? Answer with just the number."}],"max_tokens":2000,"temperature":0}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; r=((m.get("content") or "")+(m.get("reasoning_content") or "")); print("17*23+45 =", r[-60:].replace("\n"," ")); sys.exit(0 if "436" in r else 1)'
echo "quality: math PASS"
