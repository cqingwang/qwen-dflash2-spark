#!/usr/bin/env bash
# Start the 2-node TP2 DFlash2 serving pair (rank0 = master on NODE1).
# Usage: PORT=8004 NODE1=192.168.1.205 NODE2=192.168.1.206 bash 04-launch-pair.sh
# Health ~4-5 min. Serves model name qwen3.8-27b-dflash2 on $PORT (NODE1).
# NOTE: qwen-dflash2-rank.sh is copied to $BASE by 05-install-assets.sh; if you
# skipped that, run: cp scripts/qwen-dflash2-rank.sh $BASE/ first.
set -euo pipefail
cd "$(dirname "$0")"

NODE1="${NODE1:-192.168.1.205}"
NODE2="${NODE2:-192.168.1.206}"
PORT="${PORT:-8004}"
MP="${MP:-29577}"
K="${K:-16}"
BASE="${BASE:-/home/admin/qwen-repro}"
MODEL_DIR="$BASE/unsloth-nvfp4-bf16head"
DRAFT_DIR="${DRAFT_DIR:-$BASE/dflash2}"

# K must be a multiple of the draft block_size (8). 16 is the measured peak.
if [ $((K % 8)) -ne 0 ]; then
  echo "ERROR: K=$K not a multiple of draft block_size 8" >&2
  exit 1
fi

# install/refresh the rank script on both nodes with current K
sed "s/num_speculative_tokens\":[0-9]*/num_speculative_tokens\":$K/" \
  scripts/qwen-dflash2-rank.sh > /tmp/qwen-dflash2-rank.sh
scp -q /tmp/qwen-dflash2-rank.sh "$NODE1:$BASE/qwen-dflash2-rank.sh"
scp -q /tmp/qwen-dflash2-rank.sh "$NODE2:$BASE/qwen-dflash2-rank.sh"

ssh -o ConnectTimeout=8 "$NODE2" "docker rm -f qwen38-dflash2-tp2-r1 2>/dev/null || true"
ssh -o ConnectTimeout=8 "$NODE1" "docker rm -f qwen38-dflash2-tp2-r0 2>/dev/null || true"

# rank1 (headless worker) first, then rank0 (master w/ API server)
ssh -o ConnectTimeout=8 "$NODE2" "RANK=1 HOST_IP=$NODE2 MASTER_ADDR=$NODE1 MASTER_PORT=$MP \
  MODEL_DIR=$MODEL_DIR DRAFT_DIR=$DRAFT_DIR PORT=$PORT bash $BASE/qwen-dflash2-rank.sh"
ssh -o ConnectTimeout=8 "$NODE1" "RANK=0 HOST_IP=$NODE1 MASTER_ADDR=$NODE1 MASTER_PORT=$MP \
  MODEL_DIR=$MODEL_DIR DRAFT_DIR=$DRAFT_DIR PORT=$PORT bash $BASE/qwen-dflash2-rank.sh"

c=000
for i in $(seq 1 90); do
  sleep 5
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://$NODE1:$PORT/health" || true)
  [ "$c" = "200" ] && break
done
echo "health=$c after $((i*5))s"
