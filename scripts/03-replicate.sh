#!/usr/bin/env bash
# Replicate the bf16head checkpoint + draft dir to node 2 (once).
# Run from node 1 after 02. Adjust NODE2.
set -euo pipefail
BASE="${BASE:-/home/admin/qwen-repro}"
NODE2="${NODE2:-192.168.1.206}"
rsync -a --info=stats1 "$BASE/unsloth-nvfp4-bf16head" "$NODE2:$BASE/"
rsync -a --info=stats1 "$BASE/dflash2" "$NODE2:$BASE/"
echo "replicated to $NODE2"
