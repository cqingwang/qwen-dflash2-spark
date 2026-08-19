#!/usr/bin/env bash
# Dequantize lm_head to BF16 in a copy of the NVFP4 checkpoint (hardlink-safe).
# Run ONCE on one node (r1 gets it via rsync in 03).
# Needs python3+torch+safetensors. If the host lacks torch, run inside the image:
#   BASE=/home/admin/qwen-repro bash 02-make-bf16head.sh   # host with torch
# or via container:
#   docker run --rm -v /home/admin/qwen-repro:/work -v $PWD/scripts:/s \
#     --entrypoint bash local/vllm-dflash2-pr52816:v2 \
#     -c 'SRC=/work/unsloth-nvfp4 DST=/work/unsloth-nvfp4-bf16head python3 /s/make_bf16head.py'
set -euo pipefail
BASE="${BASE:-/home/admin/qwen-repro}"
export BASE
python3 "$(dirname "$0")/make_bf16head.py"
