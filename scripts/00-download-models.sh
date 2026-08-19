#!/usr/bin/env bash
# Download target + draft checkpoints for DFlash2 serving.
# Run once per node (both DGX Sparks need both dirs).
set -euo pipefail
BASE="${BASE:-/home/admin/qwen-repro}"
mkdir -p "$BASE"

pip install -q "huggingface_hub[cli]" 2>/dev/null || true

# Target: NVFP4 main model (lm_head is quantized here; 02-make-bf16head.sh fixes it)
[ -d "$BASE/unsloth-nvfp4" ] || hf download unsloth/Qwen3.8-27B-NVFP4 \
  --local-dir "$BASE/unsloth-nvfp4"

# Draft: z-lab DFlash2 drafter (1.92B, block_size 8)
[ -d "$BASE/dflash2" ] || hf download z-lab/Qwen3.8-27B-DFlash2 \
  --local-dir "$BASE/dflash2"

du -sh "$BASE"/unsloth-nvfp4 "$BASE"/dflash2
