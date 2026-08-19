# Architecture

```
                    ┌───────────────────────────────┐
   client ──HTTP──▶ │  vLLM API server (rank 0)     │◀── http://NODE1:8004/v1
                    │  speculative worker: DFlash2  │
                    │  target: Qwen3.8-27B NVFP4    │
                    │  (lm_head BF16)               │
                    └───────────┬───────────────────┘
                                │ NCCL / Gloo (mp backend)
                                │ RoCE v2 over Mellanox ConnectX
                                │ master :29577
                    ┌───────────┴───────────────────┐
                    │  rank 1 (--headless)          │
                    │  same image + checkpoints     │
                    └───────────────────────────────┘
```

- **TP=2, nnodes=2**: each GB10 holds half the shards of the 27B NVFP4
  target plus the full 1.92B draft (draft is replicated, not sharded).
- **Allreduce**: NCCL over RoCE (`rocep1s0f1`, RoCEv2, AF_INET, gid 0).
  Full env set lives in `scripts/qwen-dflash2-rank.sh`.
- **Draft → target interaction**: the DFlash2 selector head runs TopK over
  the *target* `lm_head` logits — which is why lm_head must be unquantized
  BF16 (see README pitfalls #1–2).
- **Spec config**: `{"method":"dflash","model":"/draft","num_speculative_tokens":16}`
  — K=16 = 2 draft blocks of 8, measured GB10 optimum.

## Checkpoint surgery detail

`make_bf16head.py`:
1. hardlink-copy `unsloth-nvfp4` → `unsloth-nvfp4-bf16head` (fast, no 21 GB rewrite)
2. delete DST safetensors (breaks hardlinks for the file we rewrite)
3. load `model.safetensors`, pop `lm_head.weight` + `lm_head.weight_scale`,
   dequant `w * scale → bf16`, save back as single file
4. patch `config.json` (hardlink-safe temp+replace): strip `re:.*lm_head`
   from `config_groups.group_0.targets`, add `lm_head` to `ignore`

Net effect: `lm_head` runs as an unquantized BF16 GEMM; everything else stays
NVFP4/FP8 as shipped. Quality delta: none measurable (6/6 greedy match).

## Verification chain (what "done" means)

1. `/health` → 200 on NODE1:PORT
2. chat completion returns expected literal string
3. `edit_bench.py` C1 edit-heavy ≥ ~120 tok/s, accept ≥ ~90%
4. `07-quality-gate.sh` math check passes
5. (optional) LiteLLM route → endpoint e2e
