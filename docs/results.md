# Results & methodology

All numbers: 2× DGX Spark (GB10, 128 GB unified each), TP=2 over RoCE v2,
target `unsloth/Qwen3.8-27B-NVFP4` (bf16head variant), draft
`z-lab/Qwen3.8-27B-DFlash2` (1.92B, block 8), image
`local/vllm-dflash2-pr52816:v2` (vLLM PR 52816 @ `19c9351`),
`gpu_memory_utilization 0.80`, `max_model_len 262144`, prefix caching on.

## Benchmarks

### C1 single-stream edit-heavy (3000 tok, engine counters)

| Config | tok/s | accept | mean tok/pass |
|---|---|---|---|
| DSpark (MTP) K=14 | 91–95 | 68.6% | ~9.7 |
| DFlash2 K=7 | 73–74 | 99.6% | 7.0 |
| DFlash2 K=16 (×10 runs) | 131.5–134.8 | 93.1% | 15.89 |
| DFlash2 K=24 | 121–129 | 66.4% | 16.9 |

K=16 peak: mean accept 15.89/16 — the drafter is nearly perfect at 2 blocks;
at 3 blocks (K=24) acceptance falls to 66% and net tps drops despite more
tokens per pass.

### Quality (greedy, temperature 0)

6 prompts (arithmetic, geography, syllogism, code bugfix, code gen,
explanation), checked against pre-surgery baseline answers:

- 17×23+45 → 436 ✓
- Canberra population → ~490k ✓
- syllogism → correct ✓
- bugfix → `a + b` ✓
- code + explanation → clean ✓

**6/6 identical** — BF16 lm_head dequant introduces no measurable drift
under greedy decoding.

### Fresh-generation (low-acceptance) regime

edit-heavy is the optimistic end (output mostly recoverable from prompt);
fresh generation behaves differently (lower acceptance, lower tps) —
`edit_bench.py` reports both. Only compare C1-to-C1.

## Measurement provenance

- tps + acceptance read from vLLM engine counters via `edit_bench.py`
  (client-side timing cross-checked; matched within ~2%).
- Variance: 10 consecutive edit-heavy runs, 2026-08-19.
- E2E serving verified through LiteLLM (route `qwen3.8-mtp` →
  `100.83.32.14:8004`) with correct literal completion.

## How this compares to stock

Stock NVFP4 serving (no spec decode) on this hardware: ~62 tok/s C1.
DSpark K=14: 91–95. DFlash2 K=16: 131–135. Speedup vs stock: **2.1×**;
vs DSpark: **+43%**.
