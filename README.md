# Qwen3.8-27B DFlash2 on 2× DGX Spark (GB10) — 135 tok/s single-stream

> **Mirrors**: this repo is also on Hugging Face at
> [cfontes/qwen-dflash2-spark](https://huggingface.co/cfontes/qwen-dflash2-spark)
> (identical content). GitHub is canonical.

**One-stop, copy-paste reproducible stack**: run Qwen3.8-27B NVFP4 with DFlash2
block speculative decoding on two NVIDIA DGX Spark (GB10) nodes over RoCE,
hitting **124–135 tok/s single-stream (C1)** — a **+43% speedup** over DSpark
(MTP-style) speculative decoding at equal quality, verified end-to-end.

Everything needed is in this repo: build the vLLM image, fix the checkpoints,
launch the 2-node pair, and verify. **No external state assumed** beyond two
DGX Sparks with Docker + SSH, internet to pull the base image and checkpoints.

## TL;DR quickstart

```bash
git clone https://github.com/fattchris/qwen-dflash2-spark.git
cd qwen-dflash2-spark
# on BOTH nodes:                     ~30 min (bandwidth-bound)
bash scripts/00-download-models.sh
# on node 1 only:                    ~1–2 h (compiles vLLM fork)
bash scripts/01-build-image.sh
# on node 1 only:                    ~10 min
bash scripts/02-make-bf16head.sh
# from node 1:                       ~5 min  (rsync ~23 GB)
bash scripts/03-replicate.sh
# from anywhere with SSH to both:    ~4–5 min to health 200
bash scripts/04-launch-pair.sh
# verify generation + throughput:
bash scripts/06-verify.sh && bash scripts/07-quality-gate.sh
```

Result: OpenAI-compatible endpoint at `http://NODE1:8004/v1`, model name
`qwen3.8-27b-dflash2`, 131–135 tok/s typical on the
edit-heavy C1 benchmark.

---

## What this is

- **Target model**: `unsloth/Qwen3.8-27B-NVFP4` (FP4 weights, FP8 attention) —
  the bandwidth-floor baseline on GB10.
- **Draft model**: `z-lab/Qwen3.8-27B-DFlash2` (1.92B params, block_size 8,
  local convolution + candidate selector head).
- **Serving stack**: vLLM from **PR [vllm-project/vllm#52816](https://github.com/vllm-project/vllm/pull/52816)**
  ([z-lab/vllm-fork@19c9351](https://github.com/z-lab/vllm-fork/commit/19c9351904df4c63042671bc67a866ca48dc7d6f),
  branch `subsir/upstream-dflash2`), compiled aarch64 in the official
  `vllm/vllm-openai:0.27.1-aarch64` image, TP=2 across two Sparks via RoCE v2.
- **The one real fix** that makes the combo work: **dequantize `lm_head` to
  BF16** in a copy of the NVFP4 checkpoint. Every public NVFP4 Qwen3.8
  checkpoint quantizes `lm_head`, but DFlash2's candidate selector performs
  TopK over the **target** model's `lm_head` logits and hard-rejects
  quantized heads — so nothing out of the box works. `02-make-bf16head.sh`
  does the surgery + config fix atomically and hardlink-safely.

## Results (GB10 ×2, TP=2, NVFP4 target + 1.92B draft)

| Config | tok/s (C1 edit-heavy) | accept | mean tok/pass |
|---|---|---|---|
| DSpark (MTP) K=14 | 91–95 | 68.6% | ~9.7 |
| DFlash2 K=7 | 73–74 | **99.6%** | 7.0 |
| **DFlash2 K=16** | **124–135** | 93.1% | 15.9 |
| DFlash2 K=24 | 121–129 | 66.4% | 16.9 |

- K sweep on GB10: sweet spot is **K=16 (2 draft blocks)**. K=24 drops
  acceptance (66%) and net throughput. K must be a multiple of block_size 8.
- ×5 repeat variance: 131.5–134.8 tok/s, accept 93.1% every run.
- Quality: 6/6 greedy correct vs baseline (math/Canberra/syllogism/bugfix/code)
  — BF16 lm_head dequant introduces zero measurable drift.
- Aggregate/concurrent throughput scales differently — C1 numbers are
  single-stream only, and only comparable to other C1 numbers.

## Repo layout

```
scripts/00-download-models.sh   # hf download target + draft
scripts/01-build-image.sh       # compile z-lab vllm-fork inside official image
scripts/Dockerfile.df2          # overlay built vllm/ onto clean official image
scripts/02-make-bf16head.sh + make_bf16head.py   # the lm_head surgery
scripts/03-replicate.sh         # rsync checkpoints node1 → node2
scripts/04-launch-pair.sh       # TP2 launch, health-wait
scripts/06-verify.sh            # health + generation + bench
scripts/07-quality-gate.sh      # greedy math gate (17*23+45 == 436)
scripts/qwen-dflash2-rank.sh    # the actual docker run (single rank)
bench/edit_bench.py             # C1 edit-heavy/fresh benchmark (from engine counters)
bench/quality_capture.py        # 6-prompt greedy capture
bench/prod_tps.py               # simple client-side tps probe
docs/architecture.md            # how the pieces fit, RoCE env details
docs/README-agent.md            # runbook for AI agents operating this stack
docs/results.md                 # full benchmark tables + methodology
```

## Step-by-step (with what each does)

### 0. Prereqs
Two DGX Spark nodes with: Docker, passwordless SSH from your workstation and
node1→node2, internet. Defaults assume `admin@192.168.1.205` (node1/rank0) and
`admin@192.168.1.206` (node2/rank1); override with `NODE1= NODE2=` env vars.
Note the image only exists for aarch64 — this stack is GB10-specific.

### 1. Download models (~30 min, both nodes)
```bash
bash scripts/00-download-models.sh
```
Pulls `unsloth/Qwen3.8-27B-NVFP4` (~21 GB) and `z-lab/Qwen3.8-27B-DFlash2`
(~4 GB). Idempotent — safe to re-run.

### 2. Build image (~1–2 h, node1)
```bash
bash scripts/01-build-image.sh
```
Clones the z-lab vllm fork at the pinned PR commit, builds it in-place inside
the pulled official image (compiler + CUDA deps already present), tars the
built `vllm/` tree, and overlays it onto a fresh official image via
`Dockerfile.df2`. Tag: `local/vllm-dflash2-pr52816:v2`. Then `docker save` it
to node2 — see below.

Wait — cross-node image: `01` builds on node1 only. Move it to node2:
```bash
docker save local/vllm-dflash2-pr52816:v2 | gzip > /tmp/df2.tgz
scp /tmp/df2.tgz node2:/tmp/ && ssh node2 'gunzip -c /tmp/df2.tgz | docker load'
```
~22.5 GB; over a 10 GbE LAN ~30 min, over Tailscale slower.

### 3. lm_head surgery (~10 min, node1)
```bash
bash scripts/02-make-bf16head.sh
```
Creates `unsloth-nvfp4-bf16head/` — a hardlink copy of the NVFP4 checkpoint
with `lm_head.weight` dequantized (`w * scale → bf16`, vocab 151k × 4.1k
hidden) and the config fixed (see Pitfalls #2). The original checkpoint is
left pristine (verify with `fix_configs.py`'s proof mode).

### 4. Replicate to node2 (~5 min)
```bash
bash scripts/03-replicate.sh
```
rsyncs the bf16head checkpoint (uses rsync hardlink semantics where possible)
+ draft dir to node2.

### 4b. Container image to node2 — **don't skip** (see step 2 note).

### 5. Launch pair (~4–5 min to health)
```bash
bash scripts/04-launch-pair.sh
```
Installs the rank script (with your K) on both nodes, removes stale containers,
starts rank1 (headless) then rank0 (master/API), and health-polls up to 7.5 min.

### 6–7. Verify
```bash
bash scripts/06-verify.sh
bash scripts/07-quality-gate.sh
```
06 checks health, a real generation ("DFLASH2 OK"), and a single bench run.
07 is a greedy math gate: `17*23+45` must answer `436`.

## Pitfalls (learned the hard way — read before debugging)

1. **lm_head must be BF16** — DFlash2's selector TopKs over target logits; the
   image asserts `UnquantizedEmbeddingMethod`. Dequant via 02. (This is the
   headline fix; every public NVFP4 Qwen3.8 checkpoint ships a quantized
   lm_head.)
2. **Quant-config precedence**: `config_groups.group_0.targets` **overrides**
   the `ignore` list. Your `ignore: ["lm_head"]` does nothing while
   `re:.*lm_head` sits in `targets`. 02 strips it from targets AND adds it to
   ignore. (Cost me three failed boots to find.)
3. **Hardlink bleed**: `shutil.copytree(copy_function=os.link)` keeps the
   original and the copy sharing inodes — a later `json.dump` into
   `config.json` **rewrites the shared inode and corrupts the original
   checkpoint**. Always write-temp-then-`os.replace()` configs (02 does).
4. **Copy path exists**: `02` runs `copytree` into an existing dir → symlink
   loops. It removes `.safetensors` in DST first; if DST is partial, delete it
   and re-run.
4b. **Served model name** is exactly `qwen3.8-27b-dflash2` (dot-separated,
   set by `--served-model-name` in the rank script). Clients must match it.
5. **NCCL/RoCE env**: exact working set is in `qwen-dflash2-rank.sh`
   (`NCCL_IB_HCA=rocep1s0f1`, RoCE v2, `NCCL_CUMEM_ENABLE=0`,
   `VLLM_MARLIN_USE_ATOMIC_ADD==1` etc.). Don't hand-tune without capturing a
   baseline first.
6. **Health timing**: boot → 200 takes ~4–5 min. My first 650 s
   health-poll loop assumed faster; don't panic at 200 s.
7. **TP2 MPI-rank pairing**: rank1 must be up **before** rank0 starts serving
   (rank1 joins via `--headless`). 04 launches r1 first for this reason. If
   you restart rank0 only, expect a stuck handshake → restart both, r1 first.
8. **Stale background jobs**: if you automate around this stack, ensure old
   launcher jobs are dead before relaunching — a zombie once `docker rm -f`'d
   a healthy pair and relaunched the wrong checkpoint.
9. **launchd/kickstart gotcha** (Mac LiteLLM gateways): `launchctl kickstart -k`
   recycles only the worker; the master keeps the old config. Kill the master
   PID and let launchd respawn for config changes to take.
10. **K must be multiple of 8** (draft block_size). K=16 peak on GB10.

## Credit & lineage

- DFlash2: z-lab (draft weights + vLLM fork). Blog: inco.ai/blog/dflash2.
- NVFP4 target: unsloth/Qwen3.8-27B-NVFP4.
- vLLM PR 52816 by SubSir (z-lab), open as of 2026-08-19.
- This repo = integration engineering + the lm_head BF16 fix + GB10 K-sweep +
  2-node RoCE recipe. MIT licensed. Benchmarks by Chris Fontes.
