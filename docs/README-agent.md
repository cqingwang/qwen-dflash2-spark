# Agent runbook

For AI agents operating this stack. Humans: see README.md first.

## Fast paths

**Is it up?**
```bash
curl -s -o /dev/null -w '%{http_code}\n' http://NODE1:8004/health   # want 200
```

**Bring the pair up**
```bash
bash scripts/04-launch-pair.sh    # r1 first, then r0; health ≤ 7.5 min
```

**Bring it down**
```bash
ssh NODE2 'docker rm -f qwen38-dflash2-tp2-r1' ; ssh NODE1 'docker rm -f qwen38-dflash2-tp2-r0'
```

**Quick health of both ranks**
```bash
ssh NODE1 'docker logs --tail 3 qwen38-dflash2-tp2-r0 2>&1' | tail -3
ssh NODE2 'docker logs --tail 3 qwen38-dflash2-tp2-r1 2>&1' | tail -3
```

**Bench once**
```bash
python3 bench/edit_bench.py http://NODE1:8004/v1 qwen3.8-27b-dflash2
```

## Failure signatures → root causes

| Symptom | Root cause | Fix |
|---|---|---|
| boot dies: unquantized LM head gate | lm_head quantized (wrong checkpoint) | run 02 + 03, relaunch |
| health 000 after >8 min | see logs; often wrong checkpoint or NCCL | `docker logs qwen38-dflash2-tp2-r0` |
| accept ~99.6%, tps low | K=7 (1 block) | set K=16 in launch |
| accept ~66%, tps below peak | K=24 | set K=16 |
| r0 up, r1 dead | r1 started after r0 | relaunch both, r1 first |

## Hard rules

- K must be a multiple of 8.
- rank1 before rank0, always.
- Kill stale background launcher jobs before relaunching (zombie once rm -f'd
  a healthy pair and relaunched the wrong checkpoint).
- Never edit config.json in place on hardlink copies — temp+os.replace.
- The original `unsloth-nvfp4/` checkpoint must stay pristine (it is the
  rollback). If corrupted, re-download via 00.
