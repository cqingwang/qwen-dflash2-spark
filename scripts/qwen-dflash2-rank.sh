#!/usr/bin/env bash
set -euo pipefail
: "${RANK:?}" "${HOST_IP:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}" "${MODEL_DIR:?}" "${DRAFT_DIR:?}" "${PORT:?}"
name="qwen38-dflash2-tp2-r${RANK}"
headless=(); [[ "$RANK" != 0 ]] && headless=(--headless)
docker rm -f "$name" >/dev/null 2>&1 || true
exec docker run -d --name "$name" --restart=no --gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
-e VLLM_HOST_IP="$HOST_IP" -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
-e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e TP_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_NET=IB -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_GID_INDEX=0 -e NCCL_IB_TIMEOUT=22 -e NCCL_IB_RETRY_CNT=7 -e NCCL_CROSS_NIC=0 -e NCCL_IB_QPS_PER_CONNECTION=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
-v "$MODEL_DIR:/model:ro" -v "$DRAFT_DIR:/draft:ro" -v /home/admin/qwen-repro/vllm-cache:/root/.cache/vllm \
local/vllm-dflash2-pr52816:v2 /model --served-model-name qwen3.8-27b-dflash2 --host 0.0.0.0 --port "$PORT" --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --distributed-executor-backend mp --nnodes 2 --node-rank "$RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" "${headless[@]}" --max-model-len 262144 --gpu-memory-utilization 0.80 --max-num-batched-tokens 16384 --enable-prefix-caching --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice --limit-mm-per-prompt.image 2 --limit-mm-per-prompt.video 0 --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":16}'
