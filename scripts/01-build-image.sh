#!/usr/bin/env bash
# Build the vLLM image with DFlash2 support (PR vllm-project/vllm#52816).
# Strategy: build the fork in-place inside the official aarch64 image
# (compiler + CUDA deps already present), tar the built vllm/ tree,
# overlay onto a fresh official image. ~1-2h on a GB10.
set -euo pipefail
cd "$(dirname "$0")"

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.27.1-aarch64}"
FORK="${FORK:-https://github.com/z-lab/vllm-fork.git}"
BRANCH="${BRANCH:-subsir/upstream-dflash2}"
COMMIT="${COMMIT:-19c9351904df4c63042671bc67a866ca48dc7d6f}"
TAG="${TAG:-local/vllm-dflash2-pr52816:v2}"

docker pull "$BASE_IMAGE"

# 1) build fork in-place -> produces compiled .so inside the source tree
docker run --rm -v "$PWD:/out" "$BASE_IMAGE" bash -exc '
  git clone '"$FORK"' /src
  cd /src && git checkout '"$COMMIT"'
  pip install -q ninja cmake
  pip install -ve . --no-deps
  tar czf /out/o.tgz -C /src/vllm .
'

# 2) overlay built tree onto a clean official image
docker build -f Dockerfile.df2 --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$TAG" .
docker rmi -f "$BASE_IMAGE-bak" 2>/dev/null || true
echo "built $TAG"
