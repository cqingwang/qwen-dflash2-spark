#!/usr/bin/env bash
# Build the vLLM image with DFlash2 support (PR vllm-project/vllm#52816).
# The pinned commit only changes Python runtime files. Keep the official image's
# prebuilt native extensions and overlay the fork's changed files directly.
set -euo pipefail
cd "$(dirname "$0")"

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.27.1-aarch64}"
FORK="${FORK:-https://github.com/z-lab/vllm-fork.git}"
BRANCH="${BRANCH:-subsir/upstream-dflash2}"
COMMIT="${COMMIT:-19c9351904df4c63042671bc67a866ca48dc7d6f}"
TAG="${TAG:-local/vllm-dflash2-pr52816:v4}"
PROXY_URL="${PROXY_URL:-http://192.168.2.114:7890}"

# GitHub checkout runs on the host because the minimal base image has no git.
# Set both cases because git/libcurl and Python tooling do not always read the
# same proxy variable spelling.
export HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL" ALL_PROXY="$PROXY_URL"
export http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" all_proxy="$PROXY_URL"

docker pull "$BASE_IMAGE"

# 1) Checkout on the host; the minimal vLLM image does not ship git.
SRC_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vllm-dflash2-src.XXXXXX")"
trap 'chmod -R u+rwX "$SRC_DIR" 2>/dev/null || true; rm -rf "$SRC_DIR"' EXIT
git clone "$FORK" "$SRC_DIR"
git -C "$SRC_DIR" checkout "$COMMIT"

# 2) Package only files changed by the pinned DFlash2 commit. This avoids
# recompiling the 468 native targets already shipped in the official image.
git -C "$SRC_DIR" diff-tree --no-commit-id --name-only -r "$COMMIT" \
  | awk '/^vllm\// && $0 != "vllm/config/vllm.py" { print }' \
  | tar czf o.tgz -C "$SRC_DIR" -T -

# 3) overlay built tree onto a clean official image
docker build -f Dockerfile.df2 --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$TAG" .
docker rmi -f "$BASE_IMAGE-bak" 2>/dev/null || true
echo "built $TAG"
