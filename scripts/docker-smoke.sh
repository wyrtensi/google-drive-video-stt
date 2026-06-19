#!/usr/bin/env bash
#
# Container smoke check for the config-owned deployment.
#
# This is a MANUAL / CI check, not a pytest. It proves the deployment is
# self-contained without `.env`:
#   1. A fresh config initializes inside the mounted /app/data volume.
#   2. The generated config points at files that exist inside that volume.
#   3. Packaged prompts are reachable inside the image.
#
# Usage:
#   scripts/docker-smoke.sh [image]
set -euo pipefail

IMAGE="${1:-google-drive-video-stt:smoke}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Building image: $IMAGE"
docker build -t "$IMAGE" .

SMOKE_DATA_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$SMOKE_DATA_DIR"
}
trap cleanup EXIT

run_gdstt() {
  docker run --rm -v "$SMOKE_DATA_DIR:/app/data" "$IMAGE" gdstt "$@"
}

echo "==> Initializing config in a clean /app/data volume"
run_gdstt config init --force
run_gdstt config set stt.deepgram.api_key smoke-deepgram-key >/dev/null
run_gdstt config set openai.api_key smoke-openai-key >/dev/null

echo "==> Running gdstt doctor in the container"
output="$(run_gdstt doctor 2>&1)"
echo "$output"

echo "==> Verifying config path is under /app/data"
config_line="$(printf '%s\n' "$output" | grep '^config:' || true)"
if [[ "$config_line" != *"/app/data/config.yml"* ]]; then
  echo "FAIL: expected config under /app/data, got: ${config_line:-<none>}" >&2
  exit 1
fi

echo "==> Verifying provider validation works without external services"
docker run --rm -v "$SMOKE_DATA_DIR:/app/data" "$IMAGE" \
  python -c "from src.config import load_config; cfg = load_config(); assert cfg.deepgram_keyterms"

echo "==> Verifying packaged prompts are reachable inside the container"
docker run --rm "$IMAGE" \
  python -c "from src.presets import load_packaged_prompt; assert load_packaged_prompt('keypoints.md').strip()"

echo "==> OK: clean config-only Docker smoke passed"
