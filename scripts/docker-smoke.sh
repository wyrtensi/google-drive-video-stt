#!/usr/bin/env bash
#
# Container smoke check for the config-owned preset DAG deployment.
#
# This is a MANUAL / CI check, not a pytest. It proves the two deployment fixes
# this image depends on:
#   1. Config persists in the mounted volume: `gdstt doctor` reports a `config:`
#      path under /app/data (DATA_DIR=/app/data is baked into the image), so the
#      first-run .env->YAML migration and credentials/token land in ./data.
#   2. Packaged prompts are reachable: load_packaged_prompt('keypoints.md')
#      resolves inside the container, proving the prompts ship inside the
#      `src` package (no top-level assets/ copy needed).
#
# Usage:
#   scripts/docker-smoke.sh [image]
#
# Requires a local `.env` (used for the first-run migration) and a writable
# ./data directory. It only runs `gdstt doctor`, which never contacts Drive,
# Deepgram, or OpenAI and spends nothing.
set -euo pipefail

IMAGE="${1:-google-drive-video-stt:smoke}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Building image: $IMAGE"
docker build -t "$IMAGE" .

mkdir -p data
ENV_ARGS=()
if [[ -f .env ]]; then
  ENV_ARGS+=(--env-file .env)
else
  echo "warning: no .env found; doctor still runs but no migration source is present" >&2
fi

echo "==> Running gdstt doctor in the container"
# Capture stdout AND stderr so a failing run is debuggable. `set -e` would abort
# on a non-zero `doctor` exit before we echo the captured output, so guard the
# assignment with `|| doctor_status=$?` and echo unconditionally below.
doctor_status=0
output="$(docker run --rm -v "$PWD/data:/app/data" "${ENV_ARGS[@]}" "$IMAGE" gdstt doctor 2>&1)" || doctor_status=$?
echo "$output"
if [[ "$doctor_status" -ne 0 ]]; then
  echo "FAIL: gdstt doctor exited with status $doctor_status (output above)" >&2
  exit "$doctor_status"
fi

echo "==> Verifying config path is under /app/data"
config_line="$(printf '%s\n' "$output" | grep '^config:' || true)"
if [[ "$config_line" != *"/app/data/config.yml"* ]]; then
  echo "FAIL: expected config under /app/data, got: ${config_line:-<none>}" >&2
  exit 1
fi

# Whether `keypoints` shows up in `doctor`'s preset DAG is a config property (the
# .env->YAML migration only enables it when OPENAI_KEYPOINTS=true), NOT a packaging
# property - so it can't prove prompts ship in the image. Instead load the prompt
# directly inside the container: this fails iff the packaged asset is unreachable.
echo "==> Verifying packaged prompts are reachable inside the container"
if ! docker run --rm "$IMAGE" \
  python -c "from src.presets import load_packaged_prompt; assert load_packaged_prompt('keypoints.md').strip()"; then
  echo "FAIL: packaged prompt 'keypoints.md' unreachable inside the container" >&2
  exit 1
fi

echo "==> OK: config persists under /app/data and packaged prompts load"
