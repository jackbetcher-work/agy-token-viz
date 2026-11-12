#!/usr/bin/env bash
# Pre-download and verify local Hugging Face GGUF weights for `agy --visualize-tokens` / `jetski --visualize-tokens`.
# Uses HF_TOKEN / HUGGING_FACE_HUB_TOKEN from environment if set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

exec "$PROJECT_ROOT/bin/agy" --viz-preload-only "$@"
