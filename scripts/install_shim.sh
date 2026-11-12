#!/usr/bin/env bash
# Helper to wire `token_usage_viz/bin` ahead of `agy` and `jetski` on PATH so `--visualize-tokens` works everywhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
BIN_DIR="$PROJECT_ROOT/bin"

DRY_RUN=0
RC_FILE="${HOME}/.zshrc"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--print)
      DRY_RUN=1
      shift
      ;;
    --rc-file)
      RC_FILE="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--dry-run] [--rc-file ~/.zshrc]

Configures your shell so \`agy --visualize-tokens\` and \`jetski --visualize-tokens\`
invoke the token visualizer shims while normal \`agy\` and \`jetski\` commands pass
straight through to the underlying binaries.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

SNIPPET_MARKER="# >>> agy-token-viz >>>"
SNIPPET_END="# <<< agy-token-viz <<<"
read -r -d '' SNIPPET <<EOF || true
$SNIPPET_MARKER
export PATH="$BIN_DIR:\$PATH"
$SNIPPET_END
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%s\n' "$SNIPPET"
  exit 0
fi

touch "$RC_FILE"
if grep -Fq "$SNIPPET_MARKER" "$RC_FILE"; then
  echo "[agy-token-viz] Shim already configured in $RC_FILE"
else
  printf '\n%s\n' "$SNIPPET" >> "$RC_FILE"
  echo "[agy-token-viz] Added $BIN_DIR to PATH in $RC_FILE"
fi
echo "[agy-token-viz] Run 'source $RC_FILE' (or open a new terminal), then try:"
echo "  agy --visualize-tokens"
echo "  jetski --visualize-tokens"
