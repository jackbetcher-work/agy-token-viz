#!/usr/bin/env bash
# Shared Bash wrapper shim implementation for `agy` and `jetski` supporting `--visualize-tokens`.
# Sourced by `bin/agy` and `bin/jetski` with `SHIM_TARGET`, `SELF_REAL`, and `SHIM_DIR` set.
set -euo pipefail

PROJECT_ROOT="$(cd "$SHIM_DIR/.." && pwd -P)"

if [[ "$SHIM_TARGET" == "jetski" ]]; then
  VIZ_TAG="[jetski-token-viz]"
  REAL_BIN_ENV_NAME="JETSKI_REAL_BIN"
  INSTALL_HINT="Install Jetski CLI or set JETSKI_REAL_BIN=/path/to/jetski."
else
  VIZ_TAG="[agy-token-viz]"
  REAL_BIN_ENV_NAME="AGY_REAL_BIN"
  INSTALL_HINT="Install Antigravity CLI or set AGY_REAL_BIN=/path/to/agy."
fi

discover_real_cli() {
  local explicit_bin=""
  if [[ "$SHIM_TARGET" == "jetski" ]]; then
    explicit_bin="${JETSKI_REAL_BIN:-${TOKEN_VIZ_REAL_BIN:-}}"
  else
    explicit_bin="${AGY_REAL_BIN:-${TOKEN_VIZ_REAL_BIN:-}}"
  fi

  if [[ -n "$explicit_bin" && -x "$explicit_bin" ]]; then
    printf '%s\n' "$explicit_bin"
    return 0
  fi

  local entry candidate cand_real
  local IFS=':'
  for entry in ${PATH:-}; do
    [[ -z "$entry" || ! -d "$entry" ]] && continue
    candidate="${entry%/}/$SHIM_TARGET"
    if [[ -f "$candidate" && -x "$candidate" ]]; then
      cand_real="$(resolve_realpath "$candidate")"
      if [[ "$cand_real" != "$SELF_REAL" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done

  local fallbacks=()
  if [[ "$SHIM_TARGET" == "jetski" ]]; then
    fallbacks=(
      "/usr/local/bin/jetski"
      "/usr/local/google/jetski-cli/jetski-cli"
      "$HOME/.local/bin/jetski"
    )
  else
    fallbacks=(
      "$HOME/.local/bin/agy"
    )
  fi

  local fb
  for fb in "${fallbacks[@]}"; do
    if [[ -f "$fb" && -x "$fb" ]]; then
      cand_real="$(resolve_realpath "$fb")"
      if [[ "$cand_real" != "$SELF_REAL" ]]; then
        printf '%s\n' "$fb"
        return 0
      fi
    fi
  done

  return 1
}

resolve_python_bin() {
  if [[ "$SHIM_TARGET" == "jetski" ]]; then
    printf '%s\n' "${JETSKI_VIZ_PYTHON:-${TOKEN_VIZ_PYTHON:-${AGY_VIZ_PYTHON:-$PROJECT_ROOT/.venv/bin/python}}}"
  else
    printf '%s\n' "${AGY_VIZ_PYTHON:-${TOKEN_VIZ_PYTHON:-$PROJECT_ROOT/.venv/bin/python}}"
  fi
}

ENABLE_VIZ=0
PRELOAD_ONLY=0
VIZ_ARGS=()
CLI_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --visualize-tokens)
      ENABLE_VIZ=1
      shift
      ;;
    --no-visualize-tokens)
      ENABLE_VIZ=0
      shift
      ;;
    --viz-preload-only)
      ENABLE_VIZ=1
      PRELOAD_ONLY=1
      VIZ_ARGS+=("$1")
      shift
      ;;
    --viz-include-history|--viz-heuristic-only)
      ENABLE_VIZ=1
      VIZ_ARGS+=("$1")
      shift
      ;;
    --viz-model|--viz-model-file|--viz-model-dir|--viz-categories|--viz-config|--viz-target|--viz-data-dir|--viz-bin|--viz-agy-bin|--viz-jetski-bin)
      ENABLE_VIZ=1
      if [[ $# -lt 2 ]]; then
        echo "$VIZ_TAG Error: $1 requires a value." >&2
        exit 2
      fi
      VIZ_ARGS+=("$1" "$2")
      shift 2
      ;;
    --viz-model=*|--viz-model-file=*|--viz-model-dir=*|--viz-categories=*|--viz-config=*|--viz-target=*|--viz-data-dir=*|--viz-bin=*|--viz-agy-bin=*|--viz-jetski-bin=*)
      ENABLE_VIZ=1
      VIZ_ARGS+=("$1")
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        CLI_ARGS+=("$1")
        shift
      done
      ;;
    *)
      CLI_ARGS+=("$1")
      shift
      ;;
  esac
done

# Allow preloading weights even if the target CLI binary is not installed
if [[ "$PRELOAD_ONLY" -eq 1 ]]; then
  PYTHON_BIN="$(resolve_python_bin)"
  PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    exec "$PYTHON_BIN" -m agy_token_viz.cli \
      --viz-target "$SHIM_TARGET" \
      ${VIZ_ARGS[@]+"${VIZ_ARGS[@]}"}
fi

if ! REAL_CLI="$(discover_real_cli)"; then
  echo "$VIZ_TAG Error: Could not locate the underlying '$SHIM_TARGET' CLI binary." >&2
  echo "$INSTALL_HINT" >&2
  exit 127
fi

# Bypass visualizer for non-session CLI subcommands & help/metadata flags
if [[ "${#CLI_ARGS[@]}" -gt 0 ]]; then
  for a in "${CLI_ARGS[@]}"; do
    if [[ "$a" == "-h" || "$a" == "--help" || "$a" == "--version" || "$a" == "--stamp" || "$a" == "--bg-updater" ]]; then
      exec "$REAL_CLI" "${CLI_ARGS[@]}"
    fi
  done

  SUBCMD_IDX=0
  while [[ "$SUBCMD_IDX" -lt "${#CLI_ARGS[@]}" ]]; do
    case "${CLI_ARGS[$SUBCMD_IDX]}" in
      --dangerously-skip-permissions|--sandbox|--new-project|--disable-slash-commands|--remote-agent|--remote-control|--isolated_env|--standalone|--debug)
        SUBCMD_IDX=$((SUBCMD_IDX + 1))
        ;;
      *)
        break
        ;;
    esac
  done

  if [[ "$SUBCMD_IDX" -lt "${#CLI_ARGS[@]}" ]]; then
    case "${CLI_ARGS[$SUBCMD_IDX]}" in
      help|models|mcp|plugin|plugins|agent|agents|changelog|update|install|mic-serve|acp|remote-control|chatski|hub|web)
        exec "$REAL_CLI" "${CLI_ARGS[@]}"
        ;;
      swarm)
        NEXT_IDX=$((SUBCMD_IDX + 1))
        if [[ "$NEXT_IDX" -ge "${#CLI_ARGS[@]}" || "${CLI_ARGS[$NEXT_IDX]}" != "join" ]]; then
          exec "$REAL_CLI" "${CLI_ARGS[@]}"
        fi
        ;;
    esac
  fi
fi

# Zero-overhead pass-through when --visualize-tokens is not requested
if [[ "$ENABLE_VIZ" -eq 0 ]]; then
  exec "$REAL_CLI" ${CLI_ARGS[@]+"${CLI_ARGS[@]}"}
fi

PYTHON_BIN="$(resolve_python_bin)"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "$VIZ_TAG Error: Python runtime not found at $PYTHON_BIN" >&2
  exit 1
fi

export "$REAL_BIN_ENV_NAME=$REAL_CLI"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m agy_token_viz.cli \
  --viz-target "$SHIM_TARGET" \
  ${VIZ_ARGS[@]+"${VIZ_ARGS[@]}"} \
  -- \
  ${CLI_ARGS[@]+"${CLI_ARGS[@]}"}
