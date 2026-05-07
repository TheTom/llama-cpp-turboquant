#!/usr/bin/env bash
# Gemma 4 E4B Edge target only (no MTP / no assistant draft).
# Override MAIN_GGUF or LLAMA_SERVER as needed.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${LLAMA_SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-${ROOT}/.scratch/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf}"

CTX="${CTX:-16384}"
NGL="${NGL:-99}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-turbo3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FA="${FA:-on}"
TEMP="${TEMP:-}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"
ENABLE_SLOTS="${ENABLE_SLOTS:-1}"
LOG_TIMESTAMPS="${LOG_TIMESTAMPS:-1}"
LOG_PREFIX="${LOG_PREFIX:-1}"
NO_WARMUP="${NO_WARMUP:-0}"
PARALLEL="${PARALLEL:-1}"

if [[ ! -f "$SERVER" ]]; then
  echo "error: missing ${SERVER} (build with: cmake --build build --target llama-server)" >&2
  exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: ${MAIN}" >&2
  exit 1
fi

ARGS=(
  -m "$MAIN"
  -c "$CTX"
  -ngl "$NGL"
  -ctk "$CTK"
  -ctv "$CTV"
  -fa "$FA"
  --host "$HOST"
  --port "$PORT"
  --parallel "$PARALLEL"
  -np "$PARALLEL"
  --cont-batching
)

if [[ -n "$TEMP" ]]; then
  ARGS+=(--temp "$TEMP")
fi

[[ "$ENABLE_METRICS"  != "0" ]] && ARGS+=(--metrics)
[[ "$ENABLE_SLOTS"    != "0" ]] && ARGS+=(--slots)
[[ "$LOG_TIMESTAMPS"  != "0" ]] && ARGS+=(--log-timestamps)
[[ "$LOG_PREFIX"      != "0" ]] && ARGS+=(--log-prefix)
[[ "$NO_WARMUP"       != "0" ]] && ARGS+=(--no-warmup)

echo "info: baseline E4B (no MTP) CTX=${CTX} NGL=${NGL} FA=${FA} CTK=${CTK}" >&2
echo "info: MAIN=${MAIN}" >&2
exec "$SERVER" "${ARGS[@]}" "$@"
