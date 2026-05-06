#!/usr/bin/env bash
# Gemma 4 (or any GGUF) on llama-server with TurboQuant KV only — no draft / no -md.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${LLAMA_SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-${ROOT}/.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf}"

CTX="${CTX:-16384}"
NGL="${NGL:-99}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-turbo3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FA="${FA:-on}"

if [[ ! -f "$SERVER" ]]; then
  echo "error: missing ${SERVER} (build with: cmake --build build --target llama-server)" >&2
  exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: GGUF not found: ${MAIN}" >&2
  exit 1
fi

exec "$SERVER" \
  -m "$MAIN" \
  -c "$CTX" \
  -ngl "$NGL" \
  -ctk "$CTK" \
  -ctv "$CTV" \
  -fa "$FA" \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
