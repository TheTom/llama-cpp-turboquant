#!/usr/bin/env bash
# Run Gemma 4 (gemma-4-26B-A4B-it-UD-Q4_K_XL) on llama-server in one of 4 modes.
#
# Modes (named, first positional argument):
#   f16-mtp      KV-cache f16   + MTP draft (gemma-assistant-mtp-q4)
#   turbo3-mtp   KV-cache turbo3 + MTP draft
#   f16-base     KV-cache f16   (baseline, no MTP)
#   turbo3-base  KV-cache turbo3 (baseline, no MTP)
#
# Model files are resolved relative to this script's directory:
#   ./gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
#   ./gemma-assistant-mtp-q4.gguf
#
# Override via env: LLAMA_SERVER, MAIN_GGUF, DRAFT_GGUF, CTX, NGL, HOST, PORT.

set -euo pipefail

usage() {
  cat <<EOF
usage: $(basename "$0") <mode> [extra llama-server args...]

modes:
  f16-mtp       f16 KV cache + MTP draft
  turbo3-mtp    turbo3 KV cache + MTP draft
  f16-base      f16 KV cache, no speculative decoding
  turbo3-base   turbo3 KV cache, no speculative decoding

env overrides:
  LLAMA_SERVER  path to llama-server binary (default: ./build/bin/llama-server)
  MAIN_GGUF     path to target model
  DRAFT_GGUF    path to MTP draft head
  CTX           context size (default 16384)
  NGL           gpu layers (default 99)
  HOST/PORT     bind address (default 127.0.0.1:8080)
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

MODE="$1"
shift

ROOT="$(cd "$(dirname "$0")" && pwd)"

SERVER="${LLAMA_SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-${ROOT}/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf}"
DRAFT="${DRAFT_GGUF:-${ROOT}/gemma-assistant-mtp-q4.gguf}"

CTX="${CTX:-16384}"
NGL="${NGL:-99}"
NGL_DRAFT="${NGL_DRAFT:-99}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FA="${FA:-on}"

case "$MODE" in
  f16-mtp)     CTK=f16;     CTV=f16;     CTKD=f16;     CTVD=f16;     SPEC=mtp ;;
  turbo3-mtp)  CTK=turbo3;  CTV=turbo3;  CTKD=turbo3;  CTVD=turbo3;  SPEC=mtp ;;
  f16-base)    CTK=f16;     CTV=f16;     SPEC=off ;;
  turbo3-base) CTK=turbo3;  CTV=turbo3;  SPEC=off ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "error: unknown mode '$MODE'" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ ! -f "$SERVER" ]]; then
  echo "error: llama-server not found: $SERVER" >&2
  exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: target model not found: $MAIN" >&2
  exit 1
fi
if [[ "$SPEC" == "mtp" && ! -f "$DRAFT" ]]; then
  echo "error: MTP draft model not found: $DRAFT" >&2
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
  --parallel 1
  -np 1
  --cont-batching
  --metrics
  --slots
)

if [[ "$SPEC" == "mtp" ]]; then
  ARGS+=(
    -ngld "$NGL_DRAFT"
    -ctkd "$CTKD"
    -ctvd "$CTVD"
    --mtp-head "$DRAFT"
    --spec-type mtp
    --draft-block-size "${DRAFT_BLOCK_SIZE:-4}"
    --draft-max "${DRAFT_MAX:-16}"
    --draft-min "${DRAFT_MIN:-0}"
  )
fi

echo "info: mode=$MODE  ctk/ctv=$CTK/$CTV  spec=$SPEC  host=$HOST:$PORT" >&2
exec "$SERVER" "${ARGS[@]}" "$@"
