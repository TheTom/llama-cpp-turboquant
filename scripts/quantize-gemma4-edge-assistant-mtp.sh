#!/usr/bin/env bash
# Quantize Gemma 4 Edge (E2B/E4B) gemma4_assistant MTP GGUF to Q4 with llama-quantize.
#
# Usage:
#   ./scripts/quantize-gemma4-edge-assistant-mtp.sh e4b [Q4_K_M]
#   ./scripts/quantize-gemma4-edge-assistant-mtp.sh e2b [Q4_K_M]
#   ./scripts/quantize-gemma4-edge-assistant-mtp.sh <in.gguf> <out.gguf> [type] [threads]
#
# Defaults: type Q4_K_M, threads = nproc. Input must be F16 (or other) GGUF from convert_hf_to_gguf.py.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUANT="${LLAMA_QUANTIZE:-${ROOT}/build/bin/llama-quantize}"
TYPE="${ASSISTANT_QUANT_TYPE:-Q4_K_M}"
THREADS="${QUANT_THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)}"

if [[ ! -x "$QUANT" && ! -f "$QUANT" ]]; then
  echo "error: llama-quantize not found: ${QUANT} (cmake --build build --target llama-quantize)" >&2
  exit 1
fi

usage() {
  echo "usage: $0 e4b|e2b [${TYPE}]" >&2
  echo "       $0 <input.gguf> <output.gguf> [type] [threads]" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

if [[ "$1" == "e4b" ]]; then
  IN="${ROOT}/.scratch/gemma-e4b-assistant-mtp.gguf"
  [[ $# -ge 2 ]] && TYPE="$2"
  OUT="${ROOT}/.scratch/gemma-e4b-assistant-mtp-${TYPE}.gguf"
elif [[ "$1" == "e2b" ]]; then
  IN="${ROOT}/.scratch/gemma-e2b-assistant-mtp.gguf"
  [[ $# -ge 2 ]] && TYPE="$2"
  OUT="${ROOT}/.scratch/gemma-e2b-assistant-mtp-${TYPE}.gguf"
elif [[ $# -ge 2 ]]; then
  IN="$1"
  OUT="$2"
  TYPE="${3:-$TYPE}"
  THREADS="${4:-$THREADS}"
else
  usage
fi

if [[ ! -f "$IN" ]]; then
  echo "error: input GGUF not found: ${IN}" >&2
  exit 1
fi

echo "info: quantize ${IN} -> ${OUT} type=${TYPE} threads=${THREADS}" >&2
exec "$QUANT" "$IN" "$OUT" "$TYPE" "$THREADS"
