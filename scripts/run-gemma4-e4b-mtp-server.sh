#!/usr/bin/env bash
# Gemma 4 E4B Edge target + gemma4_assistant MTP draft (centroid LM head when use_ordered_embeddings=true).
# Override paths via MAIN_GGUF, DRAFT_GGUF, or LLAMA_SERVER.
#
# MTP_PRESET (ordered-embeddings MTP is heavy on Edge):
#   throughput — block 2, max 6
#   lift       — block 3, max 8
#   balanced   — block 3, max 8
#   quality    — block 4, max 16 (max depth; best when acceptance is high)
# Override with DRAFT_BLOCK_SIZE, DRAFT_MAX. Optional: export LLAMA_MTP_SKIP_STREAK_THRESHOLD=1–32
# to skip drafting after consecutive zero-accept MTP batches (off by default in the binary).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${LLAMA_SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-${ROOT}/.scratch/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf}"
# Prefer quantized MTP head when present (see scripts/quantize-gemma4-edge-assistant-mtp.sh).
_DRAFT_Q4="${ROOT}/.scratch/gemma-e4b-assistant-mtp-Q4_K_M.gguf"
_DRAFT_F16="${ROOT}/.scratch/gemma-e4b-assistant-mtp.gguf"
if [[ -n "${DRAFT_GGUF:-}" ]]; then
  DRAFT="$DRAFT_GGUF"
elif [[ -f "$_DRAFT_Q4" ]]; then
  DRAFT="$_DRAFT_Q4"
else
  DRAFT="$_DRAFT_F16"
fi

VERIFY_ASSISTANT_GGUF="${VERIFY_ASSISTANT_GGUF:-1}"

MTP_PRESET="${MTP_PRESET:-throughput}"
case "$MTP_PRESET" in
  throughput)
    : "${DRAFT_BLOCK_SIZE:=2}"
    : "${DRAFT_MAX:=6}"
    ;;
  lift)
    : "${DRAFT_BLOCK_SIZE:=3}"
    : "${DRAFT_MAX:=8}"
    ;;
  balanced)
    : "${DRAFT_BLOCK_SIZE:=3}"
    : "${DRAFT_MAX:=8}"
    ;;
  quality)
    : "${DRAFT_BLOCK_SIZE:=4}"
    : "${DRAFT_MAX:=16}"
    ;;
  *)
    echo "warn: unknown MTP_PRESET=${MTP_PRESET}; using throughput defaults" >&2
    : "${DRAFT_BLOCK_SIZE:=2}"
    : "${DRAFT_MAX:=6}"
    ;;
esac

CTX="${CTX:-16384}"
NGL="${NGL:-99}"
NGL_DRAFT="${NGL_DRAFT:-99}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-turbo3}"
CTKD="${CTKD:-turbo3}"
CTVD="${CTVD:-turbo3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FA="${FA:-on}"
SPEC="${SPEC:-mtp}"
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
if [[ "$SPEC" == "mtp" ]]; then
  if [[ ! -f "$DRAFT" ]]; then
    echo "error: draft (assistant) GGUF not found: ${DRAFT}" >&2
    echo "hint: hf download google/gemma-4-E4B-it-assistant --local-dir ${ROOT}/.scratch/gemma-4-E4B-it-assistant" >&2
    echo "hint: PYTHONPATH=${ROOT}/gguf-py python3 ${ROOT}/convert_hf_to_gguf.py ${ROOT}/.scratch/gemma-4-E4B-it-assistant --outfile ${_DRAFT_F16} --outtype f16" >&2
    echo "hint: ${ROOT}/scripts/quantize-gemma4-edge-assistant-mtp.sh e4b   # then re-run (prefers Q4_K_M draft automatically)" >&2
    exit 1
  fi
  if [[ "$VERIFY_ASSISTANT_GGUF" != "0" ]]; then
    if ! python3 "${ROOT}/scripts/verify-gemma4-assistant-gguf.py" "$DRAFT"; then
      echo "error: assistant GGUF verification failed" >&2
      exit 1
    fi
  fi
fi

ARGS=(
  -m "$MAIN"
  -c "$CTX"
  -ngl "$NGL"
  -ngld "$NGL_DRAFT"
  -ctk "$CTK"
  -ctv "$CTV"
  -ctkd "$CTKD"
  -ctvd "$CTVD"
  -fa "$FA"
  --host "$HOST"
  --port "$PORT"
  --parallel "$PARALLEL"
  -np "$PARALLEL"
  --cont-batching
)

if [[ "$SPEC" == "mtp" ]]; then
  ARGS+=(
    --mtp-head "$DRAFT"
    --spec-type mtp
    --draft-block-size "${DRAFT_BLOCK_SIZE}"
    --draft-max "${DRAFT_MAX}"
    --draft-min "${DRAFT_MIN:-0}"
  )
else
  echo "info: speculative decoding disabled (SPEC=${SPEC}); running baseline" >&2
fi

if [[ -n "$TEMP" ]]; then
  ARGS+=(--temp "$TEMP")
fi

[[ "$ENABLE_METRICS"  != "0" ]] && ARGS+=(--metrics)
[[ "$ENABLE_SLOTS"    != "0" ]] && ARGS+=(--slots)
[[ "$LOG_TIMESTAMPS"  != "0" ]] && ARGS+=(--log-timestamps)
[[ "$LOG_PREFIX"      != "0" ]] && ARGS+=(--log-prefix)
[[ "$NO_WARMUP"       != "0" ]] && ARGS+=(--no-warmup)

echo "info: SPEC=${SPEC} MTP_PRESET=${MTP_PRESET} DRAFT_BLOCK_SIZE=${DRAFT_BLOCK_SIZE} DRAFT_MAX=${DRAFT_MAX} LLAMA_MTP_SKIP_STREAK_THRESHOLD=${LLAMA_MTP_SKIP_STREAK_THRESHOLD:-}" >&2
echo "info: CTX=${CTX} NGL=${NGL} FA=${FA} CTK=${CTK} CTKD=${CTKD}" >&2
echo "info: MAIN=${MAIN}" >&2
echo "info: DRAFT=${DRAFT}" >&2
exec "$SERVER" "${ARGS[@]}" "$@"
