#!/usr/bin/env bash
# Gemma 4 target + gemma4_assistant MTP draft with TurboQuant KV (turbo3 by default).
# Override paths via MAIN_GGUF, DRAFT_GGUF, or LLAMA_SERVER.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${LLAMA_SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-${ROOT}/.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf}"
# Default assistant GGUF name after fixed conversion (embedding dim 1024, not backbone 2816).
DRAFT="${DRAFT_GGUF:-${ROOT}/.scratch/gemma-assistant-mtp.gguf}"

# Skip with VERIFY_ASSISTANT_GGUF=0
VERIFY_ASSISTANT_GGUF="${VERIFY_ASSISTANT_GGUF:-1}"

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

# Toggle speculative decoding for A/B perf comparison.
#   SPEC=mtp (default) - enable Gemma4 MTP assistant draft
#   SPEC=off           - disable speculative decoding (baseline run)
SPEC="${SPEC:-mtp}"

# Sampler default override. TEMP=0 forces greedy sampling on the target so
# MTP draft acceptance can be compared against an upper bound (no sampler
# divergence between target and MTP head).
TEMP="${TEMP:-}"

# Measurement flags (always on by default; set to 0 to drop)
ENABLE_METRICS="${ENABLE_METRICS:-1}"
ENABLE_SLOTS="${ENABLE_SLOTS:-1}"
LOG_TIMESTAMPS="${LOG_TIMESTAMPS:-1}"
LOG_PREFIX="${LOG_PREFIX:-1}"
NO_WARMUP="${NO_WARMUP:-0}"

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
    echo "hint: convert assistant with repo convert_hf_to_gguf.py -> e.g. .scratch/gemma-assistant-mtp.gguf" >&2
    exit 1
  fi

  if [[ "$VERIFY_ASSISTANT_GGUF" != "0" ]]; then
    if ! python3 "${ROOT}/scripts/verify-gemma4-assistant-gguf.py" "$DRAFT"; then
      echo "hint: PYTHONPATH=${ROOT}/gguf-py python3 ${ROOT}/convert_hf_to_gguf.py ${ROOT}/.scratch/gemma-4-26B-A4B-it-assistant --outfile ${ROOT}/.scratch/gemma-assistant-mtp.gguf --outtype f16" >&2
      exit 1
    fi
  fi
fi

PARALLEL="${PARALLEL:-1}"

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
    --draft-block-size "${DRAFT_BLOCK_SIZE:-4}"
    --draft-max "${DRAFT_MAX:-16}"
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

echo "info: SPEC=${SPEC} CTX=${CTX} NGL=${NGL} FA=${FA}" >&2
exec "$SERVER" "${ARGS[@]}" "$@"
