#!/usr/bin/env bash
# scripts/oscar-bias-diag.sh
# Prompt-dependent token bias diagnostic for OSCAR2 vs f16/q2_0 KV cache.
#
# Runs a deterministic matrix:
#   prompts x cache_types -> llama-cli invocations
# Captures first/first-10 generated tokens, throughput, and a quality label.
# Per-run outputs go to logs/oscar-bias/diag-<timestamp>/<run_id>/.
# Master TSV report: logs/oscar-bias/diag-<timestamp>/report.tsv
#
# Usage:
#   ./scripts/oscar-bias-diag.sh --model /path/to/model-rot.gguf
#   ./scripts/oscar-bias-diag.sh --model <path> --prompts a --skip-cache f16
#   ./scripts/oscar-bias-diag.sh --binary ./build/bin/llama-cli --ctx 262144
#
# Exit is 0 even if individual runs fail; check TSV `quality` column.

set -u
set -o pipefail

MODEL=""
BINARY="./build/bin/llama-cli"
NGL=99
CTX=8192
N_GEN=64
SEED=42
PROMPTS_FILTER="ab"
SKIP_F16=0
SKIP_Q2=0
SKIP_OSCAR=0
LOG_ROOT="logs/oscar-bias"

# Prompt class A: factual / low branching
declare -A PROMPT_CLASS=(
  [A1]="A" [A2]="A"
  [B1]="B" [B2]="B" [B3]="B"
)

declare -A PROMPT_TEXT=(
  [A1]="The capital of France is"
  [A2]="Water boils at exactly"
  [B1]="Write a Python function that returns Fibonacci numbers up to n."
  [B2]="Translate the following to Spanish: Hello world"
  [B3]="Count from 1 to 5"
)

CACHE_TYPES=(f16 q2_0 oscar2)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)           MODEL="$2"; shift 2 ;;
    --binary)          BINARY="$2"; shift 2 ;;
    --ngl)             NGL="$2"; shift 2 ;;
    --ctx)             CTX="$2"; shift 2 ;;
    --n-gen)           N_GEN="$2"; shift 2 ;;
    --seed)            SEED="$2"; shift 2 ;;
    --prompts)         PROMPTS_FILTER="$2"; shift 2 ;;
    --skip-cache)      case "$2" in f16) SKIP_F16=1 ;; q2_0) SKIP_Q2=1 ;; oscar2) SKIP_OSCAR=1 ;; *) echo "bad --skip-cache: $2"; exit 2 ;; esac; shift 2 ;;
    --log-root)        LOG_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ERROR: --model is required" >&2
  exit 2
fi
if [[ ! -x "$BINARY" && ! -f "$BINARY" ]]; then
  echo "ERROR: binary not found: $BINARY" >&2
  exit 2
fi
case "$PROMPTS_FILTER" in
  a|A) KEYS=(A1 A2) ;;
  b|B) KEYS=(B1 B2 B3) ;;
  ab|AB|aB|Ab|a|b) KEYS=(A1 A2 B1 B2 B3) ;;
  *) echo "bad --prompts: $PROMPTS_FILTER"; exit 2 ;;
esac

MODEL_SHORT=$(basename "$MODEL")
TS=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$LOG_ROOT/diag-$TS"
mkdir -p "$RUN_DIR"
TSV="$RUN_DIR/report.tsv"

printf 'run_id\tmodel_short\tprompt_class\tprompt_id\tprompt_text\tctx\tn_gen\tcache_k\tcache_v\tseed\tfirst_token\tfirst_10_tokens\tquality\ttps\tnotes\n' > "$TSV"

# Classify generated text quality: blank | slashes | ok | garbled.
# Best-effort heuristic; column is informational, not a verdict.
classify_quality() {
  local out="$1"
  local alnum slash total
  alnum=$(printf '%s' "$out" | tr -cd '[:alnum:]' | wc -c)
  slash=$(printf '%s' "$out" | tr -cd '/' | wc -c)
  total=$(printf '%s' "$out" | tr -cd '[:print:]' | wc -c)
  if [[ $total -lt 2 ]]; then
    echo "blank"; return
  fi
  if [[ $total -ge 8 && $slash -gt $((total / 2)) ]]; then
    echo "slashes"; return
  fi
  # Require real-letter density >= 50% and at least one space-separated word.
  # Note: multi-word output needed to qualify as 'ok'; single-word replies
  # (e.g. 'Hello!') intentionally fall through to 'garbled' for this diagnostic.
  if [[ $total -ge 12 && $alnum -gt $((total / 2)) ]] \
     && printf '%s' "$out" | grep -qE '[A-Za-z][A-Za-z][[:space:]][A-Za-z]'; then
    echo "ok"; return
  fi
  echo "garbled"
}

# Strip characters that would break tab-separated TSV rows.
tsv_safe() {
  printf '%s' "$1" | tr '\t\n\r' '   '
}

run_one() {
  local p_id="$1"; local cache="$2"
  local p_class="${PROMPT_CLASS[$p_id]}"
  local p_text="${PROMPT_TEXT[$p_id]}"
  local run_id="run-${p_id}-${cache}"
  local out_dir="$RUN_DIR/$run_id"
  mkdir -p "$out_dir"
  local stdout_log="$out_dir/stdout.log"
  local stderr_log="$out_dir/stderr.log"
  local meta_log="$out_dir/meta.txt"

  printf 'model=%s\nprompt_class=%s\nprompt_id=%s\ncache=%s\nctx=%s\nn_gen=%s\nseed=%s\n' \
    "$MODEL" "$p_class" "$p_id" "$cache" "$CTX" "$N_GEN" "$SEED" > "$meta_log"

  local exit_code
  "$BINARY" \
    -m "$MODEL" \
    -ngl "$NGL" \
    -c "$CTX" \
    -fa on \
    --cache-type-k "$cache" \
    --cache-type-v "$cache" \
    --temp 0 \
    --seed "$SEED" \
    --no-jinja \
    --no-conversation \
    -n "$N_GEN" \
    -p "$p_text" \
    > "$stdout_log" 2> "$stderr_log"
  exit_code=$?

  # First token: take first whitespace-delimited word of generated text.
  # llama-cli usually appends generated text after a "..." or newline.
  local generated sanitized
  generated=$(awk '/^|\s*$|^prompt eval|^sampling|^system_prompt/{next} {gsub(/\r/, ""); print}' "$stdout_log" \
    | sed -e 's/^[[:space:]]*//' -e '/^$/d' \
    | tail -n +1)
  # Sanitize BEFORE deriving first_token so stray tabs/CRs from llama-cli
  # don't fragment the first whitespace-delimited word.
  sanitized=$(printf '%s' "$generated" | tr '\t\n\r' '   ')
  local first_token first10 quality tps notes
  first10=$(printf '%s' "$sanitized" | head -c 80)
  first_token=$(printf '%s' "$first10" | awk '{print $1}')
  quality=$(classify_quality "$generated")
  # Throughput: parse from stderr typical "total time = Xs, Y tokens, Z tps".
  tps=$(grep -oE '[0-9]+\.[0-9]+ tps' "$stderr_log" 2>/dev/null \
    | head -n1 | awk '{print $1}')
  [[ -z "$tps" ]] && tps="0.00"
  notes=""
  if [[ $exit_code -ne 0 ]]; then
    notes="exit=${exit_code}"
  fi
  if [[ "$cache" == "q2_0" ]]; then
    if grep -q 'LLAMA_KV_NO_HADAMARD' "$stderr_log" 2>/dev/null; then
      notes="$notes NO_HADAMARD_ABSENT"
    else
      notes="$notes NO_HADAMARD_PROBED"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_id" "$MODEL_SHORT" "$p_class" "$p_id" \
    "$(tsv_safe "${p_text:0:60}")" "$CTX" "$N_GEN" "$cache" "$cache" "$SEED" \
    "$(tsv_safe "${first_token:-NONE}")" "$(tsv_safe "$first10")" \
    "$(tsv_safe "$quality")" "$(tsv_safe "$tps")" "$(tsv_safe "$notes")" \
    >> "$TSV"

  printf '  [%s/%s] quality=%s tps=%s exit=%d\n' "$p_id" "$cache" "$quality" "$tps" "$exit_code"
}

echo "OSCAR bias diagnostic"
echo "  model: $MODEL"
echo "  binary: $BINARY"
echo "  ctx: $CTX  n_gen: $N_GEN  seed: $SEED"
echo "  prompts: $PROMPTS_FILTER (${KEYS[*]})"
echo "  output: $TSV"
echo "---"

cache_enable() {
  case "$1" in
    f16)    [[ $SKIP_F16 -eq 0 ]] ;;
    q2_0)   [[ $SKIP_Q2 -eq 0 ]] ;;
    oscar2) [[ $SKIP_OSCAR -eq 0 ]] ;;
  esac
}

n_runs=0
for p_id in "${KEYS[@]}"; do
  for cache in "${CACHE_TYPES[@]}"; do
    if cache_enable "$cache"; then
      run_one "$p_id" "$cache"
      n_runs=$((n_runs + 1))
    fi
  done
done

echo "---"
echo "completed $n_runs runs"
echo "report: $TSV"
echo ""
echo "Next steps:"
echo "  column -t -s \$'\\t' < $TSV | less -S"
echo "  awk -F'\\t' 'NR>1 && \$13 != \"ok\"' $TSV"
