#!/bin/bash
# NIAH follow-up on Qwen3.5-27B to test whether boundary_skip fixes the
# middle/end position retrieval failures observed under V3 skip=0.
#
# Baseline context (from earlier runs):
#   27B V3 skip=0: start PASS, middle FAIL, end FAIL
#
# This script tests skip=4 and skip=8 at start/middle/end to see if either
# recovers the failing positions without breaking the passing one.
#
# Uses -n 1024 because Qwen3.5-27B is a reasoning model: it emits a <think>
# block before the final answer. At -n 32 the reasoning trace never finishes.
set +e

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
COMP="$LLAMA_DIR/build-test/bin/llama-completion"
M_27B="/Users/tom/local_llms/models/Qwen3.5-27B-Q8_0.gguf"
RESULTS="$LLAMA_DIR/boundary_27b_niah_results.txt"
TMP_PROMPT="/tmp/boundary_27b_niah_prompt.txt"

> "$RESULTS"

log() {
  echo "$(date '+%H:%M:%S') $*" | tee -a "$RESULTS"
}

build_prompt() {
  local pos_chars=$1
  python3 -c "
pos = $pos_chars
needle = 'The secret code word is PURPLE ELEPHANT 7742.'
question = ' What is the secret code word mentioned earlier? Answer with just the code word and number, nothing else:'
total = 32768 * 4
with open('$WIKI') as f:
    w = f.read()
before = w[:pos]
after_len = total - pos - len(needle) - len(question) - 50
if after_len < 100:
    after_len = 100
after = w[pos:pos+after_len]
with open('$TMP_PROMPT', 'w') as f:
    f.write(f'{before} {needle} {after} {question}')
"
}

run_niah() {
  local boundary_skip=$1 pos_chars=$2
  build_prompt "$pos_chars"

  log ">>> 27B V3 prefix=128 boundary_skip=$boundary_skip @$pos_chars"
  local start=$(date +%s)
  local out
  out=$("$COMP" -m "$M_27B" -f "$TMP_PROMPT" -n 1024 -c 32768 --temp 0 -no-cnv --no-display-prompt \
        --triatt-budget 29491 --triatt-hybrid 2 --triatt-prefix 128 --triatt-boundary-skip "$boundary_skip" \
        2>/tmp/boundary_27b_niah_stderr.txt)
  local rt=$(($(date +%s) - start))

  local gen
  gen=$(echo "$out" | grep -v "^common_perf_print\|^llama_memory_breakdown\|^ggml_metal\|^llama_perf")

  local result="FAIL"
  if echo "$gen" | grep -q "PURPLE ELEPHANT 7742"; then
    result="PASS"
  elif echo "$gen" | grep -qi "PURPLE ELEPHANT"; then
    result="PARTIAL_WORD"
  elif echo "$gen" | grep -qi "7742"; then
    result="PARTIAL_NUMBER"
  fi

  local evs
  evs=$(grep -c "evict: evicted" /tmp/boundary_27b_niah_stderr.txt 2>/dev/null | head -1)
  log "  $result (runtime=${rt}s, evictions=$evs)"
  local tail_gen
  tail_gen=$(echo "$gen" | grep -v "^$" | tail -5 | tr '\n' ' ' | cut -c1-300)
  log "  tail: $tail_gen"
  echo "" >> "$RESULTS"
}

# Poll for the PPL job to finish before starting NIAH, so we don't fight
# the Metal backend.
echo "Waiting for 27B boundary skip PPL job to finish..."
while pgrep -f "llama-perplexity.*Qwen3.5-27B.*triatt-boundary-skip" >/dev/null 2>&1; do
  sleep 10
done
echo "PPL job done, starting NIAH runs."

echo "============================================" | tee -a "$RESULTS"
echo "Qwen3.5-27B @ 32K NIAH: boundary_skip ablation"
echo "Started: $(date)"
echo "============================================" | tee -a "$RESULTS"

log ""
log "========== boundary_skip=4 (skips first attention layer il=3) =========="
run_niah 4 400      # start (known pass on skip=0, control)
run_niah 4 65000    # middle (known FAIL on skip=0)
run_niah 4 120000   # end (known FAIL on skip=0)

log ""
log "========== boundary_skip=8 (skips first two attention layers il=3,7) =========="
run_niah 8 400
run_niah 8 65000
run_niah 8 120000

echo "============================================" | tee -a "$RESULTS"
echo "Done: $(date)" | tee -a "$RESULTS"
echo "============================================" | tee -a "$RESULTS"
