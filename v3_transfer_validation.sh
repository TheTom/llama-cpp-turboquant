#!/bin/bash
# V3 transfer validation at current HEAD (b9b9f56).
#
# Three targets:
#   1. Qwen3.5-27B @ 32K: baseline + V3 90% prefix=128, PPL 3-chunk, NIAH 3 positions
#   2. Qwen3.5-35B-A3B @ 32K: same
#   3. Qwen2.5-7B @ 64K: same
#
# Goal: prove V3 is a real policy, not a 7B/32K coincidence.
set +e

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
MODELS_DIR="/Users/tom/local_llms/models"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
BENCH="$LLAMA_DIR/build-test/bin/llama-perplexity"
COMP="$LLAMA_DIR/build-test/bin/llama-completion"
M_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q8_0.gguf"
M_27B="$MODELS_DIR/Qwen3.5-27B-Q8_0.gguf"
M_35B="$MODELS_DIR/Qwen3.5-35B-A3B-Q8_0.gguf"
RESULTS="$LLAMA_DIR/v3_transfer_validation_results.txt"
TMP_PROMPT="/tmp/v3_transfer_prompt.txt"

> "$RESULTS"

log() {
  echo "$(date '+%H:%M:%S') $*" | tee -a "$RESULTS"
}

# PPL runner
run_ppl() {
  local name=$1 model=$2 ctx=$3 budget=$4 hybrid=$5
  local triatt=""
  [ "$budget" -gt 0 ] && triatt="--triatt-budget $budget"

  log ">>> PPL $name (ctx=$ctx, budget=$budget, V3=$hybrid)"
  local start=$(date +%s)
  local out
  out=$(TRIATT_HYBRID=$hybrid TRIATT_PREFIX=128 "$BENCH" -m "$model" -f "$WIKI" -c "$ctx" -b 512 --chunks 3 $triatt 2>&1)
  local rt=$(($(date +%s) - start))

  local ppl=$(echo "$out" | grep "Final estimate" | tail -1)
  local evs=$(echo "$out" | grep -c "evict: evicted")
  local cal=$(echo "$out" | grep -c "initial calibration from")
  log "  $ppl"
  log "  evict_rounds=$evs calibrations=$cal runtime=${rt}s"
  echo "" >> "$RESULTS"
}

# NIAH prompt builder — needle position given in CHARACTERS from start
build_prompt() {
  local ctx=$1 pos_chars=$2
  python3 -c "
pos = $pos_chars
ctx = $ctx
needle = 'The secret code word is PURPLE ELEPHANT 7742.'
question = ' What is the secret code word mentioned earlier? Answer with just the code word and number, nothing else:'
total = ctx * 4
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
  local name=$1 model=$2 ctx=$3 budget=$4 hybrid=$5 pos_chars=$6
  local triatt=""
  [ "$budget" -gt 0 ] && triatt="--triatt-budget $budget"

  build_prompt "$ctx" "$pos_chars"

  log ">>> NIAH $name @$pos_chars (ctx=$ctx, budget=$budget, V3=$hybrid)"
  local start=$(date +%s)
  local out
  out=$(TRIATT_HYBRID=$hybrid TRIATT_PREFIX=128 "$COMP" -m "$model" -f "$TMP_PROMPT" -n 32 -c "$ctx" --temp 0 -no-cnv --no-display-prompt $triatt 2>/tmp/v3_transfer_stderr.txt)
  local rt=$(($(date +%s) - start))

  local gen
  gen=$(echo "$out" | grep -v "^common_perf_print\|^llama_memory_breakdown\|^ggml_metal\|^$" | head -10)

  local result="FAIL"
  if echo "$gen" | grep -q "PURPLE ELEPHANT 7742"; then
    result="PASS"
  elif echo "$gen" | grep -qi "PURPLE ELEPHANT"; then
    result="PARTIAL_WORD"
  elif echo "$gen" | grep -qi "7742"; then
    result="PARTIAL_NUMBER"
  fi

  local evs
  evs=$(grep -c "evict: evicted" /tmp/v3_transfer_stderr.txt 2>/dev/null | head -1)
  log "  $result (runtime=${rt}s, evictions=$evs)"
  log "  generated: $(echo "$gen" | tr '\n' ' ' | cut -c1-200)"
  echo "" >> "$RESULTS"
}

echo "============================================" | tee -a "$RESULTS"
echo "V3 transfer validation at HEAD" | tee -a "$RESULTS"
echo "Started: $(date)" | tee -a "$RESULTS"
echo "============================================" | tee -a "$RESULTS"

# --------------------------------------------------
# TARGET 1: Qwen3.5-27B @ 32K
# --------------------------------------------------
log ""
log "========== 1) Qwen3.5-27B @ 32K =========="
run_ppl "27B baseline" "$M_27B" 32768 0     0
run_ppl "27B V3 90%"   "$M_27B" 32768 29491 2

run_niah "27B baseline" "$M_27B" 32768 0     0 400
run_niah "27B baseline" "$M_27B" 32768 0     0 65000
run_niah "27B baseline" "$M_27B" 32768 0     0 120000
run_niah "27B V3 90%"   "$M_27B" 32768 29491 2 400
run_niah "27B V3 90%"   "$M_27B" 32768 29491 2 65000
run_niah "27B V3 90%"   "$M_27B" 32768 29491 2 120000

# --------------------------------------------------
# TARGET 2: Qwen3.5-35B-A3B @ 32K
# --------------------------------------------------
log ""
log "========== 2) Qwen3.5-35B-A3B @ 32K =========="
run_ppl "35B baseline" "$M_35B" 32768 0     0
run_ppl "35B V3 90%"   "$M_35B" 32768 29491 2

run_niah "35B baseline" "$M_35B" 32768 0     0 400
run_niah "35B baseline" "$M_35B" 32768 0     0 65000
run_niah "35B baseline" "$M_35B" 32768 0     0 120000
run_niah "35B V3 90%"   "$M_35B" 32768 29491 2 400
run_niah "35B V3 90%"   "$M_35B" 32768 29491 2 65000
run_niah "35B V3 90%"   "$M_35B" 32768 29491 2 120000

# --------------------------------------------------
# TARGET 3: Qwen2.5-7B @ 64K
# --------------------------------------------------
log ""
log "========== 3) Qwen2.5-7B @ 64K =========="
run_ppl "7B 64K baseline" "$M_7B" 65536 0     0
run_ppl "7B 64K V3 90%"   "$M_7B" 65536 58982 2

# At 64K, needle positions are 2x the 32K ones (scaled by context size)
run_niah "7B 64K baseline" "$M_7B" 65536 0     0 800
run_niah "7B 64K baseline" "$M_7B" 65536 0     0 130000
run_niah "7B 64K baseline" "$M_7B" 65536 0     0 240000
run_niah "7B 64K V3 90%"   "$M_7B" 65536 58982 2 800
run_niah "7B 64K V3 90%"   "$M_7B" 65536 58982 2 130000
run_niah "7B 64K V3 90%"   "$M_7B" 65536 58982 2 240000

echo "============================================" | tee -a "$RESULTS"
echo "Done: $(date)" | tee -a "$RESULTS"
echo "============================================" | tee -a "$RESULTS"
