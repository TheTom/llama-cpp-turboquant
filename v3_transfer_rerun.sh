#!/bin/bash
# V3 transfer validation RERUN at current HEAD (b9b9f56).
#
# Skips 27B PPL (already verified earlier at 7.4640 baseline, 7.4853 V3).
# Runs:
#   - 35B-A3B @ 32K PPL (baseline + V3)
#   - 7B @ 64K PPL (baseline + V3)
#   - NIAH for 27B, 35B, 7B 64K — with -n 1024 so reasoning models have
#     room to finish their <think> block AND produce the final answer.
#     Grep checker over generated output (--no-display-prompt) is safe
#     because the needle is a random string ("PURPLE ELEPHANT 7742") that
#     the model effectively cannot hallucinate.
set +e

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
MODELS_DIR="/Users/tom/local_llms/models"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
BENCH="$LLAMA_DIR/build-test/bin/llama-perplexity"
COMP="$LLAMA_DIR/build-test/bin/llama-completion"
M_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q8_0.gguf"
M_27B="$MODELS_DIR/Qwen3.5-27B-Q8_0.gguf"
M_35B="$MODELS_DIR/Qwen3.5-35B-A3B-Q8_0.gguf"
RESULTS="$LLAMA_DIR/v3_transfer_rerun_results.txt"
TMP_PROMPT="/tmp/v3_rerun_prompt.txt"

> "$RESULTS"

log() {
  echo "$(date '+%H:%M:%S') $*" | tee -a "$RESULTS"
}

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

# NIAH runner — bumped -n to 1024 so reasoning models can finish their
# <think> block AND produce the final answer. For qwen2.5 (non-reasoning)
# it's harmless, just slightly slower.
run_niah() {
  local name=$1 model=$2 ctx=$3 budget=$4 hybrid=$5 pos_chars=$6 n_predict=$7
  local triatt=""
  [ "$budget" -gt 0 ] && triatt="--triatt-budget $budget"

  build_prompt "$ctx" "$pos_chars"

  log ">>> NIAH $name @$pos_chars (ctx=$ctx, budget=$budget, V3=$hybrid, n=$n_predict)"
  local start=$(date +%s)
  local out
  out=$(TRIATT_HYBRID=$hybrid TRIATT_PREFIX=128 "$COMP" -m "$model" -f "$TMP_PROMPT" -n "$n_predict" -c "$ctx" --temp 0 -no-cnv --no-display-prompt $triatt 2>/tmp/v3_rerun_stderr.txt)
  local rt=$(($(date +%s) - start))

  # Strip ggml/perf lines; keep all generated content (incl. any <think> blocks)
  local gen
  gen=$(echo "$out" | grep -v "^common_perf_print\|^llama_memory_breakdown\|^ggml_metal\|^llama_perf")

  local result="FAIL"
  # The needle is a random string that cannot be hallucinated.
  # Anywhere in the generated output = real retrieval from KV cache.
  if echo "$gen" | grep -q "PURPLE ELEPHANT 7742"; then
    result="PASS"
  elif echo "$gen" | grep -qi "PURPLE ELEPHANT"; then
    result="PARTIAL_WORD"
  elif echo "$gen" | grep -qi "7742"; then
    result="PARTIAL_NUMBER"
  fi

  local evs
  evs=$(grep -c "evict: evicted" /tmp/v3_rerun_stderr.txt 2>/dev/null | head -1)
  log "  $result (runtime=${rt}s, evictions=$evs)"
  # Show last 5 non-empty lines of generation — for reasoning models this
  # will typically include the final answer after </think>
  local tail_gen
  tail_gen=$(echo "$gen" | grep -v "^$" | tail -5 | tr '\n' ' ' | cut -c1-300)
  log "  tail: $tail_gen"
  echo "" >> "$RESULTS"
}

echo "============================================" | tee -a "$RESULTS"
echo "V3 transfer rerun (fixed NIAH for reasoning models)" | tee -a "$RESULTS"
echo "Started: $(date)" | tee -a "$RESULTS"
echo "Note: 27B PPL already verified (baseline 7.4640 -> V3 7.4853 +0.29%)" | tee -a "$RESULTS"
echo "============================================" | tee -a "$RESULTS"

# --------------------------------------------------
# 35B-A3B @ 32K PPL
# --------------------------------------------------
log ""
log "========== 35B-A3B @ 32K PPL =========="
run_ppl "35B baseline" "$M_35B" 32768 0     0
run_ppl "35B V3 90%"   "$M_35B" 32768 29491 2

# --------------------------------------------------
# 7B @ 64K PPL
# --------------------------------------------------
log ""
log "========== 7B @ 64K PPL =========="
run_ppl "7B 64K baseline" "$M_7B" 65536 0     0
run_ppl "7B 64K V3 90%"   "$M_7B" 65536 58982 2

# --------------------------------------------------
# NIAH: 27B @ 32K (reasoning model, -n 1024)
# --------------------------------------------------
log ""
log "========== 27B NIAH @ 32K (n=1024) =========="
run_niah "27B baseline" "$M_27B" 32768 0     0 400    1024
run_niah "27B baseline" "$M_27B" 32768 0     0 65000  1024
run_niah "27B baseline" "$M_27B" 32768 0     0 120000 1024
run_niah "27B V3 90%"   "$M_27B" 32768 29491 2 400    1024
run_niah "27B V3 90%"   "$M_27B" 32768 29491 2 65000  1024
run_niah "27B V3 90%"   "$M_27B" 32768 29491 2 120000 1024

# --------------------------------------------------
# NIAH: 35B-A3B @ 32K (reasoning model, -n 1024)
# --------------------------------------------------
log ""
log "========== 35B-A3B NIAH @ 32K (n=1024) =========="
run_niah "35B baseline" "$M_35B" 32768 0     0 400    1024
run_niah "35B baseline" "$M_35B" 32768 0     0 65000  1024
run_niah "35B baseline" "$M_35B" 32768 0     0 120000 1024
run_niah "35B V3 90%"   "$M_35B" 32768 29491 2 400    1024
run_niah "35B V3 90%"   "$M_35B" 32768 29491 2 65000  1024
run_niah "35B V3 90%"   "$M_35B" 32768 29491 2 120000 1024

# --------------------------------------------------
# NIAH: 7B @ 64K (not a reasoning model, but keep -n 1024 for consistency)
# --------------------------------------------------
log ""
log "========== 7B NIAH @ 64K (n=64) =========="
run_niah "7B 64K baseline" "$M_7B" 65536 0     0 800    64
run_niah "7B 64K baseline" "$M_7B" 65536 0     0 130000 64
run_niah "7B 64K baseline" "$M_7B" 65536 0     0 240000 64
run_niah "7B 64K V3 90%"   "$M_7B" 65536 58982 2 800    64
run_niah "7B 64K V3 90%"   "$M_7B" 65536 58982 2 130000 64
run_niah "7B 64K V3 90%"   "$M_7B" 65536 58982 2 240000 64

echo "============================================" | tee -a "$RESULTS"
echo "Done: $(date)" | tee -a "$RESULTS"
echo "============================================" | tee -a "$RESULTS"
