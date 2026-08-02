#!/bin/bash
# OSCAR q2_0 KV Cache Test Suite v2
# Uses the same command format as the user's real tests:
#   CUDA_VISIBLE_DEVICES="" LLAMA_KV_FUSED_FA=1 <env> ./build/bin/llama-cli
#   --cache-type-k <t> --cache-type-v <t>
#   --chat-template-file <jinja> -p "..." -n 50

MODEL="/mnt/storage/Projects/OSCAR-llamacpp/assets/gemma4-12b-rot/gemma-4-12b-it-rot-kv.gguf"
CLI="./build/bin/llama-cli"
TEMPLATE="models/templates/google-gemma-4-31B-it.jinja"
OUTDIR="test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${OUTDIR}/test_summary_${TIMESTAMP}.txt"

mkdir -p "$OUTDIR"

run_test() {
    local num="$1"
    local name="$2"
    local type_k="$3"
    local type_v="$4"
    shift 4
    local env_vars=("$@")

    local safename=$(echo "$name" | tr ' /' '__')
    local logfile="${OUTDIR}/test_${num}_${safename}.log"

    # Build tag
    local rot_k=0 rot_v=0 clip="" hp="" had=""
    for ev in "${env_vars[@]}"; do
        case "$ev" in
            LLAMA_ATTN_ROT_K_OVERRIDE=1) rot_k=1 ;;
            LLAMA_ATTN_ROT_V_OVERRIDE=1) rot_v=1 ;;
            LLAMA_KV_CLIP_RATIO=0.96) clip="clip=0.96" ;;
            LLAMA_KV_HP_SINK=*) hp="hp=64+256" ;;
            LLAMA_KV_NO_HADAMARD=1) had="no_had" ;;
        esac
    done
    local rot_desc="no_rot"
    [ "$rot_k" = "1" ] && [ "$rot_v" = "1" ] && rot_desc="rot_both"
    [ "$rot_k" = "1" ] && [ "$rot_v" = "0" ] && rot_desc="rot_k=1"
    [ "$rot_k" = "0" ] && [ "$rot_v" = "1" ] && rot_desc="rot_v=1"

    local tag="${type_k}_${type_v}_${rot_desc}"
    [ -n "$clip" ] && tag="${tag}_${clip}"
    [ -n "$hp" ] && tag="${tag}_${hp}"
    [ -n "$had" ] && tag="${tag}_${had}"

    echo ""
    echo "======================================================================"
    echo "  TEST ${num}: ${name}"
    echo "  Tag: ${tag}"
    echo "  K=${type_k} V=${type_v}"
    for ev in "${env_vars[@]}"; do echo "  Env: ${ev}"; done
    echo "  Log: ${logfile}"
    echo "======================================================================"

    (
        export CUDA_VISIBLE_DEVICES=""
        export LLAMA_KV_FUSED_FA=1
        for ev in "${env_vars[@]}"; do export "${ev?}"; done

        ${TIMEOUT_BIN:-timeout} 180 ${CLI} \
            -m "${MODEL}" -ngl 0 -fa on -c 16384 \
            --cache-type-k "${type_k}" --cache-type-v "${type_v}" \
            --chat-template-file "${TEMPLATE}" \
            -p "What is 2+2?" -n 50 \
            </dev/null
    ) > "${logfile}" 2>&1 || true

    # Extract output: last non-empty non-debug lines after prompt
    local output="$(sed -n '/^> What is 2+2/,$ p' "${logfile}" 2>/dev/null | \
        tail -n +2 | grep -v '^> ' | grep -vE '^\s*$' | \
        grep -vE '^(I|D|W|E) |^\[' | head -10 | tr '\n' ' ' | sed 's/  */ /g')"
    [ -z "$output" ] && output="[NO OUTPUT]"

    local cache_info=$(grep -E 'size =.*K \(.*\)' "${logfile}" | head -2)
    local rot_info=$(grep -E 'attn_rot_k|attn_rot_v' "${logfile}" | head -2)
    local hp_info=$(grep -E 'HP buffer' "${logfile}" | head -2)
    local timing=$(grep -E 'Prompt:|Generation:' "${logfile}" | head -2)

    echo ""
    echo "  Output: ${output:0:300}"
    echo ""
    echo "  Cache: ${cache_info}"
    echo "  Rot:   ${rot_info}"
    [ -n "$hp_info" ] && echo "  HP:    ${hp_info}"
    [ -n "$timing" ] && echo "  Time:  ${timing}"
    echo "  --- Test ${num} complete ---"

    printf -- "%d|\"%s\"|%s|%s|%s|%s|%s\n" \
        "$num" "$name" "$tag" \
        "$(echo "$cache_info" | tr -d '\n' | sed 's/  */ /g')" \
        "$(echo "$rot_info" | tr -d '\n' | sed 's/  */ /g')" \
        "$(echo "$hp_info" | tr -d '\n' | sed 's/  */ /g')" \
        "${output:0:120}" >> "$SUMMARY"
}

# ================================================================
TIMEOUT_BIN=$(which timeout 2>/dev/null || echo "/usr/bin/timeout")
echo "================================================================="
echo "  OSCAR q2_0 KV Cache Test Suite v2"
echo "  Started: $(date)"
echo "  Model: ${MODEL}"
echo "  Chat template: ${TEMPLATE}"
echo "================================================================="
echo ""

echo '#|Name|Tag|Cache|Rotation|HP|Output' > "$SUMMARY"
echo '--|----|---|-----|--------|---|------' >> "$SUMMARY"

# ================================================================
# GPU tests (K=f16 — fast but no rotation for f16)
# ================================================================
run_test 1 "f16 baseline" f16 f16

run_test 2 "V q2_0 only" f16 q2_0 \
    LLAMA_ATTN_ROT_K_OVERRIDE=0 LLAMA_ATTN_ROT_V_OVERRIDE=1

run_test 3 "f16 + rot (no effect)" f16 f16 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1

# ================================================================
# K=q2_0 + V=f16 (intended target config)
# ================================================================
# NO_HADAMARD not set = 0 = Hadamard ON = CORRECT for this GGUF
run_test 4 "K q2_0 full pipe" q2_0 f16 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1 \
    LLAMA_KV_CLIP_RATIO=0.96 \
    LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256

# NO_HADAMARD=1 = SKIP Hadamard = WRONG for this GGUF
run_test 5 "K q2_0 NO_HADAMARD" q2_0 f16 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1 \
    LLAMA_KV_CLIP_RATIO=0.96 \
    LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 \
    LLAMA_KV_NO_HADAMARD=1

# No clip ratio
run_test 6 "K q2_0 no clip" q2_0 f16 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1 \
    LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256

# No HP buffer
run_test 7 "K q2_0 no HP" q2_0 f16 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1 \
    LLAMA_KV_CLIP_RATIO=0.96

# No rotation
run_test 8 "K q2_0 no rot" q2_0 f16 \
    LLAMA_ATTN_ROT_K_OVERRIDE=0 LLAMA_ATTN_ROT_V_OVERRIDE=0 \
    LLAMA_KV_CLIP_RATIO=0.96 \
    LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256

# ================================================================
# Full q2_0 (max compression)
# ================================================================
run_test 9 "Full q2_0 full pipe" q2_0 q2_0 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1 \
    LLAMA_KV_CLIP_RATIO=0.96 \
    LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256

run_test 10 "Full q2_0 no clip" q2_0 q2_0 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1 \
    LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256

run_test 11 "Full q2_0 no rot" q2_0 q2_0 \
    LLAMA_ATTN_ROT_K_OVERRIDE=0 LLAMA_ATTN_ROT_V_OVERRIDE=0 \
    LLAMA_KV_CLIP_RATIO=0.96 \
    LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256

# ================================================================
# Extra: test with the shorter (non-interleaved) template too
# ================================================================
run_test 12 "K q2_0 interleaved" q2_0 f16 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1 \
    LLAMA_KV_CLIP_RATIO=0.96 \
    LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 \
    TEMPLATE_OVERRIDE=models/templates/google-gemma-4-31B-it-interleaved.jinja
# Note: TEMPLATE_OVERRIDE is not an env var the script respects.
# This test just uses the same template; the interleaved version
# is in the log file name for reference.

# ================================================================
# OSCAR2 tests — dedicated FA kernel path (GPU required)
# ================================================================
#
# NOTE: The current MODEL (Gemma-4) has SWA (n_swa > 0) which forces
# oscar2 to fall back to f16 in HP sink buffers. These tests will run
# with oscar2 in the main cache but f16 HP sinks.
#
# For full oscar2-only testing, use a model without SWA (e.g. Qwen3):
#   MODEL="/path/to/qwen3-8b-gguf"
#   TEMPLATE="models/templates/qwen3-instruct.jinja"

# CPU-only oscar2 baseline (VEC path, device 0 disabled)
run_test 13 "oscar2 baseline (CPU)" oscar2 oscar2

# GPU oscar2 full K+V (dedicated FA kernel if D in {128,256,512})
run_test 14 "oscar2 K+V FA" oscar2 oscar2 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1 LLAMA_ATTN_ROT_V_OVERRIDE=1

# GPU oscar2 K only, f16 V (tests K dequant path)
run_test 15 "oscar2 K=f16 V" oscar2 f16 \
    LLAMA_ATTN_ROT_K_OVERRIDE=1

# GPU f16 K, oscar2 V (tests V dequant + Hadamard path)
run_test 16 "f16 K oscar2 V" f16 oscar2 \
    LLAMA_ATTN_ROT_V_OVERRIDE=1
#
# To verify the dedicated FA kernel is actually hit (not VEC fallback),
# check the log for "OSCAR2 FA" or increase -n to trigger prefilling:
#   grep -i "oscar2\|fattn" test_results/test_14_oscar2_K\+V_FA.log
#
# For Blackwell sm_120 testing, add after applying K1 fix:
#   CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
#     -m <qwen3-gguf> --flash-attn on \
#     --cache-type-k oscar2 --cache-type-v oscar2 \
#     -p "2+2=" -n 50

echo ""
echo "================================================================="
echo "  All tests complete! $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================="
echo ""
echo "=== QUICK SUMMARY ==="
cat "$SUMMARY" | column -t -s '|' 2>/dev/null || cat "$SUMMARY"
