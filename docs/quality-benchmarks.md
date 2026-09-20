# TurboQuant+ Quality Benchmarks

## Why These Benchmarks Matter

Speed without quality is useless. We claim 4.6× KV cache compression at 91-97% of q8_0 speed.
But we have ZERO quantitative quality data on the actual llama.cpp build. "Coherent text" and
Python cosine similarity (0.985) are not sufficient.

The llama.cpp CONTRIBUTING.md requires for new quant types:
1. Perplexity vs FP16/BF16
2. KL divergence data
3. Performance data via llama-bench

Papers (KIVI, QJL, RotateKV) all use wikitext-2 perplexity as the primary metric.
Prince Canuma validated with NIAH at 8K/32K/64K context.

## What We're Measuring

### 1. Perplexity (wikitext-2)
- **What**: How well the model predicts next tokens over a standard text corpus
- **Why**: Gold standard for LLM quality. Lower = better. Sensitive to KV cache quality.
- **Target**: turbo3 within 1% of q8_0 perplexity. If >2%, quality problem.
- **Comparison**: f16, q8_0, q4_0, q4_1, q5_0, turbo3

### 2. KL Divergence vs f16
- **What**: How different the output probability distribution is vs full precision
- **Why**: Measures distributional shift, not just top-token accuracy
- **Metrics**: mean KLD, delta-p RMS, same-top-p percentage
- **Required by**: llama.cpp CONTRIBUTING.md for upstream acceptance

### 3. Passkey Retrieval (built-in NIAH)
- **What**: Can the model retrieve a specific passkey from a long haystack?
- **Why**: Tests attention pattern preservation over long context
- **Comparison**: f16 vs turbo3 at various context lengths and needle positions

### 4. Generation Quality (qualitative)
- **What**: Side-by-side text generation comparison
- **Why**: Catches issues that aggregate metrics miss (repetition, coherence)

## Test Configuration

- **Model**: Qwen 3.5 35B-A3B MoE Q8_0 (primary), Qwopus v2 27B Q8_0 (secondary)
- **Hardware**: Apple M5 Max 128GB, Metal GPU
- **Dataset**: wikitext-2-raw (downloaded via scripts/get-wikitext-2.sh)
- **Context**: 512 tokens for perplexity (fast), 2048+ for NIAH
- **Chunks**: 8 for initial, 32 for final numbers

## Commands

### Download dataset
```bash
cd ~/local_llms/llama.cpp && bash scripts/get-wikitext-2.sh
```

### Perplexity suite
```bash
LLAMA=~/local_llms/llama.cpp/build-turbo/bin
MODEL=~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf
WIKI=~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw

for ct in f16 q8_0 q4_0 q4_1 q5_0 turbo3; do
  echo "=== $ct ==="
  $LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 512 \
    -ctk $ct -ctv $ct -fa on --chunks 8 -ngl 99 \
    2>&1 | tee results/ppl_${ct}.log
done
```

### KL Divergence
```bash
# Save f16 baseline logits
$LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 512 \
  -ctk f16 -ctv f16 -fa on --chunks 8 -ngl 99 \
  --kl-divergence-base results/f16_logits.kld

# Compute KL divergence for each cache type
for ct in q8_0 q4_0 turbo3; do
  $LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 512 \
    -ctk $ct -ctv $ct -fa on --chunks 8 -ngl 99 \
    --kl-divergence --kl-divergence-base results/f16_logits.kld \
    2>&1 | tee results/kld_${ct}.log
done
```

## Results

*To be filled after running benchmarks.*

### Perplexity (wikitext-2, 512 context)

| Cache Type | Bits/val | Perplexity | vs f16 | vs q8_0 |
|------------|----------|------------|--------|---------|
| f16 | 16 | — | baseline | — |
| q8_0 | 8 | — | — | baseline |
| q4_0 | 4 | — | — | — |
| q4_1 | 4.5 | — | — | — |
| q5_0 | 5.5 | — | — | — |
| **turbo3** | **3.5** | — | — | — |

### KL Divergence vs f16

| Cache Type | Mean KLD | Delta-p RMS | Same Top-p % |
|------------|----------|-------------|-------------|
| q8_0 | — | — | — |
| q4_0 | — | — | — |
| **turbo3** | — | — | — |

### Passkey Retrieval

| Cache Type | 1K | 2K | 4K | 8K |
|------------|----|----|----|----|
| f16 | — | — | — | — |
| q8_0 | — | — | — | — |
| turbo3 | — | — | — | — |

## INITIAL RESULTS — QUALITY FAILURE

### Perplexity (wikitext-2, 512 context, 8 chunks)

| Cache Type | Bits/val | Perplexity | vs f16 |
|------------|----------|------------|--------|
| f16 | 16 | 6.121 | baseline |
| q8_0 | 8 | 6.111 | -0.16% (better!) |
| q4_0 | 4 | 6.142 | +0.34% |
| **turbo3** | **3.5** | **165.6** | **+2607%** ❌ |

**turbo3 perplexity is 27× worse than f16. This is catastrophic.**

The model generates "coherent-looking" text but the actual predictions are garbage.
Speed benchmarks were meaningless — we were measuring how fast the model produces wrong answers.

### Root cause investigation needed
The block size 32 change + MSE-only + pre-rotate-queries may have introduced a bug:
1. The norm handling: storing full 128-element group norm in each 32-element block
   but dequant treats it as a per-block norm — scale mismatch?
2. The pre-rotate-queries: rotation matrix might not match the quantize rotation
3. The 3-bit index split (qs + signs) might have a packing error
4. The non-vec flash attention instantiation might use wrong nl parameter

MUST FIX before claiming any quality results.

### Bisect: block 128 → STILL PPL=165.6
Block size is NOT the issue. PPL identical at block 128 and block 32.

### Bisect: disable Q rotation → STILL PPL=165.6
Pre-rotate-queries is NOT the issue. PPL identical with/without Q rotation.

### Root cause #2 confirmed: dynamic_cast fails for MoE hybrid memory
The Q rotation and V inverse rotation NEVER execute because mctx is
llama_memory_hybrid (not llama_kv_cache). The kv_cache IS inside the
hybrid object but we can't reach it.

### What this means
Without Q rotation and without V inverse rotation:
- K cache is quantized in ROTATED space, dequanted in ROTATED space
- Q is in ORIGINAL space (no pre-rotation)
- <original_Q, rotated_K> = GARBAGE (cosine ≈ 0.02)
- V cache similarly in rotated space, output never inverse-rotated

The entire time, K/V values were in the wrong coordinate system.
The "coherent text" was an illusion from the language model's resilience.

### Fix approach
Need to get rotation tensors through the hybrid memory hierarchy.
Virtual methods on llama_memory_context_i are the cleanest path.
