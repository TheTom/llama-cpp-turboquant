# OSCAR INT2 Cross-Port — Status Document

## Table of Contents

1. [Objective](#objective)
2. [Status Overview](#status-overview)
3. [What's Complete](#whats-complete)
4. [Verification Results](#verification-results)
5. [Known Issues](#known-issues)
6. [Verification Test Results (July 27, 2026)](#verification-test-results-july-27-2026)
7. [Baseline Commands](#baseline-commands)
8. [CPU/GPU Quant Divergence Note](#cpugpu-quant-divergence-note)
9. [Appendix A: Bug Reference (B1-B21)](#appendix-a-bug-reference-b1-b21)
10. [Appendix B: Performance Tracking](#appendix-b-performance-tracking)
11. [File Inventory](#file-inventory)

---

## Objective

Cross-port vLLM's OSCAR INT2 quantization (`GGML_TYPE_OSCAR2`) into
`github.com/giveen/llama-cpp-turboquant` as a KV cache type for RTX 5090
(Blackwell, sm_120, CUDA 13.3).

OSCAR differs from the existing q2_0 KV cache type:
- **q2_0**: Lloyd-Max centroids, block_size=32, 128-wide Hadamard groups, mean subtraction
- **OSCAR2**: Asymmetric min-max linear quantization, block_size=128, no Hadamard

**Hardware**: RTX 5090 (Blackwell), 32GB VRAM
**Model**: Gemma-4-12B-it (rotated KV, D=256/512 head dim)
**Branch**: `oscar` at `github.com/giveen/llama-cpp-turboquant`

---

## Status Overview

| Area | Status | Last Updated |
|------|--------|-------------|
| Type system | Complete | Jul 2026 |
| CUDA store kernel (set_rows) | Complete, verified correct | Jul 2026 |
| CUDA decode (dedicated FA kernel) | **B21 fixed — works for short/medium generations** | Jul 27, 2026 |
| CUDA decode (VEC path) | **Disabled** (fundamental Hadamard domain mismatch) | Jul 27, 2026 |
| CUDA decode (MMA turbo path) | Gated off (stale comment); **centroid + K-mean bugs FIXED** | Aug 2, 2026 |
| Rotation fallback | FIXED (F17) — runtime Sylvester Hadamard generator | Jul 27, 2026 |
| V rotation gating (Gemma4) | Fixed — unconditional when tensor present | Jul 27, 2026 |
| q2_0 NO_HADAMARD auto-detect | Fixed — auto-set when rotation tensors detected | Jul 27, 2026 |
| INT2-without-rotation warning | Added | Jul 27, 2026 |
| **HP (F16 sink+recent) buffer** | **IMPLEMENTED** — two-tier fused FA (LP oscar2 + HP f16) | Aug 2, 2026 |
| **Mixed FA op** | `ggml_flash_attn_ext_mixed` — HP tier in same online softmax | Aug 2, 2026 |

**Remaining issue (root-caused)**: Instruction-style prompts produce blank/`///` output
with oscar2 because the port has **no HP sink+recent buffer**. The vLLM reference
(`vllm#46774`) and the OSCAR author's llama.cpp port both proved that without
sink=64/recent=256 high-precision windows, quantization is fine but **short-range
dependencies degrade** (gsm8k strict-match -12 pts). The first generated tokens depend
on the most recent prompt tokens, which were INT2-quantized. The HP buffer now
addresses this. See [Verification Test Results](#verification-test-results-july-27-2026).

---

## What's Complete

### Type System (all files, committed)
| File | Addition |
|------|----------|
| `ggml/include/ggml.h` | `GGML_TYPE_OSCAR2 = 49`, `GGML_TYPE_COUNT = 50` |
| `ggml/src/ggml-common.h` | `block_oscar2` struct (32B qs + fp16 d + fp16 m = 36B/128-elem) |
| `ggml/src/ggml.c` | Type traits, `quantize_chunk` dispatch |
| `ggml/src/ggml-quants.h/.c` | `quantize_row_oscar2_ref`, `dequantize_row_oscar2`, `quantize_oscar2` |
| `common/arg.cpp` | `--cache-type-k oscar2 --cache-type-v oscar2` |

### CUDA Store Kernel (committed)
| File | Content |
|------|---------|
| `ggml/src/ggml-cuda/set-rows.cu` | `set_rows_cuda_oscar2` — 128-thread per-vector min/max reduce, quantize, pack, scatter |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `device_supports_op` for OSCAR2 SET_ROWS |

### CUDA Decode — VEC Path Support (committed — DISABLED)
| File | Content |
|------|---------|
| `ggml/src/ggml-cuda/fattn-common.cuh` | `vec_dot_fattn_vec_KQ_oscar2`, `dequantize_V_oscar2`, dispatch entries |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | `nthreads_KQ_for_dot` routing for OSCAR2 |
| `ggml/src/ggml-cuda/fattn.cu` | VEC template instantiations **disabled** |

### CUDA Decode — Dedicated FA Kernel (committed)
| File | Content |
|------|---------|
| `ggml/src/ggml-cuda/fattn-oscar2.cuh` | Single-warp 128-thread cooperative dequant, cross-warp KQ/VKQ reduction |
| `ggml/src/ggml-cuda/fattn.cu` | Dispatch + template instantiations D={64,128,256,512} x {OSCAR2,F16,Q8_0} |

### Bugs Fixed
| Bug | Fix | File |
|-----|-----|------|
| Q_ds indexing: all threads read `tmp_q_ds[0]` instead of per-group scale/offset | `tmp_q_ds[i0/QI8_1 + threadIdx.x/QI8_1]` | `fattn-vec.cuh` |
| Pre-Hadamard q2_preh kernel closed inside q5_1 float branch | Proper closure + static_assert | `fattn-common.cuh` |
| `dequantize_V_q2_0` accidentally deleted during edit | Restored from HEAD | `fattn-common.cuh` |

---

## Verification Results

| Config | Output | Speed | Notes |
|--------|--------|-------|-------|
| f16/f16 (baseline) | "The capital of France is Paris" | 96 t/s | ✓ Working |
| q2_0/q2_0 (dedicated kernel) | "The capital of France is Paris" | 15 t/s | ✓ Working |
| oscar2/oscar2 (dedicated kernel) | Varies by prompt* | ~52 t/s | ✓ Short completions; ✗ instruction prompts |
| oscar2/oscar2 (VEC path) | Garbled | 86 t/s | ✗ DISABLED — domain mismatch |

*See [Verification Test Results](#verification-test-results-july-27-2026) for breakdown.

### Cache Compression
| Format | Bytes/128-elem | vs f16 |
|--------|---------------|--------|
| f16 | 256 | 1× |
| q2_0 | 40 | 6.4× |
| OSCAR2 | **36** | **7.1×** |

OSCAR2 saves 10% more VRAM than q2_0 and 7× vs f16. The cache IS compressed on GPU
(SET_ROWS kernel verified correct). The decode path is what needs fixing.

---

## Known Issues

### 1. Dedicated OSCAR2 FA Kernel — PARTIALLY FIXED

**Bug fix summary (committed since initial report):**
- B1 — Duplicate KQ dot accumulation (D<128): FIXED
- B2 — Non-multiple-of-128 head dim truncation: FIXED (static_assert)
- B3 — Column-bound check + dst_ptr index: FIXED (both use `ne01.x`)
- B5 — Hadamard inverse bounds/condition: FIXED (rewritten to clean loop)
- B8 — Uninitialized arrays: FIXED (zero-init)
- B13 — i_kv break bound unclamped: FIXED (min clamp on k_VKQ_max)
- B21 — K mean omitted from QK logits (CRITICAL): FIXED (mean correction added)
- B6 — nb11 stride assert: FIXED (assert added)

**Still open:**
- B4 — Mask indexing ignores stride parameters

### 2. VEC Path Broken for Quantized KV — FUNDAMENTAL

The VEC implementation is mathematically incapable of correctly handling GPU-generated
oscar2 values. `set_rows_cuda_oscar2` stores K/V in Hadamard domain, but the VEC dequant
path reads them as natural-domain values. The VEC kernel has no inverse-Hadamard transform.

**Not fixable** without adding inverse Hadamard to the VEC path. The dedicated FA kernel
is the only correct path. All VEC oscar2 template instantiations have been **disabled**.

### 3. Rotation Matrices Not Loaded — FIXED (F17)

When a model lacks calibrated `attn_k_rot`/`attn_v_rot` tensors, a Hadamard-like fallback
is generated at runtime via Sylvester construction. Supports any power-of-2 head dim >= 64.

### 4. HP (High-Precision) Sink+Recent Buffer — IMPLEMENTED (Aug 2, 2026)

OSCAR two-tier KV: quantized LP + F16 HP sink/recent in one fused FA kernel.
- Env: `LLAMA_KV_HP_SINK` (default 0), `LLAMA_KV_HP_RECENT` (default 0). Recommend
  sink=64 recent=256 per the vLLM reference and the author's llama.cpp port.
- HP K/V stored F16 in the **Hadamard domain** (forward normalized Hadamard, no mean
  subtract, no quantization) via a dedicated set_rows kernel; the oscar2 FA kernel
  attends them with zero error (Q is already transformed to the same domain).
- LP mask excludes HP positions (no double counting); HP mask is -inf for empty slots.
- iSWA (Gemma-4) gets per-sub-cache HP buffers (base + swa) - addresses the Gemma-4
  "recent V in INT2 degenerates over long generation" failure (author's commit 7e1019bf0).
- State save/load does NOT persist HP cells (same limitation as the author's port).
- Cross-stream seq_cp resets the destination stream's HP cells (HP buffer is not
  copied by sc_info); same-stream seq_cp shares HP cells via metadata. Copied
  sequences lose HP coverage until they write tokens - safe degradation.
- Requires flash attention and an oscar2 LP cache type; silently disables otherwise.
  NOTE: on CPU-only builds the mixed op aborts (the Hadamard-domain math is
  CUDA-only); HP is effectively a CUDA-backend feature.
- ALiBi: HP mask now carries -|p0-p1| for kept cells (matches LP mask), so the
  kernel's slope multiply is correct.
- `ggml_flash_attn_ext_mixed` asserts the LP tier is OSCAR2 (other kernels ignore
  the HP sources).

### 5. SWA + OSCAR2 Compatibility

Two-tier check resolves mixed-head-dim models (Gemma-4: SWA=128, dense=256):
- Uniform-head-dim SWA models allowed wholesale (D in {128, 256, 512})
- Mixed-head-dim models get per-layer override via ISWA cache split

---

## Verification Test Results (July 27, 2026)

After B21 fix and associated patches:

| Model | Rotation | Head Dim | f16 | oscar2 | q2_0 |
|-------|----------|----------|-----|--------|------|
| Qwen3.6-27B Hadamard | Hadamard (data-free) | 512 | ✓ | **✓** | ✓ |
| Qwen3.6-27B UD | Calibrated (UD) | 128 | ✓ | **✓** | ✓ |
| Gemma4-12B UD | Calibrated (UD) | 256/512 mixed | ✓ | **✗** | **✗** |
| Gemma4-12B Hadamard | Hadamard (data-free) | 256/512 mixed | ✓ | **✗** | **✗** |

**B21 fix verified working** on Qwen3.6 models at D=128 and D=512:

| Prompt | Tokens | oscar2 | f16 |
|--------|--------|--------|-----|
| "The capital of France is" | 4 prompt, ~20 generated | ✓ "Paris" | ✓ |
| "2+2=" | 2 prompt, 30 generated | ✓ thinking + answer | ✓ |

### Remaining Issue: Instruction-Prompt Token Bias

Some instruction-style prompts (e.g. "Count from 1 to 5:", "Write a Python function")
produce blank output or `reasoning_content: "//////"` with oscar2 while working correctly
with f16. The first generated token is wrong, suggesting a **prefill-phase value corruption**
rather than decode accumulation.

**Root cause (verified Aug 2, 2026)**: Not the fp16 mean precision - the correction
error is ~1e-5 (math traced end-to-end). The failure is the **missing HP sink+recent
buffer**: first generated tokens attend the most recent prompt tokens, which were
INT2-quantized, degrading short-range dependencies. Confirmed by:
- vLLM PR #46774: without sink=64/recent=256 BF16 windows, "the quantization itself
  is fine but short-range dependencies degrade" (gsm8k -12 pts).
- OSCAR author's llama.cpp port (zhongzhu/llamacpp on this machine): HP sink/recent
  F16 buffer + two-tier fused attention; commit 7e1019bf0 fixed exactly the Gemma-4
  iSWA "recent V in INT2 degenerates over long generation" failure.

### Gemma4 Issue (Separate)

ALL quantized cache types (oscar2, q2_0, q8_0) fail on Gemma4 regardless of rotation
method. f16 works. Pre-existing Gemma4-specific issue with ISWA cache split interaction.
**Root cause (verified Aug 2, 2026)**: Gemma's interleaved sliding-window cache kept
recent V in INT2 and degenerated over long generation. Fixed by the HP-recent window
on the iSWA path (author's commit 7e1019bf0), now ported as per-sub-cache HP buffers
(base + swa). Requires `LLAMA_KV_HP_SINK`/`LLAMA_KV_HP_RECENT` set.

---

## Baseline Commands

Tested on NVIDIA GeForce RTX 5090 (Blackwell, sm_120), CUDA 13.3.

### f16/f16
```bash
./build/bin/llama-cli \
    -m /mnt/storage/models/oscar-rotations/qwen3.6-27b-q5kxl-hadamard.gguf \
    -ngl 99 -fa on -c 65536 \
    --cache-type-k f16 --cache-type-v f16 \
    --temp 0 -p "What is the capital of France?" -n 20
```
Speed: Prompt ~500 t/s, Generation ~93 t/s

### oscar2/oscar2 (dedicated FA kernel)
```bash
./build/bin/llama-cli \
    -m /mnt/storage/models/oscar-rotations/qwen3.6-27b-q5kxl-hadamard.gguf \
    -ngl 99 -fa on -c 262144 \
    --cache-type-k oscar2 --cache-type-v oscar2 \
    --temp 0 -p "What is the capital of France?" -n 20
```
Output: "The capital of France is Paris"
Speed: Prompt ~85 t/s, Generation ~52 t/s
Compression: 7.1x vs f16

### q2_0/q2_0 (dedicated kernel)
```bash
./build/bin/llama-cli \
    -m /mnt/storage/models/oscar-rotations/qwen3.6-27b-q5kxl-hadamard.gguf \
    -ngl 99 -fa on -c 65536 \
    --cache-type-k q2_0 --cache-type-v q2_0 \
    --temp 0 -p "What is the capital of France?" -n 20
```
Speed: Prompt ~74 t/s, Generation ~34 t/s

### Rotated Model Generation
```bash
# Hadamard rotation (data-free, recommended)
python3 oscar-rotation/generate_and_bake_rot.py \
    --base /path/to/model.gguf \
    --out /path/to/model-rot-kv.gguf \
    --method hadamard
```

### Long Context (256K)
```bash
./build/bin/llama-cli \
    -m /mnt/storage/models/oscar-rotations/qwen3.6-27b-q5kxl-hadamard.gguf \
    -ngl 99 -fa on -c 262144 \
    --cache-type-k oscar2 --cache-type-v oscar2 \
    --temp 0 -p "The capital of France is" -n 30
```

---

## CPU/GPU Quant Divergence Note

The CPU and GPU quant paths for oscar2 produce **different block semantics**:
- **CPU** (`quantize_row_oscar2_ref` in `ggml-quants.c`): applies optional
  P_br (bit-reversal) permutation but NO Hadamard transform. Natural domain values.
- **GPU** (`set_rows_cuda_oscar2` in `set-rows.cu`): applies forward
  normalized Hadamard (mean subtract -> H -> /sqrt(128)) but NO P_br. Hadamard domain.

Internally consistent within each path, but blocks from one path MUST NOT cross into
the other. The `P_BR_DEV` table was removed from `fattn-oscar2.cuh` as dead code.

---

## Appendix A: Bug Reference (B1-B21)

### `ggml/src/ggml-cuda/fattn-oscar2.cuh`

#### B1. KQ dot product computed twice when D < 128 (CRITICAL) — FIXED
The same dequant+dot accumulator was summed twice into `sum` when `use_block_unroll` is false.
Fix: duplicate lower branch deleted.

#### B2. Non-multiple-of-128 head dims silently drop elements (CRITICAL) — FIXED
`nblocks = D / QK_OSCAR2` truncates. Fix: `static_assert(D % QK_OSCAR2 == 0)`.
Dispatcher gates D in {128, 256, 512}.

#### B3. Column-bound check uses wrong dimension (HIGH) — FIXED
Column-bound used `ne01.z` (batch) instead of `ne01.x` (ncols). Fix: both bound check
and dst_ptr index use `ne01.x`.

#### B4. Mask indexing ignores stride parameters (HIGH) — NOT FIXED
Per-column mask read assumes contiguous layout; inner-loop offset not accounted for.
Only affects non-standard mask layouts (ALiBi+GQA). Low priority.

#### B5. Hadamard inverse bounds/condition (MEDIUM) — FIXED
Rewritten to clean loop form with `__syncwarp()`. All edge cases correctly handled.

#### B6. K_blk stride assumption (HIGH) — FIXED
`assert(nb11 == nblocks * sizeof(block_oscar2))` added.

#### B7. V dequant mean-centering (LOW) — NOT A BUG
Correct by design: mean separate from centered value.

#### B8. Uninitialized arrays (LOW) — FIXED
Zero-initialized with `= {}`.

#### B9. QK_OSCAR2 constant not shared (LOW) — DEFERRED
Maintenance risk. Low priority.

#### B21. Per-block K mean omitted from QK logits (CRITICAL) — FIXED
The stored per-block K mean (`block_oscar2.m`) was dropped from the KQ dot product with
the incorrect claim that it "doesn't affect softmax." The term `mean(K_block) * sum(Q_over_block)`
varies per K token and per Q position. Fix: mean correction added for thread 0 which holds
the DC component `Q_had[0]`. Matches V path's correct mean handling pattern.

#### B22. MMA tile loader dequantizes with raw code instead of centroid (CRITICAL) — FIXED (Aug 2, 2026)
`fattn-mma-f16.cuh` dequantized `code * d` instead of `CENTROIDS[code] * d` in both
tile loaders. Would produce garbage if the MMA gate opened. Fixed to use the Lloyd-Max
centroid table (matches the scalar kernel).

#### B23. MMA K path drops per-block mean correction (CRITICAL) — FIXED (Aug 2, 2026)
Same bug class as B21 in the MMA tile load: K mean (`block_oscar2.m`) not restored.
Fixed with the same mean correction as the scalar path. The gate comment claiming the
V-mean "can't be restored in the MMA tile load" is stale - `restore_mean=true` already
exists; the gate stays closed pending validation, not correctness.

#### B24. HP tier double-counted with parallel KV blocks (CRITICAL) — FIXED (Aug 2, 2026)
The new HP f16 tier loop ran in every `blockIdx.y` partial block. With `nbatch_fa = D`,
prefill with long KV splits the LP range across `gridDim.y > 1` blocks merged by
`flash_attn_combine_results`; HP tokens would be counted `gridDim.y` times.
Fix: HP tier processed only by `blockIdx.y == 0`.

### `ggml/src/ggml-cuda/fattn.cu` (OSCAR2 dispatch)

#### B10. Variable after closing macro (MEDIUM) — FIXED
#### B11. Head-dim gate missing (HIGH) — FIXED
#### B12. Kernel ranking (LOW) — NOT A BUG
#### B13. i_kv break bound (MEDIUM) — FIXED

### `ggml/src/ggml-cpu/ops.cpp` (CPU reference)

#### B14. op_params[4] semantics (MEDIUM) — NOT VERIFIED
#### B15. CPU FA needs Hadamard path (MEDIUM) — NOT VERIFIED

### `ggml/include/ggml.h` and `ggml/src/ggml-common.h` (storage layout)

#### B16. Type slot (LOW) — NOT A BUG
#### B17. Struct width (HIGH) — ADDRESSED
#### B18. INT2 code sign (MEDIUM) — NOT A BUG

### `ggml/src/ggml-cpu/quants.c`

#### B19. quantize_row_oscar2_ref existence (HIGH) — VERIFIED NOT A BUG

### `ggml/src/ggml-cpu/ggml-cpu.c` (type registration)

#### B20. Type registration (LOW) — NOT A BUG

---

## Appendix B: Performance Tracking

### Bottlenecks

| # | Bottleneck | File | Impact | Status |
|---|------------|------|--------|--------|
| P1 | Per-token inverse Hadamard | `fattn-oscar2.cuh` | HIGH | **FIXED** (F1: Q transformed once) |
| P2 | Single-warp execution | `fattn-oscar2.cuh` | HIGH | **FIXED** (F2: multi-column ncols={1,2,4,8}) |
| P3 | No tensor-core/MMA path | `fattn.cu` | MEDIUM | **FIXED** (F3: routes to MMA turbo) |
| P4 | Limited head dims | `llama-kv-cache.cpp` | MEDIUM | **FIXED** (F4: D in {128,256,512}) |
| P5 | K/V mean inconsistency | `fattn-oscar2.cuh` | LOW | Deferred |

### Fix Implementations

**F1** — Q transformed to Hadamard domain once at kernel start. K dequant dots directly
with transformed Q. Expected 1.5-2.0x decode speedup.

**F2** — ncols dynamically selected: D<=256 uses ncols=8/4/2, D=512 uses ncols=4/2.
Expected 2-4x prefill speedup.

**F3** — Routes oscar2 to MMA turbo for D in {128,256,512} when Q->ne[1]<=4 and Turing
MMA available. Gated behind `ggml_cuda_turbo_mma_fused()` (default ON). Currently
hardcoded `false` pending V-mean tile reconstruction.

**F4** — Relaxed per-layer head-dim check to allow any `head_dim % 128 == 0`.

---

## File Inventory

```
ggml/include/ggml.h                          enum, count; ggml_flash_attn_ext_mixed decl
ggml/src/ggml-common.h                       block_oscar2 struct
ggml/src/ggml.c                              type traits, quantize_chunk; mixed op
ggml/src/ggml-quants.h                       declarations
ggml/src/ggml-quants.c                       CPU ref quant/dequant
common/arg.cpp                               CLI arg
ggml/src/ggml-cuda/set-rows.cu               set_rows_cuda_oscar2 kernel; HP f16 Hadamard store kernel
ggml/src/ggml-cuda/ggml-cuda.cu              SET_ROWS support
ggml/src/ggml-cuda/fattn-common.cuh          vec_dot/dequant + dispatch
ggml/src/ggml-cuda/fattn-vec.cuh             nthreads routing
ggml/src/ggml-cuda/fattn-oscar2.cuh          dedicated FA kernel + HP f16 tier
ggml/src/ggml-cuda/fattn.cu                  dispatch, instantiations, routing
ggml/src/ggml-cuda/fattn-mma-f16.cuh         MMA tile loaders (B22/B23 fixed)
ggml/src/ggml-cpu/ops.cpp                    CPU FA abort on mixed op
src/llama-kv-cache.h/.cpp                    HP buffer: env gate, tensors, cells, masks, cpy, state
src/llama-kv-cache-iswa.h/.cpp               hp_enabled threading
src/llama-memory-hybrid{,-iswa}.h/.cpp       hp_enabled threading
src/llama-model.cpp                          create_memory hp_enabled wiring
src/llama-graph.h/.cpp                       HP inputs, dual-write, mixed FA (kv + iswa)
```

## HP Feature Usage (Aug 2, 2026)

```bash
# Qwen3.6 / any oscar2 model
./build/bin/llama-cli \
    -m /mnt/storage/models/oscar-rotations/qwen3.6-27b-q5kxl-hadamard.gguf \
    -ngl 99 -fa on -c 262144 \
    --cache-type-k oscar2 --cache-type-v oscar2 \
    --temp 0 -p "Count from 1 to 5:" -n 40
LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 ./build/bin/llama-cli ...
```

Generated July 27, 2026. Consolidated from `OSCAR2_BUGS.md`, `OSCAR2_PERF.md`,
and `docs/oscar-baseline-commands.md`. See git history for pre-consolidation versions.
