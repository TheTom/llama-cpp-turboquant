# OSCAR INT2 Cross-Port — Status Document

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

### CUDA Decode — VEC Path Support (committed)
| File | Content |
|------|---------|
| `ggml/src/ggml-cuda/fattn-common.cuh` | `vec_dot_fattn_vec_KQ_oscar2`, `dequantize_V_oscar2`, dispatch entries |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | `nthreads_KQ_for_dot` routing for OSCAR2 |
| `ggml/src/ggml-cuda/fattn.cu` | VEC template instantiations + kernel routing |

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
| oscar2/oscar2 (dedicated kernel) | Garbled* | 29 t/s | ✗ Bug in kernel — B1/B2/B5/B8/B13 fixed; B4/B6 remain |
| oscar2/oscar2 (VEC path) | Garbled† | 86 t/s | ✗ Domain mismatch: set_rows stores Hadamard, VEC reads natural |

### Cache Compression
| Format | Bytes/128-elem | vs f16 |
|--------|---------------|--------|
| f16 | 256 | 1× |
| q2_0 | 40 | 6.4× |
| OSCAR2 | **36** | **7.1×** |

OSCAR2 saves 10% more VRAM than q2_0 and 7× vs f16. The cache IS compressed on GPU
(SET_ROWS kernel verified correct). The decode path is what needs fixing.

---

### 1. Dedicated OSCAR2 FA Kernel Produces Garbage — PARTIALLY FIXED
The kernel dispatches correctly (29 t/s vs 90 t/s for VEC) but attention output was
incoherent. Debugging showed:

**Bug fix summary (committed since initial report):**
- `B1` — Duplicate KQ dot accumulation (D<128): FIXED
- `B2` — Non-multiple-of-128 head dim truncation: FIXED (`static_assert`)
- `B3` — Column-bound check using wrong dimension: FIXED (bound + dst_ptr index both use `ne01.x`)
- `B5` — Hadamard inverse bounds/condition: FIXED (rewritten to clean loop)
- `B8` — Uninitialized arrays: FIXED (zero-init)
- `B13` — i_kv break bound unclamped: FIXED (min clamp on k_VKQ_max)
- `B21` — K mean omitted from QK logits (CRITICAL): FIXED (mean correction added to KQ dot)
- `B6` — nb11 stride assert: FIXED (assert added)
- `B3` — dst_ptr index for batch>1: FIXED (ne01.x replaces ne01.z)

**Still open:**
- `B4` — Mask indexing ignores stride parameters (only matters for non-standard mask layouts)

### 2. VEC Path Broken for Quantized KV at D>256 — FUNDAMENTAL
Pre-existing issue: VEC path fails because set_rows_cuda_oscar2 stores K/V in Hadamard
domain, but the VEC dequant path (`vec_dot_fattn_vec_KQ_oscar2`, `dequantize_V_oscar2`)
reads them as natural-domain values. The VEC kernel has no inverse-Hadamard transform
for K or V at any head dim. This affects ALL head dims (not just D>256), though D>256
has the additional template-instantiation gap.

**Not fixable without adding inverse Hadamard to VEC path.** For now, the dedicated FA
kernel (oscar2-only) is the correct path. Recommend adding a VEC-path exclusion for
oscar2 to avoid silent domain mismatch.

### 3. Rotation Matrices Not Loaded — FIXED (F17)
When a Gemma-4 GGUF lacks the optional calibrated `attn_k_rot`/`attn_v_rot` tensors,
the model loads a Hadamard-like fallback matrix `TURBO_ROTATION_RT` from
`src/turbo-rotation-data.h`. This replaces the previous identity fallback and restores
the incoherence-reduction benefit for quantized KV caches. Calibrated per-layer
rotations (from `export_rot_kv_gguf.py`) are still preferred when available.

**Limitation**: Fallback only works for `n_embd_head == 128`. D=256/512 fallback
generation needs a power-of-2 Hadamard generator.

### 4. HP (High-Precision) Sink Buffer — NOT ADDRESSED (feature, not bug)
Not implemented for OSCAR2. The HP buffer (f16 fallback for sink+recent tokens)
is a planned addition to recover quality at long contexts but is not needed for
correctness at short contexts.

### 5. SWA + OSCAR2 compatibility (HIGH)
Previously, any model with `n_swa > 0` was universally forced to f16 because
Gemma-4 has mixed head dims (SWA=128, dense=256). Simple SWA models with uniform
head dim (Gemma-2/3, Cohere Command R, etc.) were effectively blocked from oscar2.

A two-tier check in `src/llama-kv-cache.cpp::llama_kv_cache` now resolves this:
  * Uniform-head-dim SWA models are allowed wholesale (any D in {128, 256, 512}).
  * Mixed-head-dim models get a per-layer override in the cache constructor:
    layers whose `n_embd_head_k(il) == 128` keep `GGML_TYPE_OSCAR2`; layers
    whose head dim is unsupported (e.g. Gemma-4 dense layers at D=256) drop
    to `GGML_TYPE_F16`. The ISWA-aware cache slots SWA layers into the
    `swa` sub-cache and dense layers into the `base` sub-cache, so each
    sub-cache's per-layer check is uniform internally.

The legacy `oscar2_safe_for_swa` pre-scan is left in (for logging and as the
uniform-dim shortcut), but the redundant global guard inside the per-layer
block was removed because its filter-agnostic scan would have incorrectly
flagged every Gemma-4 layer as f16, defeating the per-layer fix.

Verified compatible: Gemma-2 (uniform D=128|256), Gemma-3, Cohere Command R, DFlash.
Gemma-4: oscar2 on SWA layers (D=128); f16 fallback on dense layers (D=256).
---

## File Inventory

```
ggml/include/ggml.h                          enum, count
ggml/src/ggml-common.h                       block_oscar2 struct
ggml/src/ggml.c                              type traits, quantize_chunk
ggml/src/ggml-quants.h                       declarations
ggml/src/ggml-quants.c                       CPU ref quant/dequant
common/arg.cpp                               CLI arg
ggml/src/ggml-cuda/set-rows.cu               set_rows_cuda_oscar2 kernel
ggml/src/ggml-cuda/ggml-cuda.cu              SET_ROWS support
ggml/src/ggml-cuda/fattn-common.cuh          vec_dot/dequant + dispatch
ggml/src/ggml-cuda/fattn-vec.cuh             nthreads routing
ggml/src/ggml-cuda/fattn-oscar2.cuh          dedicated FA kernel
ggml/src/ggml-cuda/fattn.cu                  dispatch, instantiations, routing
```

Generated July 13, 2026.
