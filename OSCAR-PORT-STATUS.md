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
| oscar2/oscar2 (dedicated kernel) | Garbled | 29 t/s | ✗ Bug in kernel |
| oscar2/oscar2 (VEC path) | Garbled | 86 t/s | ✗ VEC path broken for quantized KV at D>256 |

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

### 1. Dedicated OSCAR2 FA Kernel Produces Garbage
The kernel dispatches correctly (29 t/s vs 90 t/s for VEC) but attention output is
incoherent. Debugging shows:
- `kv_max_ptr` is NULL for small prompts → kernel processes all `kv_size` slots
  (most with zero data). q2_0 has the same behavior and works correctly.
- K/V scale/zero and q-codes for populated positions are correct (match SET_ROWS output).
- Suspected bug in the cooperative dequant, KQ reduction, or VKQ accumulation.

### 2. VEC Path Broken for Quantized KV at D>256
Pre-existing issue, noted in code comment "VEC path is broken on Blackwell."
Affects ALL quantized types (q2_0, oscar2) through VEC at head dims > 128.
q2_0 works because its dedicated kernel handles D=64/128/256/512. The Q_ds
indexing fix was one bug; at least one more remains in the VEC kernel's
quantized-KV path.

### 3. Rotation Matrices Not Loaded
When a Gemma-4 GGUF lacks the optional calibrated `attn_k_rot`/`attn_v_rot` tensors,
the model now falls back to the data-free Hadamard matrix from `TURBO_ROTATION_RT`
(`src/turbo-rotation-data.h`). This replaces the previous identity fallback and
restores most of the incoherence-reduction benefit for quantized KV caches.
Calibrated per-layer rotations (from `export_rot_kv_gguf.py`) are still preferred
when available.

### 4. HP (High-Precision) Sink Buffer
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
