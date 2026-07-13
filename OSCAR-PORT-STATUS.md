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
No rotation matrices are available for Gemma-4-12B. The `LLAMA_OSCAR_K_ROTATION_PATH`
env vars are unimplemented (fallback to identity rotation). This degrades OSCAR2
quality vs the paper's reported results but should still produce coherent text.

### 4. HP (High-Precision) Sink Buffer
Not implemented for OSCAR2. The HP buffer (f16 fallback for sink+recent tokens)
is a planned addition to recover quality at long contexts but is not needed for
correctness at short contexts.

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
