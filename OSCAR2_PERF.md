# OSCAR2 Performance Tracking

> Status: analysis complete, fix in progress.

## Bottlenecks Identified

### B1. Per-token inverse Hadamard in the attention inner loop
**File:** `ggml/src/ggml-cuda/fattn-oscar2.cuh`  
**Impact:** HIGH — affects both prompt processing and generation/decode speed.

The current `flash_attn_ext_oscar2` kernel inverse-transforms every K and every V token from the Hadamard domain back to the natural domain before dotting/accumulating. For each 128-element block this costs:
- 7 butterfly stages of `__syncwarp()`
- the `P_br` bit-reversal permutation shuffle
- ~768 FP ops per token per block

With `D=128` there is one block per head; with `D=512` there are four blocks per head, so the cost grows with head dimension.

### B2. Single-warp execution regardless of D or batch
**File:** `ggml/src/ggml-cuda/fattn-oscar2.cuh`  
**Impact:** HIGH — especially hurts prompt (prefill) throughput.

The kernel hard-codes `nwarps_k = 1` (32 threads). For prefill where `Q->ne[1] > 1`, only one warp is used while the rest of the GPU sits idle. Other flash-attention paths use `fattn-mma-f16` / `fattn-tile` with many warps and tensor cores.

### B3. No tensor-core/MMA path
**File:** `ggml/src/ggml-cuda/fattn.cu`  
**Impact:** MEDIUM — decode could use tensor cores for the Q@K^T and P@V matmuls.

`ggml_cuda_get_best_fattn_kernel` returns `BEST_FATTN_KERNEL_OSCAR2` and dispatches to the dedicated scalar kernel. There is no MMA equivalent for oscar2, so tensor cores are never used for the attention math itself.

### B4. Limited supported head dimensions
**File:** `src/llama-kv-cache.cpp`  
**Impact:** MEDIUM — memory/compute fallback for non-128-head models.

`llama-kv-cache.cpp` only keeps oscar2 for layers whose `head_dim == 128`; other head dims fall back to `GGML_TYPE_F16`. This reduces effective KV-cache compression for models like Gemma-4 that have mixed head dims (SWA=128, dense=256).

### B5. V path mean handling is inconsistent with K path
**File:** `ggml/src/ggml-cuda/fattn-oscar2.cuh`  
**Impact:** LOW (correctness/quality) — K transforms `centroid*d + m`, V subtracts `m`, transforms `centroid*d`, then re-adds `m`.

This is a latent inconsistency. A fully-correct Hadamard-domain pipeline would treat K and V the same way. The current results are coherent, so we preserve semantics exactly while optimizing.

## Proposed Fixes

| # | Fix | Priority | Status |
|---|-----|----------|--------|
| F1 | Move K/V attention to Hadamard domain: transform Q once, skip per-token inv-Hadamard/P_br | HIGH | implemented in `ggml/src/ggml-cuda/fattn-oscar2.cuh` |
| F2 | Add multi-column prefill path for oscar2 (ncols=4/8 for large Q->ne[1]) | HIGH | implemented in `ggml/src/ggml-cuda/fattn-oscar2.cuh` |
| F3 | Add MMA/Tensor-core path for oscar2 decode | MEDIUM | planned |
| F4 | Support head_dim ∈ {128,256,512} uniformly, extend beyond 128 where safe | MEDIUM | planned |
| F5 | Unify K/V mean handling to the mathematically-correct form | LOW | deferred |

## Implementation Notes for F1

Modified `ggml/src/ggml-cuda/fattn-oscar2.cuh`:
- Q is now transformed into the `P_br(H)` domain at kernel start, once per query token.
- K path: dequantized value is dotted directly with transformed Q; no per-token `P_br` or `inv-Hadamard`.
- V path: centered dequantized value is accumulated in the `P_br(H)` domain; per-block mean is tracked separately.
- Output: each 128-element block is inverse-transformed once, then the accumulated mean is added.
- This preserves the exact output semantics of the original kernel while removing the per-token transforms.

## Implementation Notes for F2

Modified `ggml/src/ggml-cuda/fattn-oscar2.cuh` launcher (`ggml_cuda_flash_attn_ext_oscar2_case`):
- ncols is now dynamically selected based on `Q->ne[1]` (prefill batch size) and head dimension `D`.
- D<=256: ncols=8 when Q->ne[1]>=8, ncols=4 when Q->ne[1]>=4, ncols=2 otherwise.
- D=512: ncols=4 when Q->ne[1]>=4, ncols=2 otherwise (capped to avoid register spilling).
- decode (Q->ne[1]=1): ncols=1 (unchanged).
- No kernel logic changes needed: the kernel already supports arbitrary ncols via template parameter.
- Register budget: ncols=8 at D=128 is ~108 regs/thread (~3.5K/block), ncols=4 at D=512 is ~200 regs/thread (~6.4K/block).
- Expected speedup: prefill should see 2-4x reduction in KV read redundancy.

## Build/Test Status

- CUDA `nvcc` is not available in this workspace, so F1 and F2 changes could not be compiled here.
- Next step: build locally with `-DGGML_CUDA=ON` and run a token-identity test against the previous build.
- F2 changes compile-time instantiates ncols={1,2,4,8} x lsc={true,false} = 8 new kernel variants per D value.

## Design Notes for F1

Because `H` (Hadamard) and `P_br` are both symmetric, self-inverse, and linear:

- `dot(Q, K_natural) = dot(P_br(H Q), P_br(H K))`
- `sum(w_i V_i_natural) = H^-1 P_br^-1 (sum(w_i P_br(H V_i_natural)))`

Therefore:
1. At kernel start, transform each 128-element block of Q with `P_br(H Q)` and store in `Q_reg`.
2. In the K dot path, dequantize K directly and dot with transformed Q. No per-token `P_br` or `inv-Hadamard`.
3. In the V accumulation path, accumulate the centered dequantized value `centroid*d` (still in the `P_br(H)` domain), keeping a separate per-block mean accumulator. After the KV loop, apply `P_br` then `inv-Hadamard` to the accumulated centered values, then add the accumulated mean.
4. This preserves the current kernel's exact output semantics while removing all per-token transforms.

Expected speedup:
- Decode: 1.5–2.0x on the attention phase (removes ~28 `__syncwarp()` calls per KV token for D=512).
- Prefill: also benefits because each prompt token no longer pays the per-KV transform cost.

## Validation Plan

1. Compile the CUDA backend (`ggml/src/ggml-cuda/fattn-oscar2.cuh`).
2. Run a small local forward pass with an model using `--type-k oscar2 --type-v oscar2` and verify token identity vs. the unoptimized build.
3. Benchmark prompt and decode tokens/second on a representative model (e.g., Qwen2.5 7B or Gemma-4 12B).

## Open Questions

- Should the F1 optimization be gated by an env var (e.g., `GGML_OSCAR2_FAST=1`) for the first release?
- How much shared memory is needed for a multi-warp F2 implementation?
