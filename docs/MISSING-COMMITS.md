# OSCAR Port — Complete + Session 2 Fixes

All OSCAR-specific commits from `/mnt/storage/Projects/OSCAR-llamacpp/` have been
ported to turboquant.

## Ported OSCAR Commits

| Commit | Description | Status |
|--------|-------------|--------|
| `e5c2df99b` | Q2_0 INT2 KV cache type system, CPU quants, HP sink+recent buffer, graph integration | PORTED |
| `08ad957ef` | Calibrated OSCAR rotation (per-layer R·H·P), env-var gating, outlier clip | PORTED |
| `2a486987e` | Project README (OSCAR fork) | N/A for turboquant |
| `4570ea609` | OSCAR rotation matrices + GGUF baking script | PORTED (oscar-rotation/) |
| `940c77759` | qwen3-4B-thinking 2507 support, test cases, HP mask fix | PORTED |
| `7e1019bf0` | Gemma4 rotation + iSWA HP-recent window | PORTED |
| `6f53b08ff` | fa3-like fused kernel - ggml API, CPU ref, graph fused FA path | PORTED |
| `9d26b4aa7` | Gemma/qwen fa3 expansion - conversion maps, graph fused FA prefill | PORTED |
| `f71f50fc2` | CUDA q2_0 GPU port (SET_ROWS, VEC FA, vec_dot_q2_0_q8_1) | PORTED |
| `95cb84d1e` | CUDA flash_attn_ext_mixed + dispatch routing | PORTED |

## Session 2 Fixes Applied (2026-07-11)

### Fixed — Verified by Build

| Fix | File | Root Cause |
|-----|------|-----------|
| V loop KQ read OOB | `fattn-vec.cuh:412-415` | V loop read KQ[32..127] from uninitialized shared memory (59% stale data) |
| Dequantize_V OOB | `fattn-vec.cuh:450,458` | `threadIdx.x` ranged 0..127 but only `nthreads_V=32` valid elements — fixed with `% nthreads_V` |
| HP context OOM | `llama-kv-cache.cpp:188` | Context reservation didn't account for k_hp/v_hp tensors — added `(n_hp_total>0 ? 2u : 0u)` |
| HP concat-softmax type mismatch | `llama-graph.cpp:2814-2828` | KQ masks had mismatched types (F16 vs F32) between LP and HP tiers — cast to F32 |
| HP concat-softmax empty guard | `llama-graph.cpp:2797` | Graph built before any tokens — `get_n_hp_kv()==0` caused zero-extent concat — added `>0` check |

### Verified — Working

- **f16 KV cache**: Coherent output (`<channel>` format, correct answers)
- **All quantized types (q2_0, q4_0, q8_0)**: Load, allocate, generate tokens — but output is `<unused49>` on BOTH CPU and GPU

## Root Cause — Quantized KV Incoherence

**Diagnosis (2026-07-11):** ALL quantized KV cache types (q2_0, q4_0, q8_0) produce
`<unused49>` garbage on the CPU flash attention path. f16 KV produces coherent
output. This is NOT a q2_0-specific or VEC kernel issue — it affects all quantized
types in the iSWA cache.

**Evidence:**
- f16 CPU: `2+2=` → `4` (correct)
- q8_0 CPU: `2+2=` → `<unused49>...` (garbage)
- q4_0 CPU: same garbage
- q2_0 CPU: same garbage
- q8_0 GPU: same garbage via VEC kernel

**Likely cause:** The CPU flash attention path
(`ggml_compute_forward_flash_attn_ext_f16` in `ops.cpp`) reads quantized K/V rows
using byte-level strides (`nbk1`, `nbv1`). For f16, these strides use the natural
element size. For quantized types, the stride calculation
(`ggml_row_size(type, ne[0])`) produces a different layout that doesn't match what
the vec_dot/dequant functions expect from the iSWA sub-cache's tensor organization.

This predates the OSCAR port and is upstream in turboquant's iSWA cache
implementation (from TheTom/llama-cpp-turboquant).

**Reference:** vLLM PR #46774 implements OSCAR INT2 with 0.07% accuracy drop using
purpose-built Triton kernels, confirming the INT2 approach works without the
VEC kernel complexity or HP buffer.

## Deferred Items

### Second fused FA gate for non-iSWA build_attn (from 9d26b4aa7)
Performance optimization for prefill with fused mixed FA. Would require porting HP
integration to another `build_attn` overload. Deferred — current HP concat-softmax
handles correctness.

### Metal kernel files
8 files (~1,200 LOC) in `ggml/src/ggml-metal/*`. Not needed for CUDA-only target.

## Acknowledgements

- vLLM PR #46774 by zhangj1an for reference implementation
- OSCAR paper (arXiv:2605.17757) by Zhongzhu Zhou et al.

*Last updated: 2026-07-11*
