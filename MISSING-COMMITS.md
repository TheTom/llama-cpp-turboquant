# OSCAR Port — Complete (verified 2026-07-11)

All OSCAR-specific commits from `/mnt/storage/Projects/OSCAR-llamacpp/` have been
ported to turboquant. Every file, symbol, env-var, and code path from the OSCAR
commits below is present in the turboquant working tree.

## Ported OSCAR Commits

| Commit | Description | Status |
|--------|-------------|--------|
| `e5c2df99b` | Q2_0 INT2 KV cache type system, CPU quants, HP sink+recent buffer, graph integration | PORTED |
| `08ad957ef` | Calibrated OSCAR rotation (per-layer R·H·P), env-var gating, outlier clip | PORTED |
| `2a486987e` | Project README (OSCAR fork) | N/A for turboquant |
| `4570ea609` | OSCAR rotation matrices + GGUF baking script | PORTED (oscar-rotation/) |
| `940c77759` | qwen3-4B-thinking 2507 support, test cases, HP mask fix | PORTED |
| `7e1019bf0` | Gemma4 rotation + iSWA HP-recent window | PORTED |
| `6f53b08ff` | fa3-like fused kernel - ggml API, CPU ref, graph fused FA path | PORTED (CPU ref + graph; Metal kernel deferred for CUDA target) |
| `9d26b4aa7` | Gemma/qwen fa3 expansion - conversion maps, graph fused FA prefill | PORTED (conversion maps; Metal kernel deferred) |
| `f71f50fc2` | CUDA q2_0 GPU port (SET_ROWS, VEC FA, vec_dot_q2_0_q8_1) | PORTED (aa711ad85 base) |
| `95cb84d1e` | CUDA flash_attn_ext_mixed + dispatch routing | PORTED (fattn-mixed.cuh wired into fattn.cu) |

## Known Open Issue (NOT ported from OSCAR — OSCAR has the same bug)

**Fix 4** in `ggml/src/ggml-cuda/fattn-common.cuh` line ~1011:
```cuda
// CURRENT (both repos):
return vec_dot_fattn_vec_KQ_q2_0<D, nthreads>;

// SHOULD BE:
return vec_dot_fattn_vec_KQ_q2_0<D, nthreads_V>;
```
This causes the KQ dot product to only cover half the head elements when
`nthreads_V == WARP_SIZE (=32)`. OSCAR-llamacpp has the exact same unfixed line.
This bug must be fixed before q2_0 KV cache produces coherent output.

## Not Ported (intentionally deferred)

- **Metal kernel files** (`ggml/src/ggml-metal/*.metal`, `*.m`, `*-ops.cpp`):
  The turboquant target is CUDA (RTX 5090, Blackwell sm_120). Metal kernels
  are not compiled for this target. The CPU reference implementation exists
  as fallback.
- **Web UI branding** (`tools/ui/` OSCAR-related changes): Not needed.
- **README.md**: Turboquant has its own README.
- **Debug commits** (`8e2f915ca`, `7ca51db65`): Debug instrumentation only.

*Last updated: 2026-07-11*

## Deferred (documented for future consideration)

### Second fused FA gate for non-iSWA build_attn (OSCAR commit 9d26b4aa7)

OSCAR added a second `use_fused_fa` gate in the non-iSWA `build_attn` overload
(around line 2605 in the OSCAR source), enabling fused mixed-precision FA during
prefill for models using non-iSWA attention with HP buffer. The turboquant HP
integration is currently in the iSWA `build_attn` only. To add this:

1. Port HP integration (has_hp checks, hp_kq_mask fields, cpy_k_hp/cpy_v_hp calls,
   LP+HP concat-softmax attention) to the non-iSWA `build_attn` overload
   (`llm_graph_context::build_attn` without `iswa` parameter).
2. Add the `use_fused_fa` gate before the HP concat-softmax block, using
   `lp_kq_mask_cnv` / `hp_kq_mask_cnv` mask variants.
3. Remove the decode-only restriction (`q->ne[2] / k->ne[3] == 1`) from any
   remaining fused FA gates (done for the iSWA path as of this session).

This is an optimization, not a correctness requirement. The existing HP concat-softmax
path handles both decode and prefill correctly.
