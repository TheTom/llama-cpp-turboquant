# OSCAR q2_0 Port — What's Done vs. What's Missing (verified 2026-07-11)

Working tree: `/mnt/storage/Projects/turboquant/`
Upstream reference: `/mnt/storage/Projects/OSCAR-llamacpp/` (commits e5c2df99b, f71f50fc2, 95cb84d1e)

This file tracks progress against OSCAR's q2_0 KV-cache work. Items marked
DONE were confirmed by building. Items marked TODO are not yet in the tree.

---

## DONE — Type System (aa711ad85 + this session)

| File | Addition | State |
|------|----------|-------|
| `ggml/include/ggml.h` | `GGML_TYPE_Q2_0 = 47` | DONE |
| `ggml/src/ggml-common.h` | `block_q2_0` struct, `QK2_0 = 32` | DONE |
| `ggml/src/ggml.c` | type-info table entry | DONE |
| `ggml/src/ggml-quants.c/.h` | `quantize_row_q2_0_ref`, `dequantize_row_q2_0`, `quantize_q2_0` | DONE |
| `common/arg.cpp` | `--cache-type-k q2_0` dispatch | DONE |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `[GGML_TYPE_Q2_0]` type-traits entry (from_float / vec_dot / vec_dot_type) | **ADDED this session** |
| `ggml/src/ggml-cpu/arch-fallback.h` | `ggml_vec_dot_q2_0_q8_0_generic` remap (x86_64 + others) | **ADDED this session** |
| `ggml/src/ggml-cpu/ops.cpp` | `case GGML_TYPE_Q2_0:` in clamp + other switches | **ADDED this session** |
| `ggml/src/ggml-cpu/quants.c` | `ggml_vec_dot_q2_0_q8_0_generic` impl (Lloyd-Max dequant + q8_0 dot) | **ADDED this session** |
| `ggml/src/ggml-cpu/quants.h` | `ggml_vec_dot_q2_0_q8_0` prototype | **ADDED this session** |

Verification: `ninja -C build -j8` links all 109 targets, no undefined
reference to `ggml_vec_dot_q2_0_q8_0`.

---

## DONE — CUDA Kernels (aa711ad85)

| File | What |
|------|------|
| `ggml/src/ggml-cuda/fattn-common.cuh` | `vec_dot_fattn_vec_KQ_q2_0`, `dequantize_V_q2_0`, dispatcher |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | dispatcher entries (Fixes 1–3 APPLIED in working tree) |
| `ggml/src/ggml-cuda/fattn.cu` | `FATTN_VEC_CASES_ALL_D`, forced `BEST_FATTN_KERNEL_VEC` |
| `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_q2_0_q8_1` |
| `ggml/src/ggml-cuda/set-rows.cu` | `set_rows_cuda_q2_0` (Lloyd-Max) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `device_supports_op` gate |
| `template-instances/fattn-vec-instance-q2_0-*.cu` | D=64/128/256/512 instances |

### CUDA kernel fix state
- **Fix 1** (`k` formula, fattn-vec.cuh:412): APPLIED
- **Fix 2** (`VKQ_tmp` offset, fattn-vec.cuh:676/692): APPLIED
- **Fix 3** (`i_VKQ` formula, fattn-vec.cuh:686/702): APPLIED
- **Fix 4** (`vec_dot` nthreads → nthreads_V, fattn-common.cuh:1011): **NOT APPLIED** —
  this is the primary cause of current garbage output. Line still reads
  `vec_dot_fattn_vec_KQ_q2_0<D, nthreads>` with `nthreads = 128`.

---

## DONE (compiles, NOT wired) — HP KV-cache C++ (this session)

`src/llama-kv-cache.{h,cpp}` now contains the HP scaffolding:

- `slot_info::hp_idxs` / `hp_batch_idxs` vectors (per stream)
- `n_kv_sink` / `n_kv_recent` / `n_hp_total` read from `LLAMA_KV_HP_SINK` /
  `LLAMA_KV_HP_RECENT` env vars in the constructor
- HP tensors `k_hp` / `v_hp` (GGML_TYPE_F16, size = n_hp_total) allocated per
  layer, with `k_hp_stream` / `v_hp_stream` views
- `v_hp_cells` + `hp_positions` tracking vectors resized in constructor
- Accessors: `get_n_hp_kv`, `get_k_hp`, `get_v_hp`, `cpy_k_hp`, `cpy_v_hp`,
  `build_input_hp_k_idxs`, `build_input_hp_batch_idxs`, `build_input_hp_kq_mask`,
  `set_input_hp_k_idxs`, `set_input_hp_batch_idxs`, `set_input_hp_kq_mask`

Verification: `llama-kv-cache.cpp` compiles. Three errors were fixed this
session (`ggml_copy_rows` → `ggml_set_rows`; `hp_k_idxs` → `hp_idxs` member
name; `void*` subscript in mask fill).

**NOT YET DONE within this file:**
- `find_slot` does not populate `sinfo.hp_idxs` / `sinfo.hp_batch_idxs`
- `apply_ubatch` does not update `v_hp_cells` / `hp_positions`

---

## TODO — Graph HP Integration (Phase 3, NOT STARTED)

`src/llama-graph.{h,cpp}` still needs:

1. `llm_graph_input_attn_kv` members: `hp_k_idxs`, `hp_batch_idxs`,
   `hp_kq_mask`, `hp_kq_mask_cnv`
2. `build_attn_inp_kv_impl`: build HP inputs when `mctx_cur->has_hp()`
3. `llm_graph_input_attn_kv::set_input`: call `set_input_hp_*`
4. `can_reuse`: rebuild graph if HP batch count changed
5. `build_attn_mha` / `build_attn`:
   - cast quantized V to F32 before transpose (`ggml_cont` on transposed
     quantized tensors is invalid)
   - write HP K/V via `cpy_k_hp` / `cpy_v_hp`
   - LP+HP joint attention: separate LP (q2_0) and HP (F16) KQ scores,
     concat along KV dim, joint softmax with combined mask, split weights,
     sum weighted V contributions

`llama_kv_cache_context` (the graph-facing wrapper) also needs `has_hp()`,
`get_n_hp_kv`, `get_k_hp`, `get_v_hp`, `cpy_k_hp`, `cpy_v_hp`, and the
`build_input_hp_*` / `set_input_hp_*` entry points forwarded to `kv`.

---

## TODO — Optional / Conditional

- `set_rows_cuda_q2_0` type mismatch fix from OSCAR 95cb84d1e
  (`blk.m = __float2half(mean)` → `blk.m = mean`) — apply only if V
  corruption persists after Fix 4 + HP graph land.
- Metal backend (`ggml-metal/*`) — not needed for CUDA-only RTX 5090 build.

---

## Why q2_0 Garbage Without Fix 4 + HP Graph

1. **Bug D (Fix 4 missing)**: vec_dot covers only half the dot product →
   uniform attention → `<unused49>` output.
2. **No HP graph**: even with Fix 4, raw 2-bit loses attention-sink and
   recency signal. OSCAR keeps sink+recent tokens in F16 (HP buffer) and the
   bulk in q2_0 (LP buffer); that mechanism is not yet connected on the graph
   side.

Enable at runtime with:
```bash
export LLAMA_KV_HP_SINK=4      # first 4 tokens in F16
export LLAMA_KV_HP_RECENT=128  # last 128 tokens in F16
```

---

*Updated: 2026-07-11 — reflects verified build state after this session's
CPU type-trait and HP KV-cache C++ work. Fix 4 and graph Phase 3 remain open.*
*By: Hermes agent (Jarvis persona)*
