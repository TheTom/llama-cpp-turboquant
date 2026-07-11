# OSCAR q2_0 KV Cache Port — Status Document

## Objective

Port the OSCAR INT2 (q2_0, block_q2_0) quantization type from
`github.com/giveen/OSCAR-llamacpp` into `github.com/giveen/llama-cpp-turboquant`
(TheTom/llama-cpp-turboquant) as a working KV cache type that produces
**coherent generation at 16k+ context** on RTX 5090 (sm_120, CUDA 13.3).

TurboQuant already has three turbo types (tq2_0, tq3_0, tq4_0). The goal is
parity with these — q2_0 should be selectable via `--cache-type-k q2_0` and
produce correct, coherent output on par with f16 or turbo2 baselines.

**Hardware**: RTX 5090 (Blackwell, sm_120), 32GB VRAM
**Model**: Gemma-4-12B-it (rotated KV, 512-dim head, PEG-generated RoPE)
**Branch**: `oscar` at `github.com/giveen/llama-cpp-turboquant`

---

## 1. What Has Been Done (Commit aa711ad85)

### 1.1 Type System Port

Added **GGML_TYPE_Q2_0 = 47** (block_q2_0: 12-byte blocks of 32 elements,
Lloyd-Max 2-bit centroid encoding) across 14 files, mirroring OSCAR commits
e5c2df99b and f71f50fc2:

| File | Addition |
|------|----------|
| `ggml/include/ggml.h` | `GGML_TYPE_Q2_0 = 47`, bumped `GGML_TYPE_COUNT` |
| `ggml/src/ggml-common.h` | `block_q2_0` struct (d, m, qs[8]), `QK2_0 = 32` |
| `ggml/src/ggml.c` | Type info table entry (blck, type_size, to/from float) |
| `ggml/src/ggml-quants.c/.h` | `quantize_row_q2_0_ref`, `dequantize_row_q2_0`, `quantize_q2_0` |
| `ggml/src/ggml-cpu/quants.c/.h` | CPU wrapper functions |
| `common/arg.cpp` | `--cache-type-k q2_0` CLI dispatch |

### 1.2 CUDA Kernel Port

| File | What |
|------|------|
| `ggml/src/ggml-cuda/fattn-common.cuh` | `vec_dot_fattn_vec_KQ_q2_0` + `dequantize_V_q2_0` + dispatcher entries |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Extern `DECL_FATTN_VEC_CASE` declarations for D=64/128/256/512 |
| `ggml/src/ggml-cuda/fattn.cu` | `FATTN_VEC_CASES_ALL_D` dispatch entries + forced `BEST_FATTN_KERNEL_VEC` routing |
| `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_q2_0_q8_1` for CPU fallback path |
| `ggml/src/ggml-cuda/set-rows.cu` | `set_rows_cuda_q2_0` (Lloyd-Max encoding, per-128-group mean) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `device_supports_op` for `GGML_TYPE_Q2_0` |
| `template-instances/fattn-vec-instance-q2_0-q2_0.cu` | Template instantiations D=64/128/256/512 |

### 1.3 Verified Working Baselines

| Config | Context | Verdict |
|--------|---------|---------|
| f16/f16 KV cache | 16k, 512k | Coherent |
| turbo2/turbo2 KV cache | 16k, 262k, 512k | Coherent |
| q4_0/q4_0 KV cache (VEC FA) | 2k | Coherent |
| q2_0/q2_0 — unchanged code | 512 | **Garbage output** |

### 1.4 VRAM Benchmarks (262k context, sequential, no stacking)

| K/V Type | VRAM Used | vs f16 |
|----------|-----------|--------|
| q8_0/q8_0 | 12,857 MiB | baseline |
| turbo2/turbo2 | 11,057 MiB | -14% |
| q2_0/q2_0 | 10,499 MiB | **-18%** |

---

## 2. Debugging Journey (All Attempts So Far)

### 2.1 Attempt A — CLI Alias (rejected)

Mapped `--cache-type-k q2_0` → `GGML_TYPE_TURBO2_0` internally. Coherent but
user rejected: "you haven't even ported any code over yet."

### 2.2 Attempt B — Mixed Kernel (flash_attn_ext_mixed)

Ported 235-line `flash_attn_ext_mixed` kernel from OSCAR commit 95cb84d1e
(separate LP and HP tiers). Fixed output indexing bug (iq1/iq2 swap).
Result: garbage with `qs[0]=00` (all-zero V quantization codes). Abandoned
because OSCAR's mixed kernel has the same unresolved stride/layout bug.

### 2.3 Attempt C — Standard Attention Fallback (no FA)

Routed q2_0 to `BEST_FATTN_KERNEL_NONE` removing the FA requirement
(`llama-context.cpp`). Result: server OOM at 16k (standard attention needs
full f16 V buffer for cuBLAS matmul, which is 32GB for this model size).
Also blocked by missing q2_0 block-copy support in `ggml-cuda/cpy.cu`.

### 2.4 Attempt D — VEC Kernel

Forced `BEST_FATTN_KERNEL_VEC` for q2_0, added dispatch entries from OSCAR
f71f50fc2. Produces garbage at all contexts (512, 16k, 262k, 512k).
OSCAR fork itself crashes with `GGML_ABORT` at the same point (no q2_0
dispatch entries in upstream, even in OSCAR).

### 2.5 Element Math Verification (CONFIRMED CORRECT)

Built standalone GPU tests `/tmp/q2_verify2.cu` that call
`vec_dot_fattn_vec_KQ_q2_0` and `dequantize_V_q2_0` with synthetic data:
- All 32 threads match CPU reference
- Total diff: **1.907e-06** (FP roundoff only)
- Conclusion: element-level arithmetic is correct

### 2.6 Bug Isolation via Debug Trace

Added `printf` traces to `fattn-vec.cuh` at `D=256, nthreads_V=32` path:

**Bug A** — V position `k` collapses:
```
k = k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)
```
When `nthreads_V = 32 = WARP_SIZE`, the ternary yields `0` for ALL threads,
so each k0 iteration only processes 1 position instead of distributing
across 4 thread groups. 75% of V data never reaches the accumulator.

**Bug B** — VKQ shared memory offset collapses:
```
VKQ_tmp = (half2*)KQ + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2)
```
When `nthreads_V = 32 = WARP_SIZE`, offset is `0` for ALL 128 threads.
Threads 96-127 write past KQ[1024] (shared memory buffer), corrupting the
valid data from threads 0-31. GPU out-of-bounds writes silently corrupt
results rather than crashing (page-granular allocation on CUDA).

**Bug C** — i_VKQ formula uses `threadIdx.x` instead of `threadIdx.x % nthreads_V`:
```
i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2)
```
When `nthreads_V=32=WARP_SIZE`: i_VKQ ranges 0..254 (for 128 threads),
but only 0..62 is valid (max element index = 124 + 4 = 128, capped at
D=256). Threads 64+ dequantize out-of-bounds V elements, contaminating
the entire VKQ accumulator.

**Bug D** — K vec_dot called with `nthreads=128` not `nthreads_KQ=32`:

In `fattn-common.cuh` line 1011:
```cuda
return vec_dot_fattn_vec_KQ_q2_0<D, nthreads>;
```
Here `nthreads` is the kernel block dimension (128), NOT the KQ thread
count (32). The vec_dot function uses `nthreads` as its loop step:
```
for (k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += nthreads)
```
With `nthreads=128` and `D/sizeof(int)=64`: the loop body runs exactly
ONCE (step 128 > 64). Only elements 0..127 are covered; elements 128..255
are skipped. Threads 64+ compute on `ib = (k_KQ*4)/QK2_0 = 8..15` which
are past the 8 valid blocks of q2_0 data for D=256, reading stale memory.

After `warp_reduce_sum<nthreads_KQ=32>`, each warp gets only ONE HALF of
the dot product (elements 0..127 for warp 0, 128..255 for warp 1, stale
garbage for warps 2-3). The KQ score is never the full dot product.

All other quant types (q4_0, q8_0, turbo types) share the same dispatcher
pattern but do not use the VEC kernel path for non-q2_0 types — they fall
through to the MMA/TILE kernel which handles the dot product correctly.

---

## 3. Current Fix Strategy

Three changes to `fattn-vec.cuh` + one change to `fattn-common.cuh`:

### Fix 1 — k formula (fattn-vec.cuh:412)
```cuda
// BEFORE:
k = k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)
// AFTER:
k = k0 * (nthreads_V == WARP_SIZE ? (nthreads / nthreads_V) : 1) + threadIdx.x / nthreads_V
```
When nthreads_V=32=WARP_SIZE: `k = k0 * 4 + threadIdx.x / 32`.
With 4 thread groups covering `k = 0..127` across 32 k0 iterations,
each covering positions 0..127 (non-overlapping, all 128 positions).

### Fix 2 — VKQ_tmp offset (fattn-vec.cuh:676, 692)
```cuda
// BEFORE:
VKQ_tmp = KQ + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2)
// AFTER:
VKQ_tmp = KQ + (threadIdx.x / nthreads_V)*(D/2)
```
Each thread group writes to its own region: group 0 → KQ[0..255],
group 1 → KQ[256..511], group 2 → KQ[512..767], group 3 → KQ[768..1023].
No overlap, no corruption of adjacent group data.

### Fix 3 — i_VKQ formula (fattn-vec.cuh:686, 702)
```cuda
// BEFORE:
i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2)
// AFTER:
i_VKQ = i_VKQ_0 + (threadIdx.x % nthreads_V)*(V_rows_per_thread/2)
```
All threads, regardless of group, compute element indices within
0..124 (first i_VKQ iteration) and 128..252 (second), staying within
the D=256 head dimension. No out-of-bounds V reads.

### Fix 4 — vec_dot dispatcher nthreads (fattn-common.cuh:1011)
```cuda
// BEFORE:
return vec_dot_fattn_vec_KQ_q2_0<D, nthreads>;  // nthreads=128
// AFTER:
return vec_dot_fattn_vec_KQ_q2_0<D, nthreads_V>; // nthreads_V=32
```
Pass `nthreads_V` (=32) as the vec_dot loop step to ensure the
loop iterates `D/sizeof(int)/32 = 2` times, covering all 256
elements. Also add modulo wrapping to protect against OOB access:
```cuda
const int k_KQ = (k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads)) % int(D/sizeof(int));
```

---

## 4. Current Status (as of 2026-07-11 session)

| Item | Status |
|------|--------|
| Type system (GGML_TYPE_Q2_0) | DONE — ported in aa711ad85 |
| CUDA VEC kernel (fattn-common/vec/set-rows) | DONE — ported in aa711ad85 |
| CPU type traits (vec_dot, clamp, arch-fallback) | DONE — added this session, builds clean |
| fattn-vec.cuh Fixes 1–3 (k / VKQ_tmp / i_VKQ) | APPLIED (in working tree diff) |
| fattn-common.cuh Fix 4 (vec_dot nthreads → nthreads_V) | **NOT YET APPLIED** — root cause of garbage output |
| HP sink+recent buffer: KV-cache C++ (header + impl) | DONE this session — compiles, NOT yet wired to graph |
| HP sink+recent buffer: graph integration (llama-graph.*) | **NOT STARTED** |
| Server crash | FIXED — no longer segfaults |
| Output coherence (q2_0/q2_0) | **GARBAGE** ("<unused49>...") — Fix 4 + HP graph still missing |

### Verification

- Phase 1 (CPU type traits) verified: full `ninja -C build -j8` links all
  109 targets with no errors. `ggml_vec_dot_q2_0_q8_0` CPU path now exists and
  is remapped via arch-fallback.h for x86_64.
- Phase 2 (HP KV-cache C++): `llama-kv-cache.cpp/.h` compile cleanly after
  fixing three errors in this session (`ggml_copy_rows` → `ggml_set_rows`,
  `hp_k_idxs` → `hp_idxs` member name, `void*` subscript in mask fill).
  The HP buffer is allocated, accessors and cpy/input setters are implemented,
  but `find_slot`/`apply_ubatch` do NOT yet populate `sinfo.hp_idxs` /
  `sinfo.hp_batch_idxs`, and the graph does not yet read the HP buffer.

### Root cause of remaining garbage (unchanged from prior analysis)

Bug D: `vec_dot_fattn_vec_KQ_q2_0<D, nthreads>` is called with `nthreads = 128`
(the block dim) instead of `nthreads_V = 32`. The loop
`for (k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += nthreads)` then runs once
and covers only elements 0..127 of a 256-element dot product. KQ scores are
random; softmax is uniform; output collapses to `<unused49>`.

Additionally, even after Fix 4, q2_0 at 2-bit needs the HP sink+recent buffer
(graph side) to be coherent — raw 2-bit without HP loses attention-sink and
recency signal.

---

## 5. Remaining Work (Ordered)

1. **Apply Fix 4** to `ggml/src/ggml-cuda/fattn-common.cuh` (~line 1011):
   `vec_dot_fattn_vec_KQ_q2_0<D, nthreads>` → `vec_dot_fattn_vec_KQ_q2_0<D, nthreads_V>`.
   Add modulo wrapping in the vec_dot loop for OOB safety (optional).
   *This is the primary cause of current garbage output.*
2. **Wire HP buffer into `find_slot`** (`src/llama-kv-cache.cpp`): populate
   `sinfo.hp_idxs[s]` / `sinfo.hp_batch_idxs[s]` — sink tokens (pos < n_kv_sink)
   and recent tokens (pos > seq_max - n_kv_recent) get HP slot assignments.
3. **Wire HP buffer into `apply_ubatch`**: evict/update `v_hp_cells` and
   `hp_positions` on each batch (mirror the LP cell updates).
4. **Graph Phase 3** (`src/llama-graph.{h,cpp}`):
   - Add `hp_k_idxs` / `hp_batch_idxs` / `hp_kq_mask` / `hp_kq_mask_cnv` to
     `llm_graph_input_attn_kv`.
   - In `build_attn_inp_kv_impl`: build HP inputs when `mctx_cur->has_hp()`.
   - In `llm_graph_input_attn_kv::set_input`: call `set_input_hp_*`.
   - In `can_reuse`: rebuild graph if HP batch count changed.
   - In `build_attn_mha` / `build_attn`: cast quantized V to F32 before
     transpose; write HP K/V via `cpy_k_hp`/`cpy_v_hp`; run LP+HP joint
     attention (concat KQ scores, joint softmax with combined mask, split and
     sum weighted V contributions).
5. **Rebuild and coherence-test** with `LLAMA_KV_HP_SINK=4 LLAMA_KV_HP_RECENT=128`:
   c=512 first, then 16k, then 262k.
6. Apply `set_rows_cuda_q2_0` type mismatch fix from OSCAR 95cb84d1e
   (`blk.m = __float2half(mean)` → `blk.m = mean`) only if V corruption persists.
7. Push completed fix branch to `origin/oscar`.
8. Create canonical coherence verification script.

---

## 6. Key Architecture Insights

### 6.1 nthreads_V == WARP_SIZE Edge Case

The VEC kernel was designed for two regimes:
- **nthreads_V < WARP_SIZE** (turbo types, nthreads_V=4): multiple thread
  groups per warp, each group processes different V columns, V_cols_per_iter
  = WARP_SIZE/nthreads_V = 8, k0 loop steps by 8
- **nthreads_V > WARP_SIZE** (f16/bf16 unquantized): single position per
  V iteration, register-level vectorization

The q2_0 case `nthreads_V == WARP_SIZE` falls into a gap: the code has
special-case ternaries (`nthreads_V == WARP_SIZE ? 0 : ...`) that were
designed as "not implemented yet" sentinels — OSCAR never shipped a
working implementation for this case.

The fix distributes the 128 threads across 4 thread groups of 32 each,
with each group processing 32 distinct V positions and writing to separate
shared memory regions. The output warp-accumulation loop reads from all
4 regions, summing the contributions from each group's positions.

### 6.2 K Dot Product Threading

The vec_dot_q2_0 function distributes `D/sizeof(int) = 64` key-value pairs
across `nthreads` threads. With the block dimension of 128, the loop step
of 128 means only ONE iteration covers 64 elements. The fix uses `nthreads_V`
(=32) instead, giving 2 iterations covering all 64 pairs exactly.

All other quant types share the same dispatcher pattern but route through
different kernel paths (MMA/TILE) for non-q2_0 configurations, masking
the issue. The forced VEC routing for q2_0 exposes it.

### 6.3 `-use_fast_math` Flag

Turboquant's CMakeLists.txt had a hardcoded `-use_fast_math` flag on line
218 that causes silently wrong results on sm_120 (Blackwell). This was
removed early in the debugging process after false-positive coherence
reports.

---

## 7. References

- **OSCAR fork**: https://github.com/giveen/OSCAR-llamacpp
  - Local clone: `/mnt/storage/Projects/OSCAR-llamacpp/`
- **OSCAR upstream (calibration + rotation tools)**: https://github.com/JiashuWu/OSCAR
  - Local clone: `/mnt/storage/Projects/OSCAR/`
- **TurboQuant fork (oscar branch)**: https://github.com/giveen/llama-cpp-turboquant/tree/oscar
  - Local working dir: `/mnt/storage/Projects/turboquant/`
- **Commit aa711ad85**: Port OSCAR block_q2_0 KV cache (14 files, +398 lines)
- **Block size**: block_q2_0 = 12 bytes / 32 elements = 3 bpw (Lloyd-Max 2-bit)
- **Key hardware**: RTX 5090, sm_120, CUDA 13.3, gcc-15
- **Model**: Gemma-4-12B-it (rotated KV, 262144 ctx, 256-dim head)

### vLLM Related Links

- **TurboQuant PR (merged)**: https://github.com/vllm-project/vllm/pull/38479
  `[Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity`
  vLLM's implementation of per-vector asymmetric INT2 with Hadamard rotation.
  Uses separate scale/zero buffers + Triton decode kernel — format differs from
  block_q2_0 but validates the split-KV tiled attention approach.
- **OSCAR Feature Request**: https://github.com/vllm-project/vllm/issues/46221
  `Add a 2-bit KV-cache quantisation backend from OSCAR`
  Tracks porting OSCAR's INT2 to vLLM's kwarg-based QuantConfig system.
- **SpectralQuant (builds on TurboQuant #38479)**: https://github.com/vllm-project/vllm/issues/43475
  Successor to TurboQuant with improved structural covariance methods.

---

## 8. Build & Run Commands

### 8.1 GGUF Model

```
/mnt/storage/Projects/OSCAR-llamacpp/assets/gemma4-12b-rot/gemma-4-12b-it-rot-kv.gguf
```

This is a **rotated KV** variant of Gemma-4-12B-it — the RoPE is pre-applied
to K and V tensors in the model file so the KV cache stores post-rotated data.
Requires `done_getting_tensors(true)` in `src/llama-model.cpp` for partial
tensor loading (the `rot_kv` suffix tensors are consumed at load, not stored).

Chat template:

```
/mnt/storage/Projects/OSCAR-llamacpp/assets/gemma4-12b-rot/chat_template.jinja
```

Required env var for rotated models:

```bash
export LLAMA_KV_NO_HADAMARD=1
```

### 8.2 Clean Build (full rebuild)

Script: `/mnt/storage/Projects/turboquant/build_oscar.sh`

```bash
#!/bin/bash
# build_oscar.sh — clean build of turboquant fork with q2_0 KV cache support
set -e

TQ_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$TQ_DIR/build"
LOG_FILE="$TQ_DIR/build_oscar.log"

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

cd "$TQ_DIR"

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi

cmake -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_C_COMPILER=gcc-15 \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_FLAGS="-ccbin /usr/bin/g++-15 -isystem /usr/local/cuda/include" \
  -DCMAKE_CUDA_COMPILER_ID=NVIDIA \
  -DCMAKE_CUDA_COMPILER_VERSION=13.3 \
  -DCMAKE_CUDA_STANDARD_COMPUTED_DEFAULT=17 \
  -DCMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_LTO=ON \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_LINK_DEPENDS_USE_LINKER=OFF

ninja -C "$BUILD_DIR" -j8
```

### 8.3 Ad-Hoc Quick Rebuild (no full reconfigure)

For iterating on CUDA kernel fixes (fattn-vec.cuh, fattn-common.cuh):

```bash
cd /mnt/storage/Projects/turboquant
cmake -S . -B build -G Ninja \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null 2>&1 \
&& ninja -C build -j8
```

The first cmake invocation is needed after switching branches or modifying
CMakeLists.txt; subsequent iterations can run `ninja -C build -j8` directly.

### 8.4 Run Server

```bash
cd /mnt/storage/Projects/turboquant

LLAMA_KV_NO_HADAMARD=1 \
./build/bin/llama-server \
  -m /mnt/storage/Projects/OSCAR-llamacpp/assets/gemma4-12b-rot/gemma-4-12b-it-rot-kv.gguf \
  --cache-type-k q2_0 --cache-type-v q2_0 \
  -fa on -ngl 99 -c 512 \
  --chat-template /mnt/storage/Projects/OSCAR-llamacpp/assets/gemma4-12b-rot/chat_template.jinja \
  --port 8080
```

Flags explained:
| Flag | Purpose |
|------|---------|
| `LLAMA_KV_NO_HADAMARD=1` | Skip Hadamard transform (rotated model needs this) |
| `--cache-type-k q2_0` | K cache in q2_0 (GGML_TYPE_Q2_0, 3 bpw Lloyd-Max) |
| `--cache-type-v q2_0` | V cache in q2_0 |
| `-fa on` | Flash Attention (allows quantized KV FA path) |
| `-ngl 99` | Offload 99 layers to GPU (all) |
| `-c 512` | Context size (increase for longer tests: 16384, 262144) |

### 8.5 Quick Coherence Test

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"gemma4",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":20
  }'
```

Expected coherent output (f16 baseline):

```json
{"choices":[{"finish_reason":"length","content":"Hello! How can I help"}]}
```

Current q2_0 output (broken):

```json
{"choices":[{"finish_reason":"length","content":"\u003cunused49\u003e\u003cunused49\u003e..."}]}
```

### 8.6 Isolated VEC Kernel Test (standalone GPU)

Used to prove element math is correct independently of the full inference path:

```bash
nvcc -arch=sm_120 -o /tmp/q2_verify2 /tmp/q2_verify2.cu \
  -I /mnt/storage/Projects/turboquant/ggml/src/ggml-cuda \
  -I /mnt/storage/Projects/turboquant/ggml/include \
  -I /mnt/storage/Projects/turboquant/ggml/src
/tmp/q2_verify2
```

### 8.7 Build Verification Script

```bash
# /tmp/hermes-verify-q2-port.py — runs 16 infrastructure checks
# Last verified: all exit 0
python3 /tmp/hermes-verify-q2-port.py
echo $?   # expects 0
```

---

*Generated July 11, 2026. Contact: Jeremy Morales (giveen).*
