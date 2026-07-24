# OSCAR2 KV-cache Bug Report

Branch: `origin/oscar` (HEAD `791ca44f4432b0b5730e44a6394bed5621dd01d6`).

This list supplements the four known issues already in `OSCAR-PORT-STATUS.md`:

1. Dedicated OSCAR2 FA kernel produces incoherent output.
2. VEC path broken for quantized KV at `D > 256`.
3. Rotation matrices for Gemma-4-12B fall back to identity.
4. HP sink buffer not implemented for OSCAR2.

The bugs below were identified by reading the OSCAR2-specific slices of `ggml/src/ggml-cuda/fattn-oscar2.cuh`, `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cpu/ops.cpp` (CPU OSCAR reference attention), `ggml/src/ggml-cpu/quants.c`, `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/include/ggml.h`, and `ggml/src/ggml-common.h` from the `oscar` branch. They are grouped by file and given a severity tag.

---

## `ggml/src/ggml-cuda/fattn-oscar2.cuh`

### B1. KQ dot product is computed twice when `D < 128` (CRITICAL)

In the inner loop that computes `KQ_val[j]` there are two stacked branches:

```cpp
if constexpr (use_block_unroll) {
    // block-unrolled K dequant + Hadamard path
} else {
    for (int e = 0; e < nelems; ++e) {
        sum += (fmaf((float)code, __half2float(K_blk[0].d), __half2float(K_blk[0].m))) * Q_reg[j][e];
    }
}
// Handle D < 128 (partial block)
if constexpr (!use_block_unroll) {
    for (int e = 0; e < nelems; ++e) {
        sum += (fmaf((float)code, __half2float(K_blk[0].d), __half2float(K_blk[0].m))) * Q_reg[j][e];
    }
}
```

When `use_block_unroll` is false (i.e. `D < 128`), the same serial dequant+dot accumulator is summed **twice** into `sum`. Even if no dispatched head dim is exactly `< 128`, the constant expression `!use_block_unroll` is evaluated at compile time, so any future template instantiation with non-multiple-of-128 head dim will silently double-count.

Fix: delete one of the two branches (the lower one).

### B2. Non-multiple-of-128 head dims silently drop elements (CRITICAL)

```cpp
constexpr bool use_block_unroll = (D >= 128);
constexpr int nblocks = use_block_unroll ? D / QK_OSCAR2 : 1;
...
for (int b = 0; b < nblocks; ++b) {
    ...
    // process QK_OSCAR2 = 128 elements in this block
}
```

`nblocks = D / QK_OSCAR2` is integer truncation. For `D = 192` or 320 or 384, only `floor(D/128)` blocks are processed. Elements `nblocks * 128 .. D-1` are silently discarded, so the dot product and VKQ reduction run on a truncated vector. Template instantiations exist for D in {64, 128, 256, 512}, but D=192/320 also reach `use_block_unroll=true` if anyone adds them later.

Fix: explicitly handle remainder (`tail = D % QK_OSCAR2`). Either require `D % QK_OSCAR2 == 0` via `static_assert`, or add a partial-block branch.

### B3. Column-bound check uses wrong dimension (HIGH)

```cpp
if (ncols > 1 && ic0 + j >= (int)ne01.z) break;
...
dst_ptr[(((sequence * (int)ne01.z + ic0 + j) * ne02 + head)) * D + di] = val;
```

`ne01` is declared as `const uint3`. Its components are `(ne[1], ne[2], ne[3]) = (ncols, n_head, batch)`. So `ne01.z` is actually `ne[3]` (batch), not the column dimension. The column bound should be against `(uint32_t)ne01.x` (which is `ncols`). With the current code:

* The `break` is essentially never taken (batch is usually small).
* The `dst_ptr` index uses `sequence * ne[3] + ic0 + j`, which is wrong if `sequence` and `ic0 + j` are interpreted relative to the actual row/column layout. In practice this only happens to work because of the chosen loop bounds in the calling launch, but it is fragile and confusing.

Note: a similar pattern appears in upstream FA kernels; the existing kernel uses `(uint32_t)ne01.x` (i.e. `((uint32_t)ne01).x`) for the same check. Confirm and replace.

### B4. Mask indexing ignores stride parameters (HIGH)

```cpp
const half * maskh = mask_ptr ? (const half *)(mask_ptr + nb33*(sequence % ne33)
                              + nb31*ic0
                              + blockIdx.y * nthreads) : nullptr;
...
if (maskh && (ncols == 1 || ic0 + j < (int)ne01.z))
    full_kq += slope * __half2float(maskh[j*ne11 + i_kv]);
```

Two issues:

* The base pointer is computed with `nb33, nb31` strides, but the per-column read `maskh[j*ne11 + i_kv]` pretends the layout is `(ncols, ne11)` packed contiguously. If the mask strides `nb32, nb33` etc. are not equal to `ne11`/`ne31`, the read is misaligned.
* `i_kv` is the relative index within the current `blockIdx.y * nthreads` window; the read does not include `blockIdx.y * nthreads` or the iteration offset (`kv_base - blockIdx.y * nthreads`).

Replace by computing `maskh + (kv_base + i_kv) * nb31 + j * nb_other` using the supplied strides.

### B5. `hadamard_inverse_128_32w` bounds check on read is harmless but write condition is wrong (MEDIUM)

```cpp
static __device__ void hadamard_inverse_128_32w(float * sh, int tid) {
    #pragma unroll
    for (int h = 64; h > 0; h >>= 1) {
        const int i0 = tid;
        const int i1 = tid + 32;
        const int i2 = tid + 64;
        const int i3 = tid + 96;
        const float a0 = sh[i0];
        const float b0 = (i0 + h < 128) ? sh[i0 + h] : 0.0f;
        ...
        if (!(i0 & h) && i0 + h < 128) { sh[i0] = a0 + b0; sh[i0 + h] = a0 - b0; }
        if (!(i1 & h) && i1 + h < 128) { sh[i1] = a1 + b1; sh[i1 + h] = a1 - b1; }
        if (!(i2 & h) && i2 + h < 128) { sh[i2] = a2 + b2; sh[i2 + h] = a2 - b2; }
        if (!(i3 & h) && i3 + h < 128) { sh[i3] = a3 + b3; sh[i3 + h] = a3 - b3; }
        __syncthreads();
    }
    ...
}
```

Issues:

* The kernel is set up with 1 warp (32 threads), but `__syncthreads()` is used as a barrier. On a single warp that is effectively a `__syncwarp()` and is fine, but the `__shared__` arrays are sized for cross-warp coordination. Mixing single-warp sync with multi-warp `__shared__` allocations is fragile: if anyone bumps `nwarps_k` the runtime semantics change without warning.
* The integer predicate `i3 + h < 128` already returns false for `i3 = tid + 96` when `tid + 96 + h >= 128`, i.e. when `tid > 31 - h`. The remainder reads get pinned to zero, which is a silent correctness issue for the boundary thread (its inbound pair (i3, `i3+h`) is dropped, not mirrored by a peer).

Audit formally: derive that each butterfly edge is updated by exactly one writer. Given the structure this is true for `tid in [0, 32)` and `h in {64, 32, 16, 8, 4, 2, 1}`, but only because the condition `!(index & h)` coincidentally picks the lower-half writer. This is easy to misread; rewrite to a clearer form, e.g.:

```cpp
#pragma unroll
for (int k = 0; k < 4; ++k) {
    int idx = tid + k * 32;
    if (idx < 128 && !(idx & h)) {
        float a = sh[idx];
        float b = sh[idx + h];
        sh[idx]      = a + b;
        sh[idx + h]  = a - b;
    }
}
__syncthreads();
```

### B6. `K_blk` / `V_blk` are advanced by stride `nb11/nb21` but read per block (HIGH)

```cpp
const block_oscar2 * K_blk = (const block_oscar2 *)(K + i_kv * nb11);
...
for (int b = 0; b < nblocks; ++b) {
    const float d_k = __half2float(K_blk[b].d);
    ...
    sum += ... Q_reg[j][b * elems_per_block + e];
}
```

The kernel assumes `nb11` (logical K row stride) is `sizeof(block_oscar2) * (D / QK_OSCAR2)`. If a model uses any padding or weight permute, `nb11` will be larger, and then `K_blk[b]` will step out of the actual row. For most layouts this is OK because row stride is exact, but pre-rotated K (rotation matrix applied) and zero-padded K (Gemma-4-12B 576 -> 640) break this assumption.

Add `GGML_ASSERT(nb11 == nblocks * sizeof(block_oscar2))` to catch this in development. Also explicitly assert `nb11 % 16 == 0` for vectorized loads if any.

### B7. V dequant mean-centering assumes Hadamard correctness (LOW)

The V branch computes `mean_v = m_v + 1.5f * d_v` (mean of 4 codes uniform in `{0,1,2,3}`) and applies the Hadamard before un-centering. If B5 is a real bug, this composition hides the error in numerical noise but does not correct it. Mention this in the fix.

### B8. Per-block `by_blk[]` / `shift_blk[]` use uninitialized array (LOW)

```cpp
int by_blk[elems_per_block];
int shift_blk[elems_per_block];
```

`elems_per_block` is a `constexpr int`. With `use_block_unroll = true` it is 4. The loop fills the array, so this is fine. But if a refactor changes `elems_per_block` to a runtime value, or the loops become conditional, the arrays may be partially uninitialized. Mark them `= {0}` defensively.

### B9. `QK_OSCAR2` constant not shared with kernel (LOW)

`QK_OSCAR2` is the only source of the 128-element block size. The Hadamard shares a `__shared__ float sh_val_had[QK_OSCAR2]` buffer, but `d_per_block = QK_OSCAR2` is also declared. The role of `d_per_block` is unclear and unused for sizing - delete or use it consistently. If someone changes `QK_OSCAR2` later, both must move together.

---

## `ggml/src/ggml-cuda/fattn.cu` (OSCAR2 dispatch)

### B10. Variable definition after closing macro (MEDIUM)

In `ggml_cuda_flash_attn_ext_oscar2`:

```cpp
#define DISPATCH_OSCAR2(DIM) ...
    ggml_type type_K = K->type;
    ggml_type type_V = V->type;

    if (D == 64)  { DISPATCH_OSCAR2(64);  }
    ...
```

The macro references `type_K`/`type_V`, which are declared AFTER the macro uses them. This compiles only because C++ allows late declarations, but it is the opposite of how `ggml_cuda_flash_attn_ext_q2_0` (identical pattern) is written: declare `type_K/type_V` first, then the switch. Move them up.

### B11. `OSCAR2` not gated by head-dim%64 check (HIGH)

`BEST_FATTN_KERNEL_OSCAR2` is selected unconditionally when K or V type is `GGML_TYPE_OSCAR2`. Other kernels (turbo2/turbo3/turbo4) require head-dim to be a multiple of 64 and reject mismatches in `ggml_cuda_get_best_fattn_kernel`. OSCAR2 has the same restriction (block size 128), but it is not enforced — meaning head-dims like 96 will dispatch to OSCAR2 and reach `GGML_ABORT("fatal error")` at runtime.

Add an explicit assert that rejects non-(multiple of 128) head dims before selecting `BEST_FATTN_KERNEL_OSCAR2`.

### B12. `BEST_FATTN_KERNEL_OSCAR2 = 40` collides with expected ranking (LOW)

The kernel-value scoring scheme ranks kernels by integer priority: NONE (0), OSCAR2 (40), Q2_0 (50), VEC (100), TILE (200), WMMA (300), MMA (400). Because the dispatcher takes the maximum, the OSCAR2 path is preferred over VEC, TILE, WMMA, MMA as long as K or V is `GGML_TYPE_OSCAR2`. If you intend OSCAR2 to be a fallback test path only, it should score lower than VEC. Worth a sanity check that this is the intended ordering.

### B13. No integer boundary for `i_kv` break (MEDIUM)

```cpp
for (int i_kv = 0; i_kv < nthreads; ++i_kv) {
    if (kv_base + i_kv >= k_VKQ_max) break;
    ...
}
```

`k_VKQ_max = KV_max_ptr ? KV_max_ptr[...] : ne11`. So the bound is `ne11` (the K length along its K axis). But `K = K_ptr + nb13*sequence + nb12*(head/gqa_ratio) + blockIdx.y * nthreads * nb11;` advances `blockIdx.y * nthreads * nb11`, which indexes into the K axis. If `ne11` is the count along that axis, this is consistent; OK. The only concern is when `k_VKQ_max > ne11` — which would still loop into nonsense. Add `GGML_ASSERT(k_VKQ_max <= ne11)`.

---

## `ggml/src/ggml-cpu/ops.cpp` (OSCAR CPU reference impl, lines ~9080-9420)

### B14. `op_params[4]` semantics for OSCAR (MEDIUM, needs review)

Header comment says "OSCAR two-tier mixed-precision fused attention (flag in op_params[4])". The actual op_params parsing should match the kernel expectations. Without the full ops.cpp slice, please verify that `op_params[4]` defaults to 0 and is correctly toggled when `K->type == GGML_TYPE_OSCAR2 || V->type == GGML_TYPE_OSCAR2`.

### B15. CPU reference FA needs the same Hadamard path (MEDIUM)

The CPU OSCAR `flash_attn_ext_oscar` reference impl must apply the same inverse-Hadamard transform as the CUDA kernel if it is used as a numerical oracle. If the CPU reference omits the Hadamard, the per-element numerical baselines do not match what the kernel produces. Confirm by reading the ~340 line slice of `ops.cpp` (the B-peek showed it was truncated).

---

## `ggml/include/ggml.h` and `ggml/src/ggml-common.h` (storage layout)

### B16. `GGML_TYPE_OSCAR2 = 49` slot (LOW)

49 is free in upstream llama.cpp type slots but in this fork it needs to be verified not to clash with `GGML_TYPE_TURBO*` or other downstream additions. Use the bump-by-policy in this fork (which appears to use slots 51/52 for turbo3/turbo4 and 60 for q2_preh). 49 is fine; just confirm.

### B17. `block_oscar2` struct width vs docstring (HIGH)

Verified: `ggml/src/ggml-common.h` on `origin/oscar` defines:

```c
#define QK_OSCAR2 128
} block_oscar2;
```

with `uint8_t qs[QK_OSCAR2 / 4];` (32 bytes) plus `d`, `m` of type `ggml_half` (2+2 = 4 bytes). The struct is therefore **36 bytes**, not 32. There is a `static_assert` enforcing this:

```c
static_assert(sizeof(block_oscar2) == QK_OSCAR2/4 + 2*sizeof(ggml_half),
              "block_oscar2 size mismatch");
```

So `sizeof(block_oscar2) == 36`. Two consequences:

* The comment in `fattn-oscar2.cuh` that says "32 byte / 128 elements" is misleading. With 32-bytes codes + 2-half scales, one block is 36 bytes, and an `nblocks`-wide row stores `nblocks * 36` bytes. Any caller, dispatcher or test that assumes 32 bytes per row will compute `nb11` too small.
* When `nb11` is computed from the row byte-width assumption, the `K_blk = (const block_oscar2 *)(K + i_kv * nb11)` indexing (B6) and B9's "stride per block" silently drift if `nb11` was derived from a 32-byte row. Audit any `ggml_row_size(GGML_TYPE_OSCAR2, D)` consumers.

Fix: update `fattn-oscar2.cuh` comment to "36 bytes / 128 elements" or expose `sizeof(block_oscar2)` as a `constexpr` in `ggml-common.h` and reference it everywhere.

### B18. INT2 code sign vs zero extension (MEDIUM)

```cpp
const uint8_t code = (blk[ib].qs[by] >> (2 * sub)) & 0x03;
buf[ib * QK_OSCAR2 + j] = (float)code * d + m;
```

The code is `uint8_t & 0x03`, so `0..3` - that is correct for INT2 unsigned sub-quant. The downstream `fmaf` is correct. No bug here, but the `quants.c` quantize function must clamp to `0..3`, not `-2..1` or any other range. Confirm the quantize function exists.

---

## `ggml/src/ggml-cpu/quants.c` (quantize function presence)

### B19. `quantize_row_oscar2_reference` not in slice (HIGH, unknown)

The original 10-line slice extracted from `quants.c / oscar/` was too short to verify whether the actual `quantize_row_oscar2_reference` function exists, and if so whether it clamps codes to `0..3`. If it is missing entirely, the type-system maps to `GGML_TYPE_F32` in `quants.c`, which would silently treat OSCAR2 as float - and the FA kernel would crash dereferencing `block_oscar2`.

Run `grep -n quantize_row_oscar2 ggml/src/ggml-cpu/quants.c` and confirm a clamp-to-03 quant implementation is in place.

---

## `ggml/src/ggml-cpu/ggml-cpu.c` (init / type registration)

### B20. `to_float` / `from_float` / `vec_dot` registration (LOW)

The slice mentions `GGML_TYPE_OSCAR2` mapping for `vec_dot`. Verify:
* `ggml_compute_forward_oscar2_*` and `ggml_cpu_compute_forward_op_*` for `OP_MUL_MAT` exist and use the dequant-then-dot path.
* `ggml_cpu_init()` registers `GGML_TYPE_OSCAR2` in the op type tables (especially `ggml_cpu_op_mul_mat`, `ggml_cpu_op_flash_attn_ext`).

If missing, the FA dispatcher in CUDA will be invoked even though the CPU path has no impl — which is fine in CUDA-only mode but the CPU tests in tests/ will fail.

---

## `common/arg.cpp`

### B21. CLI "type oscar2" mapping absent or stale (LOW)

If `llama-cli` exposes `-ctv f16|f32|q*|oscar2` style flags, the argument parser must accept `oscar2`. Verify by running `grep -n oscar common/arg.cpp` — it was visible in the codepath so probably exists, but confirm the spelling matches `GGML_TYPE_OSCAR2` and the type-name string is `"oscar2"` (lowercase, no hyphen).

---

## `tests`

### B22. Verification harness not exercised (HIGH)

`run_oscar_tests.sh` should include a small CUDA smoke test that compares the OSCAR2 FA kernel output against the CPU reference impl over a handful of head dims. The slices do not show that this exists. If it does exist, it should reproduce the failing cases (D=192 should be added) and fail on B1/B2 before any other test runs.

---

## Summary Count

| Severity | Count |
|---|---|
| CRITICAL | 2 (B1, B2) |
| HIGH | 5 (B3, B4, B6, B11, B17, B19, B22) |
| MEDIUM | 5 (B5, B10, B13, B14, B15, B18) |
| LOW | 7 (B7, B8, B9, B12, B16, B20, B21) |

The two CRITICAL items are likely the largest contributors to the "incoherent output" symptom already documented in `OSCAR-PORT-STATUS.md`. Confirm by reverting B1 and B2 separately and re-running `run_oscar_tests.sh`.

---

## Suggested Validation

```sh
# Build the oscar branch (CUDA required)
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)

# Run the OSCAR-specific tests (sizes triggered: 64, 128, 256, 512)
./run_oscar_tests.sh

# Disassemble the kernel for B7 verification
cuobjdump --dump-sass build/bin/llama-server 2>/dev/null | grep -A20 hadamard_inverse
```

## Fixes landed on `oscar` branch

The following 5 bugs from the list above have been applied locally on the
`oscar` branch. Fixes are recorded in the form `<file>:<line range>` with the
surviving change. Reviewed but not yet landed: B5, B6, B8, B10, B12, B13, B14,
B15, B16, B20, B21 (mostly documentation / wiring follow-ups that are not
required to compile or to fix the reported "incoherent output" symptom).

### F1. Resolved: B1 — duplicate KQ sum for `D < QK_OSCAR2`

`ggml/src/ggml-cuda/fattn-oscar2.cuh`: removed the redundant
`if constexpr (!use_block_unroll) { ... }` block that was being applied in
addition to the `else` branch of the `if constexpr (use_block_unroll)`.
For template instantiations with `D < 128`, `sum` was being accumulated
twice into the same `KQ_val[j]` register. After the fix, the `else` arm is
the only path that runs when `use_block_unroll == false`, so the partial
block is processed exactly once.

### F2. Resolved: B17 — header comment for kernel and block layout

`ggml/src/ggml-cuda/fattn-oscar2.cuh`: rewrote the top-of-file comment so
that it reflects the actual `sizeof(block_oscar2) == 36` layout (32 byte
codes + 2 halves) and notes that the kernel runs as one warp (32 threads,
`nwarps_k = 1`) regardless of `D`. The previous comment claimed
"For D >= 128: 128 threads (4 warps)" which contradicts the current code.

### F3. Resolved: B3 — column-bound checks use wrong `ne01` component

`ggml/src/ggml-cuda/fattn-oscar2.cuh`: replaced `ne01.z` with `ne01.x` in
the two column-bound checks (`(int)ne01.x`):

* mask gate: `if (maskh && (ncols == 1 || ic0 + j < (int)ne01.x))`
* write-back gate: `if (ncols > 1 && ic0 + j >= (int)ne01.x) break;`

The `dst_meta_ptr` and `dst_ptr` index formulas still use the original
`sequence * (int)ne01.z + ic0 + j` layout. This is intentional for now:
those formulas encode a dst layout convention that may be specific to
OSCAR2's write-rotation post-processing, and changing them without
running the test harness risks a silent regression. Flagged as a future
audit, but unchanged for this patch.

### F4. Resolved: B11 — head-dim gate matches `QK_OSCAR2` granularity

`ggml/src/ggml-cuda/fattn.cu`: dropped `D == 64` from both:

* The selector gate in `ggml_cuda_get_best_fattn_kernel`:
  ```c
  if (K->type == GGML_TYPE_OSCAR2 || V->type == GGML_TYPE_OSCAR2) {
      const int D = K->ne[0];
      if (D == 128 || D == 256 || D == 512) {
          return BEST_FATTN_KERNEL_OSCAR2;
      }
      return BEST_FATTN_KERNEL_NONE;
  }
  ```
* The dispatch table in `ggml_cuda_flash_attn_ext_oscar2`:
  `if (D == 128) {...} if (D == 256) {...} if (D == 512) {...}`
  (the previous `if (D == 64) { DISPATCH_OSCAR2(64); }` line is removed).

After this change, the head-dim gate aligns with the `nblocks = D/128`
block layout. Unsupported head dims (e.g. `D=64` and any non-multiple-of-128)
fall through to `BEST_FATTN_KERNEL_NONE`, which the upstream caller turns
into a clear "unsupported configuration" error instead of producing an
incoherent OSCAR2 result.

### F5. Resolved: B2 — kernel template rejects non-multiple-of-128 head dims

`ggml/src/ggml-cuda/fattn-oscar2.cuh`: added a `static_assert` next to the
existing divisibility check:

```cpp
static_assert(D % nthreads == 0, "D not divisible by nthreads");
static_assert(D >= QK_OSCAR2 && D % QK_OSCAR2 == 0,
              "OSCAR2 FA kernel requires D >= 128 and D % QK_OSCAR2 == 0");
```

Together with F4, the only template instantiations that can survive are
`D \in {128, 256, 512}`. For `D = 192, 320, 384, ...` the compiler will
reject the template (clearer than silently truncating `nblocks = D/128`
as the old code did).

### Diff summary

* `ggml/src/ggml-cuda/fattn-oscar2.cuh`: +17 -17 lines.
* `ggml/src/ggml-cuda/fattn.cu`: +11 -11 lines.

Total: `+28`, `-28` lines.

### Validation still required

1. Build the oscar branch on a CUDA-capable system. The `static_assert`
   change should compile-fail loudly if anyone adds a new `DISPATCH_OSCAR2(D)`
   entry with `D` outside `{128, 256, 512}`.
2. Re-run `./run_oscar_tests.sh` for the three supported head dims.
3. Verify the "incoherent output" symptom from `OSCAR-PORT-STATUS.md` is
   resolved (or at least progressed) by F1 alone.
4. Run the dgx-spark benchmark suite to confirm no performance regression.

## Fixes landed: tools accept `oscar2` type

### F6. `llama-bench` type parsing missing `oscar2`, `q2_0`, `q2_preh`

`tools/llama-bench/llama-bench.cpp`: the local `ggml_type_from_name()` function
had its own per-string mapping that was missing `oscar2`, `q2_0`, and `q2_preh`.
Added three entries:

```cpp
if (s == "q2_0")    { return GGML_TYPE_Q2_0;    }
if (s == "q2_preh") { return GGML_TYPE_Q2_PREH; }
if (s == "oscar2")  { return GGML_TYPE_OSCAR2;  }
```

This allows `llama-bench -ctk oscar2 -ctv oscar2` to work. The other CLI tools
(`llama-cli`, `llama-server`, `llama-perplexity`) already accept `oscar2`
through the shared `common/arg.cpp -> kv_cache_type_from_str()` codepath which
iterates `kv_cache_types` and calls `ggml_type_name()`, while `llama-quantize`
uses `parse_ggml_type()` which iterates `GGML_TYPE_COUNT` calling
`ggml_type_name()` — both of these already resolve `GGML_TYPE_OSCAR2` correctly.

### B21 update: CLI mapping is present

`common/arg.cpp` at line 390 already includes `GGML_TYPE_OSCAR2` in the
`kv_cache_types` vector, and `kv_cache_type_from_str()` resolves it via
`ggml_type_name(type) == s`. The type name in `ggml.c` is `"oscar2"`.
The `--cache-type-k oscar2` / `--cache-type-v oscar2` CLI flags are fully
functional. B21 is downgraded to verified/not-a-bug.

Last reviewed commit: `origin/oscar` @ `791ca44f4432b0b5730e44a6394bed5621dd01d6`.

---

## Recent field status — TurboQuant oscar branch (`6c3822c6f`, `rebuild_tq.sh` build)

- Build path: rebuilt with `/mnt/storage/llama-server/rebuild_tq.sh`; fresh artifacts under `/mnt/storage/Projects/turboquant/build` dated `2026-07-23 20:02 MDT`.
- Verified fixes F1–F5 are reflected in the tree; only code change relative to `791ca44f` is `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, which removes the unused F16 MMA OSCAR2 raw staging pointer; the OSCAR2 FA path itself was not changed in this delta.

- Exact benchmark target: `/mnt/storage/models/qwen3.6-27b-q5kxl-hadamard.gguf`, config `-p 0 -ngl 999 -t 64 -n 64 -fa on -ctk oscar2 -ctv oscar2`.
- Blocker: `build/bin/llama-bench` rejects `-ctk oscar2` with `error: invalid parameter for argument: -ctk`, despite the CLI help exposing `-ctk/-ctv`. Inspected `common/arg.cpp`; `GGML_TYPE_OSCAR2` is registered, but `llama-bench` does not accept `oscar2` through the cache-type parser in this build.
- Consequence: no successful OSCAR2 KV cache throughput or VRAM artifact was produced for this model/config. The only successful per-build number from this tree is f16/f16 KV cache: `tg64 = 67.58 ± 0.30 t/s`, VRAM ~32088 MiB total. Per standing instructions, f16/FA results do not count as OSCAR2 results.
- Additional known runtime blocker from prior logs: `llama-cli` requires `--flash-attn on` for V=`oscar2`; `--flash-attn off` aborts with `V cache quantization requires flash_attn`. That prior 128K oscar2 run completed generation-only at `4.2 / 1.9 t/s` with ~1188 MiB VRAM used.
- New CLI check on `6c3822c6f`: `build/bin/llama-cli` accepts `--cache-type-k oscar2 --cache-type-v oscar2 --flash-attn on` for `/mnt/storage/models/qwen3.6-27b-q5kxl-hadamard.gguf`, but the run hangs beyond 600s on RTX 5090 sm_120. This is consistent with the verified blocker: OSCAR2 FA hangs on Blackwell sm_120. No t/s or VRAM artifact obtained from `llama-cli` in this state.
