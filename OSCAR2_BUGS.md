# OSCAR2 KV-cache Bug Report

Branch: `origin/oscar` (HEAD `791ca44f4432b0b5730e44a6394bed5621dd01d6`).

This list supplements the four known issues already in `OSCAR-PORT-STATUS.md`:

1. ~~Dedicated OSCAR2 FA kernel produces incoherent output.~~ PARTIALLY FIXED — B1/B2/B3-bound/B5/B8/B13 resolved; B4 (mask stride) and B6 (nb11 assert) remain open.
2. ~~VEC path broken for quantized KV at `D > 256`.~~ STILL BROKEN — fundamental domain mismatch: set_rows stores Hadamard-domain values, VEC dequant reads as natural-domain. Not fixable without adding inverse Hadamard to VEC path. Recommend explicit VEC-path disable for oscar2.
3. ~~Rotation matrices for Gemma-4-12B fall back to identity.~~ FIXED (F17)
4. ~~HP sink buffer not implemented for OSCAR2.~~ NOT ADDRESSED (separate feature, not a bug — planned but not blocking correctness at short context)

The bugs below were identified by reading the OSCAR2-specific slices of `ggml/src/ggml-cuda/fattn-oscar2.cuh`, `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cpu/ops.cpp` (CPU OSCAR reference attention), `ggml/src/ggml-cpu/quants.c`, `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/include/ggml.h`, and `ggml/src/ggml-common.h` from the `oscar` branch. They are grouped by file and given a severity tag.

---

## `ggml/src/ggml-cuda/fattn-oscar2.cuh`

### B1. KQ dot product is computed twice when `D < 128` (CRITICAL) — FIXED

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

Fix applied: the duplicate lower branch was deleted. Current kernel has single KQ loop.

### B2. Non-multiple-of-128 head dims silently drop elements (CRITICAL) — FIXED

```cpp
constexpr bool use_block_unroll = (D >= 128);
constexpr int nblocks = use_block_unroll ? D / QK_OSCAR2 : 1;
...
for (int b = 0; b < nblocks; ++b) {
    ...
    // process QK_OSCAR2 = 128 elements in this block
}
```

`nblocks = D / QK_OSCAR2` is integer truncation. For `D = 192` or 320 or 384, only `floor(D/128)` blocks are processed. Elements `nblocks * 128 .. D-1` are silently discarded.

Fix applied: `static_assert(D % QK_OSCAR2 == 0)` and `D >= QK_OSCAR2` added in fattn-oscar2.cuh (line 96-97). The dispatcher in fattn.cu also gates D in {128, 256, 512}. Non-multiple-of-128 D now fails at compile time.

### B3. Column-bound check uses wrong dimension (HIGH) — PARTIALLY FIXED

```cpp
if (ncols > 1 && ic0 + j >= (int)ne01.z) break;
...
dst_ptr[(((sequence * (int)ne01.z + ic0 + j) * ne02 + head)) * D + di] = val;
```

`ne01` is declared as `const uint3`. Its components are `(ne[1], ne[2], ne[3]) = (ncols, n_head, batch)`. So `ne01.z` is actually `ne[3]` (batch), not the column dimension. The column bound should be against `(uint32_t)ne01.x` (which is `ncols`).

Fix applied (partial): the column-bound check now uses `ne01.x` (line 275). The `dst_ptr` index still uses `ne01.z` — this is fragile for batch > 1 but works for the single-sequence case used in testing.

### B4. Mask indexing ignores stride parameters (HIGH) — NOT FIXED

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

Fix: compute `maskh + (kv_base + i_kv) * nb31 + j * nb_other` using the supplied strides.

Status: STILL OPEN. Only affects models with non-standard mask layouts (e.g., ALiBi with GQA).

### B5. `hadamard_inverse_128_32w` bounds check on read is harmless but write condition is wrong (MEDIUM) — FIXED

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
* Single-warp kernel uses `__syncthreads()` instead of `__syncwarp()` (fragile if nwarps_k changes).
* Boundary threads at `tid > 31-h` get zero padding, not mirrored writes.

Fix applied: rewritten to a clean loop form with `idx < 128 && !(idx & h)` guard, `__syncwarp()`. Current code is correct.

### B6. `K_blk` / `V_blk` are advanced by stride `nb11/nb21` but read per block (HIGH) — NOT FIXED (comment only)

```cpp
const block_oscar2 * K_blk = (const block_oscar2 *)(K + i_kv * nb11);
...
for (int b = 0; b < nblocks; ++b) {
    const float d_k = __half2float(K_blk[b].d);
    ...
    sum += ... Q_reg[j][b * elems_per_block + e];
}
```

The kernel assumes `nb11` (logical K row stride) is `sizeof(block_oscar2) * (D / QK_OSCAR2)`. If a model uses any padding or weight permute, `nb11` will be larger, and then `K_blk[b]` will step out of the actual row.

Partial fix: INVARIANT comment added at line 196, but no `GGML_ASSERT`. Add `GGML_ASSERT(nb11 == nblocks * sizeof(block_oscar2))` to catch this in development.

### B7. V dequant mean-centering assumes Hadamard correctness (LOW) — NOT A BUG

The V branch computes `mean_v = m_v + 1.5f * d_v` (mean of 4 codes uniform in `{0,1,2,3}`) and applies the Hadamard before un-centering. Since B5 is fixed (Hadamard is correct), this composition is correct. Current kernel keeps the mean separate and adds after Hadamard inverse.

### B8. Per-block `by_blk[]` / `shift_blk[]` use uninitialized array (LOW) — FIXED

```cpp
int by_blk[elems_per_block];
int shift_blk[elems_per_block];
```

Fix applied: zero-initialized with `= {}` (lines 129-130).

### B9. `QK_OSCAR2` constant not shared with kernel (LOW) — DEFERRED

`QK_OSCAR2` is the only source of the 128-element block size. The Hadamard shares a `__shared__ float sh_val_had[QK_OSCAR2]` buffer, but `d_per_block = QK_OSCAR2` is also declared. The role of `d_per_block` is unclear and unused for sizing — delete or use it consistently. Low priority: no active bug, just a maintenance risk if QK_OSCAR2 changes.

---

## `ggml/src/ggml-cuda/fattn.cu` (OSCAR2 dispatch)

### B10. Variable definition after closing macro (MEDIUM) — FIXED

In `ggml_cuda_flash_attn_ext_oscar2`:

```cpp
#define DISPATCH_OSCAR2(DIM) ...
    ggml_type type_K = K->type;
    ggml_type type_V = V->type;
```

The macro references `type_K`/`type_V`, which were declared AFTER the macro uses them. In current code, `type_K` and `type_V` are declared before the `DISPATCH_OSCAR2` macro definition (lines 575-576 before 578-599). FIXED.

### B11. `OSCAR2` not gated by head-dim%64 check (HIGH) — FIXED

`BEST_FATTN_KERNEL_OSCAR2` is selected only when D in {128, 256, 512} (line 708-709). `static_assert(D % QK_OSCAR2 == 0)` in kernel enforces at compile time. FIXED.

### B12. `BEST_FATTN_KERNEL_OSCAR2 = 40` collides with expected ranking (LOW) — NOT A BUG

The kernel-value scoring scheme ranks kernels by integer priority. OSCAR2 (40) is lower than VEC (100) but selected directly by type-gate (line 613-615), not by max-score competition. Intentional design for oscar2's Hadamard domain requirement.

### B13. No integer boundary for `i_kv` break (MEDIUM) — FIXED

```cpp
for (int i_kv = 0; i_kv < nthreads; ++i_kv) {
    if (kv_base + i_kv >= k_VKQ_max) break;
    ...
}
```

Fix applied: `k_VKQ_max` is now clamped via `min(k_VKQ_max_raw, ne11)` (line 184), preventing out-of-bounds K/V reads even if `KV_max_ptr` reports values exceeding `ne11`.

---

## `ggml/src/ggml-cpu/ops.cpp` (OSCAR CPU reference impl, lines ~9080-9420)

### B14. `op_params[4]` semantics for OSCAR (MEDIUM, needs review)

Header comment says "OSCAR two-tier mixed-precision fused attention (flag in op_params[4])". The actual op_params parsing should match the kernel expectations. Without the full ops.cpp slice, please verify that `op_params[4]` defaults to 0 and is correctly toggled when `K->type == GGML_TYPE_OSCAR2 || V->type == GGML_TYPE_OSCAR2`.

Status: NOT VERIFIED — requires reading the CPU FA reference implementation in ops.cpp.

### B15. CPU reference FA needs the same Hadamard path (MEDIUM)

The CPU OSCAR `flash_attn_ext_oscar` reference impl must apply the same inverse-Hadamard transform as the CUDA kernel if it is used as a numerical oracle. If the CPU reference omits the Hadamard, the per-element numerical baselines do not match what the kernel produces.

Status: NOT VERIFIED — requires reading the CPU FA reference implementation. Note that the CPU `dequantize_row_oscar2` does NOT apply inverse Hadamard (it produces natural-domain values from CPU-quantized blocks, since the CPU quant path also doesn't apply forward Hadamard). The CUDA set_rows path stores Hadamard-domain values — this is a fundamental divergence between CPU and GPU quant paths.

---

## `ggml/include/ggml.h` and `ggml/src/ggml-common.h` (storage layout)

### B16. `GGML_TYPE_OSCAR2 = 49` slot (LOW) — NOT A BUG

Type slot 49 sits cleanly between Q2_PREH (48) and COUNT (50). B16 is verified/not-a-bug.

### B17. `block_oscar2` struct width vs docstring (HIGH) — ADDRESSED

`block_oscar2` is 36 bytes (32B codes + 2B d + 2B m = 36B). The `static_assert` in ggml-common.h enforces this. The struct comment at line 292-297 correctly notes "36 bytes per 128 elements". The comment in fattn-oscar2.cuh line 6-8 also correctly says "32 byte codes + 2 halves (sigma, mean) = 4 bytes total sizeof(block_oscar2) == 36".

Status: Verified correct. All size consumers use `sizeof(block_oscar2)` or the static_assert guarantee.

### B18. INT2 code sign vs zero extension (MEDIUM) — NOT A BUG

```cpp
const uint8_t code = (blk[ib].qs[by] >> (2 * sub)) & 0x03;
buf[ib * QK_OSCAR2 + j] = (float)code * d + m;
```

The code is `uint8_t & 0x03`, so `0..3` — correct for INT2 unsigned sub-quant. The quantize function clamps to `[0, 3]` (verified in B19/B20). NOT A BUG.

---

## `ggml/src/ggml-cpu/quants.c` (quantize function presence)

### B19. `quantize_row_oscar2_reference` not in slice (HIGH) — VERIFIED NOT A BUG

`ggml/src/ggml-quants.c` lines 646-693: `quantize_row_oscar2_ref` exists with proper code clamping to `[0, 3]`. The `dequantize_row_oscar2` function reverses correctly with `code * d + m`. B19 is verified/not-a-bug.

---

## `ggml/src/ggml-cpu/ggml-cpu.c` (init / type registration)

### B20. `to_float` / `from_float` / `vec_dot` registration (LOW) — NOT A BUG

CPU type table in `ggml/src/ggml-cpu/ggml-cpu.c` lines 451-456 has complete registration. B20 is verified/not-a-bug.
