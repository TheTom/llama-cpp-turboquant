// OSCAR2 dedicated flash attention kernel
// Per-128-vector min-max asymmetric INT2 dequant: val = code * d + m
// No Hadamard transform, no Lloyd-Max centroids.
// For D >= 128: 128 threads (4 warps), cooperative dequant
// For D < 128:  32 threads (1 warp), serial dequant (legacy)

#include "common.cuh"
#include "fattn-common.cuh"

// ---------------------------------------------------------------------------
// Single-threaded helpers (fallback for D < 128)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void dequant_row_oscar2(
    const block_oscar2 * blk, int D, float * buf) {
    const int nb = D / QK_OSCAR2;
    for (int ib = 0; ib < nb; ++ib) {
        const float d = __half2float(blk[ib].d);
        const float m = __half2float(blk[ib].m);
        for (int j = 0; j < QK_OSCAR2; ++j) {
            const int by  = j / 4;
            const int sub = j % 4;
            const uint8_t code = (blk[ib].qs[by] >> (2 * sub)) & 0x03;
            buf[ib * QK_OSCAR2 + j] = (float)code * d + m;
        }
    }
}

// ---------------------------------------------------------------------------
// 128-thread cooperative dequant
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void dequant_row_oscar2_parallel(
    const block_oscar2 * blk, int D, float * buf) {
    constexpr int nthreads = 128;
    const int nb = D / QK_OSCAR2;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    for (int ib = 0; ib < nb; ++ib) {
        const float d = __half2float(blk[ib].d);
        const float m = __half2float(blk[ib].m);
        for (int off = 0; off < QK_OSCAR2; off += nthreads) {
            const int elem = off + tid;
            if (elem >= QK_OSCAR2) break;
            const int by  = elem / 4;
            const int sub = elem % 4;
            const uint8_t code = (blk[ib].qs[by] >> (2 * sub)) & 0x03;
            buf[ib * QK_OSCAR2 + elem] = (float)code * d + m;
        }
        __syncthreads();
    }
}
// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

template <int D, int ncols, bool use_logit_softcap, ggml_type type_K, ggml_type type_V>
static __global__ void flash_attn_ext_oscar2(
        const char  * Q_ptr,
        const char  * K_ptr,
        const char  * V_ptr,
        const char  * mask_ptr,
        const char  * sinks_ptr,
        const int   * KV_max_ptr,
        float       * dst_ptr,
        float2      * dst_meta_ptr,
        const float   scale,
        const float   max_bias,
        const float   m0,
        const float   m1,
        const uint32_t n_head_log2,
        const float   logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {

#ifdef FLASH_ATTN_AVAILABLE
    // OPTIMIZED: use 1 warp (32 threads) instead of 4 warps (128 threads).
    // Each thread handles D/32 elements. No cross-warp __syncthreads needed.
    constexpr int nwarps_k = 1;
    constexpr int nthreads = nwarps_k * WARP_SIZE;
    static_assert(D % nthreads == 0, "D not divisible by nthreads");
    constexpr int nelems = D / nthreads;

    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    // Shared memory for cross-warp reduction only (no K/V s_buf)
    __shared__ float s_red[32];

    const int ic0 = blockIdx.x * ncols;
    const int sequence = blockIdx.z / ne02;
    const int head     = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;

    const char * Q = Q_ptr + nb03*sequence + nb02*head + nb01*ic0;
    const char * K = K_ptr + nb13*sequence + nb12*(head / gqa_ratio) + blockIdx.y * nthreads * nb11;
    const char * V = V_ptr + nb23*sequence + nb22*(head / gqa_ratio) + blockIdx.y * nthreads * nb21;
    const half * maskh = mask_ptr ? (const half *)(mask_ptr + nb33*(sequence % ne33) + nb31*ic0 + blockIdx.y * nthreads) : nullptr;
    const float * sinks = sinks_ptr ? (const float *)(sinks_ptr + (sequence*ne02 + head) * 2) : nullptr;
    GGML_UNUSED(sinks);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Block-unrolled K/V dequant: each thread handles D/32 elements.
    // For D >= 128, elements span nblocks = D/128 oscar2 blocks with 4 elements/block.
    // For D < 128, fall back to the original element-by-element loop.
    constexpr bool use_block_unroll = (D >= 128);
    constexpr int nblocks = use_block_unroll ? D / QK_OSCAR2 : 1;
    constexpr int elems_per_block = use_block_unroll ? 4 : nelems;
    constexpr int d_per_block  = QK_OSCAR2;

    // Pre-compute byte offset and bit-shift within qs[] (per-block, same for all blocks)
    int by_blk[elems_per_block];
    int shift_blk[elems_per_block];
    #pragma unroll
    for (int e = 0; e < elems_per_block; ++e) {
        const int off = tid + e * nthreads;
        by_blk[e]    = off / 4;
        shift_blk[e] = (off & 3) * 2;
    }

    // Load Q into registers
    float Q_reg[ncols][nelems];
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float * Q_j = (const float *) (Q + j*nb01);
        #pragma unroll
        for (int e = 0; e < nelems; ++e) {
            Q_reg[j][e] = Q_j[tid + e * nthreads] * scale;
        }
    }

    float KQ_max[ncols], KQ_sum[ncols];
    float VKQ[ncols][nelems];
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2;
        KQ_sum[j] = 0.0f;
        #pragma unroll
        for (int e = 0; e < nelems; ++e) VKQ[j][e] = 0.0f;
    }

    const int k_VKQ_max = KV_max_ptr ? KV_max_ptr[sequence*gridDim.x + blockIdx.x] : ne11;

    for (int kv_base = blockIdx.y * nthreads; kv_base < k_VKQ_max;
         kv_base += gridDim.y * nthreads,
         K += gridDim.y * nthreads * nb11,
         V += gridDim.y * nthreads * nb21,
         maskh += gridDim.y * nthreads) {

        for (int i_kv = 0; i_kv < nthreads; ++i_kv) {
            if (kv_base + i_kv >= k_VKQ_max) break;

            const block_oscar2 * K_blk = (const block_oscar2 *)(K + i_kv * nb11);
            const block_oscar2 * V_blk = (const block_oscar2 *)(V + i_kv * nb21);

            // ---- K dequant + dot product ----
            float KQ_val[ncols] = {0.0f};
            #pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = 0.0f;
                #pragma unroll
                for (int b = 0; b < nblocks; ++b) {
                    const float d_k = __half2float(K_blk[b].d);
                    const float m_k = __half2float(K_blk[b].m);
                    #pragma unroll
                    for (int e = 0; e < elems_per_block; ++e) {
                        const uint8_t code = (K_blk[b].qs[by_blk[e]] >> shift_blk[e]) & 0x03;
                        sum += (fmaf((float)code, d_k, m_k)) * Q_reg[j][b * elems_per_block + e];
                    }
                }
                // Handle D < 128 (partial block)
                if constexpr (!use_block_unroll) {
                    for (int e = 0; e < nelems; ++e) {
                        const uint8_t code = (K_blk[0].qs[by_blk[e]] >> shift_blk[e]) & 0x03;
                        sum += (fmaf((float)code, __half2float(K_blk[0].d), __half2float(K_blk[0].m))) * Q_reg[j][e];
                    }
                }
                KQ_val[j] = sum;
            }

            // ---- Score and online softmax ----
            #pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float full_kq;
                if constexpr (nwarps_k > 1) {
                    float warp_sum = warp_reduce_sum(KQ_val[j]);
                    if (threadIdx.x == 0) { s_red[threadIdx.y] = warp_sum; }
                    __syncthreads();
                    if (threadIdx.y == 0) {
                        float cross = threadIdx.x < nwarps_k ? s_red[threadIdx.x] : 0.0f;
                        cross = warp_reduce_sum(cross);
                        if (threadIdx.x == 0) { s_red[0] = cross; }
                    }
                    __syncthreads();
                    full_kq = s_red[0];
                } else {
                    full_kq = warp_reduce_sum(KQ_val[j]);
                }

                if (use_logit_softcap) full_kq = logit_softcap * tanhf(full_kq);
                if (maskh && (ncols == 1 || ic0 + j < (int)ne01.z))
                    full_kq += slope * __half2float(maskh[j*ne11 + i_kv]);

                const float rn = fmaxf(KQ_max[j], full_kq + FATTN_KQ_MAX_OFFSET);
                const float ks = expf(KQ_max[j] - rn);
                KQ_max[j] = rn;
                const float ke = expf(full_kq - KQ_max[j]);
                KQ_sum[j] = KQ_sum[j] * ks + ke;

                #pragma unroll
                for (int e = 0; e < nelems; ++e) VKQ[j][e] *= ks;

                // ---- V dequant + VKQ accumulation ----
                if constexpr (use_block_unroll) {
                    #pragma unroll
                    for (int b = 0; b < nblocks; ++b) {
                        const float d_v = __half2float(V_blk[b].d);
                        const float m_v = __half2float(V_blk[b].m);
                        #pragma unroll
                        for (int e = 0; e < elems_per_block; ++e) {
                            const uint8_t code = (V_blk[b].qs[by_blk[e]] >> shift_blk[e]) & 0x03;
                            VKQ[j][b * elems_per_block + e] += ke * fmaf((float)code, d_v, m_v);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < nelems; ++e) {
                        const uint8_t code = (V_blk[0].qs[by_blk[e]] >> shift_blk[e]) & 0x03;
                        VKQ[j][e] += ke * fmaf((float)code, __half2float(V_blk[0].d), __half2float(V_blk[0].m));
                    }
                }
            }

        }
    }
    // Write results
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (ncols > 1 && ic0 + j >= (int)ne01.z) break;
        const float iks = gridDim.y == 1 ? 1.0f / KQ_sum[j] : 1.0f;
        if (gridDim.y != 1 && tid == 0) {
            int mi = ((sequence * (int)ne01.z + ic0 + j) * ne02 + head) * gridDim.y + blockIdx.y;
            dst_meta_ptr[mi] = make_float2(KQ_max[j], KQ_sum[j]);
        }
        #pragma unroll
        for (int e = 0; e < nelems; ++e) {
            int di = tid + e * nthreads;
            float val = VKQ[j][e] * iks;
            if (gridDim.y == 1)
                dst_ptr[(((sequence * (int)ne01.z + ic0 + j) * ne02 + head)) * D + di] = val;
            else
                dst_ptr[(((sequence * (int)ne01.z + ic0 + j) * ne02 + head) * gridDim.y + blockIdx.y) * D + di] = val;
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
        ne31, ne32, ne33, nb31, nb32, nb33);
#endif
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_oscar2_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // Shared memory: just s_red[32] = 128 bytes (no K/V s_buf)
    constexpr size_t nbytes = 0; // no dynamic shared memory needed
    // OPTIMIZED: always 1 warp (was: D >= QK_OSCAR2 ? 4 : 1)
    constexpr int nwarps = 1;

    const int nbatch_fa = dst->src[1]->ne[1];

    auto launch = [&](int ncols, bool lsc) {
        if (lsc) {
            if (ncols == 1) {
                fattn_kernel_t k = flash_attn_ext_oscar2<D, 1, true,  type_K, type_V>;
                launch_fattn<D, 1, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false);
            } else {
                fattn_kernel_t k = flash_attn_ext_oscar2<D, 2, true,  type_K, type_V>;
                launch_fattn<D, 2, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false);
            }
        } else {
            if (ncols == 1) {
                fattn_kernel_t k = flash_attn_ext_oscar2<D, 1, false, type_K, type_V>;
                launch_fattn<D, 1, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false);
            } else {
                fattn_kernel_t k = flash_attn_ext_oscar2<D, 2, false, type_K, type_V>;
                launch_fattn<D, 2, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false);
            }
        }
    };

    launch(Q->ne[1] > 1 ? 2 : 1, logit_softcap != 0.0f);
}
