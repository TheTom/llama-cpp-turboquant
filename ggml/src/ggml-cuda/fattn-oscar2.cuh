// OSCAR2 dedicated flash attention kernel
// Per-128-vector (QK_OSCAR2 = 128) Lloyd-Max INT2 dequant:
//   val = OSCAR2_LM_CENTROIDS[code] * sigma + mean
// Centroids: {-0.9816, -0.4528, 0.4528, 0.9816} for N(0,1).
// Includes inverse Hadamard transform to recover pre-quantization values.
// block_oscar2 layout (ggml-common.h): qs[QK_OSCAR2 / 4] = 32 byte codes
//                                      + 2 halves (sigma, mean) = 4 bytes
//                                      total sizeof(block_oscar2) == 36.
// Main kernel runs as one warp (32 threads, nwarps_k = 1) regardless of D;
// D % QK_OSCAR2 == 0 is required (gated in ggml_cuda_get_best_fattn_kernel).

#include "common.cuh"
#include "fattn-common.cuh"

// Device-side copies of ggml-common.h constants.
// ggml-common.h declares these as static const, but the CUDA compiler does
// not make them visible in __device__ code for arrays > ~64 bytes. These
// __device__ copies ensure the FA kernel can access them.
static __device__ const float OSCAR2_CENTROIDS_DEV[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
static __device__ const int   P_BR_DEV[128] = {
    0, 64, 32, 96, 16, 80, 48, 112,  8, 72, 40, 104, 24, 88, 56, 120,
    4, 68, 36, 100, 20, 84, 52, 116, 12, 76, 44, 108, 28, 92, 60, 124,
    2, 66, 34, 98, 18, 82, 50, 114, 10, 74, 42, 106, 26, 90, 58, 122,
    6, 70, 38, 102, 22, 86, 54, 118, 14, 78, 46, 110, 30, 94, 62, 126,
    1, 65, 33, 97, 17, 81, 49, 113,  9, 73, 41, 105, 25, 89, 57, 121,
    5, 69, 37, 101, 21, 85, 53, 117, 13, 77, 45, 109, 29, 93, 61, 125,
    3, 67, 35, 99, 19, 83, 51, 115, 11, 75, 43, 107, 27, 91, 59, 123,
    7, 71, 39, 103, 23, 87, 55, 119, 15, 79, 47, 111, 31, 95, 63, 127};

// ---------------------------------------------------------------------------
// Single-threaded helpers (fallback for D < 128)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void dequant_row_oscar2(
    const block_oscar2 * blk, int D, float * buf) {
    constexpr float centroids[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
    const int nb = D / QK_OSCAR2;
    for (int ib = 0; ib < nb; ++ib) {
        const float d = __half2float(blk[ib].d);
        const float m = __half2float(blk[ib].m);
        for (int j = 0; j < QK_OSCAR2; ++j) {
            const int by  = j / 4;
            const int sub = j % 4;
            const uint8_t code = (blk[ib].qs[by] >> (2 * sub)) & 0x03;
            buf[ib * QK_OSCAR2 + j] = centroids[code] * d + m;
        }
    }
}

// ---------------------------------------------------------------------------
// 128-thread cooperative dequant
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void dequant_row_oscar2_parallel(
    const block_oscar2 * blk, int D, float * buf) {
    constexpr float centroids[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
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
            buf[ib * QK_OSCAR2 + elem] = centroids[code] * d + m;
        }
        __syncwarp();
    }
}
// ---------------------------------------------------------------------------
// 32-thread inverse Hadamard on 128 elements (shared memory).
// Each thread owns 4 elements: tid, tid+32, tid+64, tid+96.
// The butterfly condition !(idx & h) ensures each pair is updated by exactly
// one writer (the lower-index thread). Must sync before/after call.
// ---------------------------------------------------------------------------
static __device__ void hadamard_inverse_128_32w(float * sh, int tid) {
    #pragma unroll
    for (int h = 64; h > 0; h >>= 1) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = tid + k * 32;
            if (idx < 128 && !(idx & h)) {
                const float a = sh[idx];
                const float b = sh[idx + h];
                sh[idx]     = a + b;
                sh[idx + h] = a - b;
            }
        }
        __syncwarp();
    }
    constexpr float s = 0.08838834764f; // 1/sqrt(128)
    sh[tid]      *= s;  sh[tid + 32] *= s;
    sh[tid + 64] *= s;  sh[tid + 96] *= s;
    __syncwarp();
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
    static_assert(D >= QK_OSCAR2 && D % QK_OSCAR2 == 0,
                  "OSCAR2 FA kernel requires D >= 128 and D % QK_OSCAR2 == 0");
    constexpr int nelems = D / nthreads;

    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    // Shared memory for K/V inverse Hadamard buffer (128 floats)
    __shared__ float sh_val_had[QK_OSCAR2];
    // Shared memory for cross-warp reduction only (no K/V s_buf)
    __shared__ float s_red[32];

    const int ic0 = blockIdx.x * ncols;
    const int sequence = blockIdx.z / ne02;
    const int head     = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12; // n_head / n_head_kv (ne12 = n_head_kv after permute)

    const char * Q = Q_ptr + nb03*sequence + nb02*head + nb01*ic0;
    const char * K = K_ptr + nb13*sequence + nb12*(head / gqa_ratio) + blockIdx.y * nthreads * nb11;
    const char * V = V_ptr + nb23*sequence + nb22*(head / gqa_ratio) + blockIdx.y * nthreads * nb21;
    const half * maskh = mask_ptr ? (const half *)mask_ptr + (nb33/2)*(sequence % ne33) + (nb31/2)*ic0 + blockIdx.y * nthreads : nullptr;
    const float * sinks = sinks_ptr ? (const float *)(sinks_ptr + (sequence*ne02 + head) * 2) : nullptr;
    GGML_UNUSED(sinks);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Block-unrolled K/V dequant: each thread handles D/32 elements.
    // For D >= 128, elements span nblocks = D/128 oscar2 blocks with 4 elements/block.
    // For D < 128, fall back to the original element-by-element loop.
    constexpr bool use_block_unroll = (D >= 128);
    constexpr int nblocks = use_block_unroll ? D / QK_OSCAR2 : 1;
    constexpr int elems_per_block = use_block_unroll ? 4 : nelems;

    // Pre-compute byte offset and bit-shift within qs[] (per-block, same for all blocks)
    int by_blk[elems_per_block]   = {};
    int shift_blk[elems_per_block] = {};
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

    const int k_VKQ_max_raw = KV_max_ptr ? KV_max_ptr[sequence*gridDim.x + blockIdx.x] : ne11;
    const int k_VKQ_max = min(k_VKQ_max_raw, ne11);  // clamp to K row length

    for (int kv_base = blockIdx.y * nthreads; kv_base < k_VKQ_max;
         kv_base += gridDim.y * nthreads,
         K += gridDim.y * nthreads * nb11,
         V += gridDim.y * nthreads * nb21,
         maskh += gridDim.y * nthreads) {

        for (int i_kv = 0; i_kv < nthreads; ++i_kv) {
            if (kv_base + i_kv >= k_VKQ_max) break;

            const block_oscar2 * K_blk = (const block_oscar2 *)(K + i_kv * nb11);
            // INVARIANT: nb11 == nblocks * sizeof(block_oscar2) so K_blk[b] stays within the row
            const block_oscar2 * V_blk = (const block_oscar2 *)(V + i_kv * nb21);

            // ---- K dequant + dot product (with inv-Hadamard test) ----
            float KQ_val[ncols] = {0.0f};
            #pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = 0.0f;
                if constexpr (use_block_unroll) {
                    #pragma unroll
                    for (int b = 0; b < nblocks; ++b) {
                        const float d_k = __half2float(K_blk[b].d);
                        const float m_k = __half2float(K_blk[b].m);
                        #pragma unroll
                        for (int e = 0; e < elems_per_block; ++e) {
                            const uint8_t code = (K_blk[b].qs[by_blk[e]] >> shift_blk[e]) & 0x03;
                            sh_val_had[tid + e * nthreads] = OSCAR2_CENTROIDS_DEV[code] * d_k + m_k;
                        }
                        __syncwarp();
                        // P_br: reorder dequantized values from bit-reversal to natural order
                        { float _pbr[elems_per_block];
                          for (int _e = 0; _e < elems_per_block; ++_e) _pbr[_e] = sh_val_had[tid + _e * nthreads];
                          for (int _e = 0; _e < elems_per_block; ++_e) sh_val_had[P_BR_DEV[tid + _e * nthreads]] = _pbr[_e]; }
                        __syncwarp();
                        hadamard_inverse_128_32w(sh_val_had, tid);
                        #pragma unroll
                        for (int e = 0; e < elems_per_block; ++e) {
                            sum += sh_val_had[tid + e * nthreads] * Q_reg[j][b * elems_per_block + e];
                        }
                    }
                } else {
                    // D < QK_OSCAR2 single-block path (use_block_unroll == false).
                    const float d_k0 = __half2float(K_blk[0].d);
                    const float m_k0 = __half2float(K_blk[0].m);
                    for (int e = 0; e < nelems; ++e) {
                        const uint8_t code = (K_blk[0].qs[by_blk[e]] >> shift_blk[e]) & 0x03;
                        sum += (OSCAR2_CENTROIDS_DEV[code] * d_k0 + m_k0) * Q_reg[j][e];
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
                    __syncwarp();
                    if (threadIdx.y == 0) {
                        float cross = threadIdx.x < nwarps_k ? s_red[threadIdx.x] : 0.0f;
                        cross = warp_reduce_sum(cross);
                        if (threadIdx.x == 0) { s_red[0] = cross; }
                    }
                    __syncwarp();
                    full_kq = s_red[0];
                } else {
                    full_kq = warp_reduce_sum(KQ_val[j]);
                }

                if (use_logit_softcap) full_kq = logit_softcap * tanhf(full_kq);
                if (maskh && (ncols == 1 || ic0 + j < (int)ne01.x))
                    full_kq += slope * __half2float(maskh[j*ne11 + i_kv]);

                const float rn = fmaxf(KQ_max[j], full_kq + FATTN_KQ_MAX_OFFSET);
                const float ks = expf(KQ_max[j] - rn);
                KQ_max[j] = rn;
                const float ke = expf(full_kq - KQ_max[j]);
                KQ_sum[j] = KQ_sum[j] * ks + ke;

                #pragma unroll
                for (int e = 0; e < nelems; ++e) VKQ[j][e] *= ks;

                // ---- V dequant + inv-Hadamard + VKQ ----
                if constexpr (use_block_unroll) {
                    #pragma unroll
                    for (int b = 0; b < nblocks; ++b) {
                        const float d_v = __half2float(V_blk[b].d);
                        const float m_v = __half2float(V_blk[b].m);
                        const float mean_v = m_v;  // mean is stored directly in m field
                        #pragma unroll
                        for (int e = 0; e < elems_per_block; ++e) {
                            const int ti = tid + e * nthreads;
                            const uint8_t code = (V_blk[b].qs[by_blk[e]] >> shift_blk[e]) & 0x03;
                            sh_val_had[ti] = OSCAR2_CENTROIDS_DEV[code] * d_v + m_v - mean_v;
                        }
                        __syncwarp();
                        // P_br: reorder dequantized values from bit-reversal to natural order
                        { float _pbr[elems_per_block];
                          for (int _e = 0; _e < elems_per_block; ++_e) _pbr[_e] = sh_val_had[tid + _e * nthreads];
                          for (int _e = 0; _e < elems_per_block; ++_e) sh_val_had[P_BR_DEV[tid + _e * nthreads]] = _pbr[_e]; }
                        __syncwarp();
                        hadamard_inverse_128_32w(sh_val_had, tid);
                        #pragma unroll
                        for (int e = 0; e < elems_per_block; ++e) {
                            const int ti = tid + e * nthreads;
                            VKQ[j][b * elems_per_block + e] += ke * (sh_val_had[ti] + mean_v);
                        }
                    }
                } else {
                    const float d_v0 = __half2float(V_blk[0].d);
                    const float m_v0 = __half2float(V_blk[0].m);
                    #pragma unroll
                    for (int e = 0; e < nelems; ++e) {
                        const uint8_t code = (V_blk[0].qs[by_blk[e]] >> shift_blk[e]) & 0x03;
                        VKQ[j][e] += ke * (OSCAR2_CENTROIDS_DEV[code] * d_v0 + m_v0);
                    }
                }
            }

        }
    }
    // Write results
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (ncols > 1 && ic0 + j >= (int)ne01.x) break;
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
