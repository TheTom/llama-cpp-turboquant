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
// Note: P_br (bit-reversal permutation from the OSCAR paper) is CPU-only;
// the GPU set_rows kernel does not apply it, so P_BR_DEV is not declared here.
static __device__ const float OSCAR2_CENTROIDS_DEV[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};

// ---------------------------------------------------------------------------
// Single-threaded helpers (fallback for D < 128)
// ---------------------------------------------------------------------------

// K/V dequant happens inline in flash_attn_ext_oscar2 (per-warp, per-thread).
// Helpers removed: dequant_row_oscar2, dequant_row_oscar2_parallel.
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

    // Load Q into registers and rotate each 128-element block into the
    // Hadamard domain in which set_rows stores K and V. The normalized
    // Hadamard is orthonormal and self-inverse, so dot(Q,K) is preserved
    // when both operands live in the same transformed domain. This lets
    // the inner loop skip the per-token inverse Hadamard on K and V.
    float Q_reg[ncols][nelems];
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float * Q_j = (const float *) (Q + j*nb01);
        #pragma unroll
        for (int e = 0; e < nelems; ++e) {
            Q_reg[j][e] = Q_j[tid + e * nthreads] * scale;
        }
        // Transform Q to Hadamard domain: Q_reg[j] <- H(Q_reg[j]).
        // set_rows stores K/V as H(val - mean) / sqrt(128). With Q also in H domain,
        // the dot product (H(Q) · H(K)) preserves Q·K (H is orthogonal self-inverse).
        #pragma unroll
        for (int b = 0; b < nblocks; ++b) {
            #pragma unroll
            for (int e = 0; e < elems_per_block; ++e) {
                sh_val_had[tid + e * nthreads] = Q_reg[j][b * elems_per_block + e];
            }
            __syncwarp();
            hadamard_inverse_128_32w(sh_val_had, tid);
            #pragma unroll
            for (int e = 0; e < elems_per_block; ++e) {
                Q_reg[j][b * elems_per_block + e] = sh_val_had[tid + e * nthreads];
            }
            __syncwarp();
        }
    }

    float KQ_max[ncols], KQ_sum[ncols];
    float VKQ[ncols][nelems];
    float VKQ_mean[ncols][nblocks];
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2;
        KQ_sum[j] = 0.0f;
        #pragma unroll
        for (int e = 0; e < nelems; ++e) VKQ[j][e] = 0.0f;
        #pragma unroll
        for (int b = 0; b < nblocks; ++b) VKQ_mean[j][b] = 0.0f;
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
            assert(nb11 == nblocks * (int32_t)sizeof(block_oscar2)); // stride must match block layout
            const block_oscar2 * V_blk = (const block_oscar2 *)(V + i_kv * nb21);

            // ---- K dequant + dot product (K already lives in Hadamard domain) ----
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
                        const float val = OSCAR2_CENTROIDS_DEV[code] * d_k;
                        sum += val * Q_reg[j][b * elems_per_block + e];
                    }
                    // Per-block K mean correction: the mean was subtracted before
                    // Hadamard transform in set_rows, but is stored in block_oscar2.m.
                    // The missing term in the centered dot is mean * sum(Q_over_block).
                    // In Hadamard domain, Q_had[0] = sum(Q) / sqrt(128), so the
                    // correction is m_k * Q_had[0] * sqrt(128). Only thread 0 (tid==0)
                    // holds the DC component Q_had[0] at Q_reg[j][b * elems_per_block].
                    if (tid == 0) {
                        sum += m_k * Q_reg[j][b * elems_per_block] * 11.3137085f; // sqrtf(128.0f)
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

                // ---- V dequant + accumulate in Hadamard domain ----
                // Keep the mean separate so the final output semantics match
                // the original kernel (inv-Hadamard applied to centered value,
                // then mean added back).
                #pragma unroll
                for (int b = 0; b < nblocks; ++b) {
                    const float d_v = __half2float(V_blk[b].d);
                    const float m_v = __half2float(V_blk[b].m);
                    VKQ_mean[j][b] += ke * m_v;
                    #pragma unroll
                    for (int e = 0; e < elems_per_block; ++e) {
                        const int ti = tid + e * nthreads;
                        const uint8_t code = (V_blk[b].qs[by_blk[e]] >> shift_blk[e]) & 0x03;
                        const float val = OSCAR2_CENTROIDS_DEV[code] * d_v; // centered value in Hadamard domain
                        VKQ[j][b * elems_per_block + e] += ke * val;
                    }
                }
            } // end score for-j loop
        } // end i_kv loop
    } // end kv_base loop

    // ---- Write results: inverse-transform each 128-element block from Hadamard
    // domain back to natural domain, then add the accumulated per-block mean.
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (ncols > 1 && ic0 + j >= (int)ne01.x) break;
        const float iks = gridDim.y == 1 ? 1.0f / KQ_sum[j] : 1.0f;
        if (gridDim.y != 1 && tid == 0) {
            int mi = ((sequence * (int)ne01.x + ic0 + j) * ne02 + head) * gridDim.y + blockIdx.y;
            dst_meta_ptr[mi] = make_float2(KQ_max[j], KQ_sum[j]);
        }
        #pragma unroll
        for (int b = 0; b < nblocks; ++b) {
            #pragma unroll
            for (int e = 0; e < elems_per_block; ++e) {
                sh_val_had[tid + e * nthreads] = VKQ[j][b * elems_per_block + e] * iks;
            }
            __syncwarp();
            hadamard_inverse_128_32w(sh_val_had, tid);
            #pragma unroll
            for (int e = 0; e < elems_per_block; ++e) {
                int di = tid + e * nthreads + b * QK_OSCAR2;
                float val = sh_val_had[tid + e * nthreads] + VKQ_mean[j][b] * iks;
                if (gridDim.y == 1)
                    dst_ptr[(((sequence * (int)ne01.x + ic0 + j) * ne02 + head)) * D + di] = val;
                else
                    dst_ptr[(((sequence * (int)ne01.x + ic0 + j) * ne02 + head) * gridDim.y + blockIdx.y) * D + di] = val;
            }
            __syncwarp();
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

    constexpr size_t nbytes = 0;
    constexpr int nwarps = 1;

    // nbatch_fa controls the number of parallel KV blocks. Using D (head dim)
    // instead of K->ne[1] enables gridDim.y > 1 for prefill, allowing multiple
    // warps to process different KV position ranges in parallel via the
    // flash_attn_combine_results merge path. Decode (Q->ne[1]==1) naturally
    // falls back to single block since K->ne[1] < D → ntiles_KV = 1.
    const int nbatch_fa = D;

    // F2: multi-column prefill. Process more query tokens per block to
    // reduce redundant KV reads during prefill (large Q->ne[1]).
    // ncols=8 is safe for D<=256 (~108 regs/thread); cap at ncols=4
    // for D=512 (~200 regs/thread) to avoid register spilling.
    int ncols = 1;
    if (Q->ne[1] > 1) {
        if constexpr (D <= 256) {
            ncols = Q->ne[1] >= 8 ? 8 : (Q->ne[1] >= 4 ? 4 : 2);
        } else {
            ncols = Q->ne[1] >= 4 ? 4 : 2;
        }
    }

    auto launch = [&](int ncols_val, bool lsc) {
        if (lsc) {
            switch (ncols_val) {
                case 1: { fattn_kernel_t k = flash_attn_ext_oscar2<D, 1, true, type_K, type_V>; launch_fattn<D, 1, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false); } break;
                case 2: { fattn_kernel_t k = flash_attn_ext_oscar2<D, 2, true, type_K, type_V>; launch_fattn<D, 2, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false); } break;
                case 4: { fattn_kernel_t k = flash_attn_ext_oscar2<D, 4, true, type_K, type_V>; launch_fattn<D, 4, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false); } break;
                case 8: { fattn_kernel_t k = flash_attn_ext_oscar2<D, 8, true, type_K, type_V>; launch_fattn<D, 8, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false); } break;
                default: GGML_ABORT("unsupported ncols for oscar2 FA"); break;
            }
        } else {
            switch (ncols_val) {
                case 1: { fattn_kernel_t k = flash_attn_ext_oscar2<D, 1, false, type_K, type_V>; launch_fattn<D, 1, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false); } break;
                case 2: { fattn_kernel_t k = flash_attn_ext_oscar2<D, 2, false, type_K, type_V>; launch_fattn<D, 2, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false); } break;
                case 4: { fattn_kernel_t k = flash_attn_ext_oscar2<D, 4, false, type_K, type_V>; launch_fattn<D, 4, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false); } break;
                case 8: { fattn_kernel_t k = flash_attn_ext_oscar2<D, 8, false, type_K, type_V>; launch_fattn<D, 8, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false); } break;
                default: GGML_ABORT("unsupported ncols for oscar2 FA"); break;
            }
        }
    };

    launch(ncols, logit_softcap != 0.0f);
}
