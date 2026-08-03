// OSCAR2 dedicated flash attention kernel
// Per-128-vector (QK_OSCAR2 = 128) Lloyd-Max INT2 dequant:
//   val = inv-Hadamard(OSCAR2_LM_CENTROIDS[code] * sigma) + mean
// Levels: {-1.510257, -0.452734, 0.452734, 1.510257} for N(0,1).
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
static __device__ const float OSCAR2_CENTROIDS_DEV[4] = {-1.510257f, -0.452734f, 0.452734f, 1.510257f};

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
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        // HP tier (F16 sink+recent, stored in Hadamard domain) - null/0 when disabled
        const char  * K_hp_ptr,
        const char  * V_hp_ptr,
        const char  * mask_hp_ptr,
        const int32_t ne_hp11, const int32_t ne_hp12, const int32_t ne_hp13,
                            const int32_t nb_hp11, const int32_t nb_hp12, const int64_t nb_hp13,
                            const int32_t nb_vhp11, const int32_t nb_vhp12, const int64_t nb_vhp13,
                            const int32_t ne_hp31, const int32_t ne_hp32, const int32_t ne_hp33,
                            const int32_t nb_hp31, const int32_t nb_hp32, const int64_t nb_hp33) {

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

    // HP tier bases (per stream/head); HP rows are not tiled by blockIdx.y, so every
    // partial block processes the full HP set (correct under flash_attn_combine_results).
    const char * K_hp = K_hp_ptr ? K_hp_ptr + nb_hp13*sequence + nb_hp12*(head / gqa_ratio) : nullptr;
    const char * V_hp = V_hp_ptr ? V_hp_ptr + nb_vhp13*sequence + nb_vhp12*(head / gqa_ratio) : nullptr;
    const half * maskh_hp = mask_hp_ptr ? (const half *)mask_hp_ptr + (nb_hp33/2)*(sequence % ne_hp33) + (nb_hp31/2)*ic0 : nullptr;

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
                // the per-block mean accumulator must be rescaled on max shift too,
                // otherwise its contribution drifts relative to KQ_sum and VKQ
                #pragma unroll
                for (int b = 0; b < nblocks; ++b) VKQ_mean[j][b] *= ks;

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

    // ---- HP tier: F16 sink+recent tokens, stored in the Hadamard domain (no mean).
    // The Q transform above (normalized Hadamard) makes the HP dot exact: (H q) . (H k).
    // Masked-out (empty) HP slots carry -inf and contribute zero via exp(-inf).
    // NOTE: only blockIdx.y == 0 processes the HP tier. When gridDim.y > 1 the LP
    // KV is split across partial blocks merged by flash_attn_combine_results; the
    // HP tier must appear in exactly one partial, or it gets counted gridDim.y times.
    // After permute(0,2,1,3): ne[1] = n_hp_cells, ne[2] = n_head_kv
    const int n_hp_cells = ne_hp11;
    if (K_hp && blockIdx.y == 0) {
        for (int i_hp = 0; i_hp < n_hp_cells; ++i_hp) {
            const half * Kh = (const half *)(K_hp + i_hp * nb_hp11);
            const half * Vh = (const half *)(V_hp + i_hp * nb_vhp11);

            float KQ_val_hp[ncols];
            #pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if (ncols > 1 && ic0 + j >= (int)ne01.x) break;
                float sum = 0.0f;
                #pragma unroll
                for (int e = 0; e < nelems; ++e) {
                    sum += __half2float(Kh[tid + e * nthreads]) * Q_reg[j][e];
                }
                KQ_val_hp[j] = warp_reduce_sum(sum);
            }

            #pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if (ncols > 1 && ic0 + j >= (int)ne01.x) break;
                float full_kq = KQ_val_hp[j];
                if (use_logit_softcap) full_kq = logit_softcap * tanhf(full_kq);
                if (maskh_hp) full_kq += slope * __half2float(maskh_hp[j*n_hp_cells + i_hp]);

                const float rn = fmaxf(KQ_max[j], full_kq + FATTN_KQ_MAX_OFFSET);
                const float ks = expf(KQ_max[j] - rn);
                KQ_max[j] = rn;
                const float ke = expf(full_kq - KQ_max[j]);
                KQ_sum[j] = KQ_sum[j] * ks + ke;

                #pragma unroll
                for (int e = 0; e < nelems; ++e) VKQ[j][e] *= ks;

                // HP V accumulate (f16 in Hadamard domain, no mean term)
                #pragma unroll
                for (int e = 0; e < nelems; ++e) {
                    VKQ[j][e] += ke * __half2float(Vh[tid + e * nthreads]);
                }
            }
        }
    }

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
        ne31, ne32, ne33, nb31, nb32, nb33,
        K_hp_ptr, V_hp_ptr, mask_hp_ptr,
        ne_hp11, ne_hp12, ne_hp13, nb_hp11, nb_hp12, nb_hp13, nb_vhp11, nb_vhp12, nb_vhp13,
        ne_hp31, ne_hp32, ne_hp33, nb_hp31, nb_hp32, nb_hp33);
#endif
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

// Dedicated launcher for the oscar2 FA kernel. Replicates the grid/workspace setup
// of launch_fattn (ncols1 = ncols2 = 1, no stream-k, no f16 conversion) and
// additionally forwards the HP tier tensors (dst->src[5..7]) to the kernel.
template <int D, typename KF>
static void launch_fattn_oscar2(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        KF fattn_kernel,
        const int nwarps,
        const size_t nbytes_shared,
        const int nbatch_fa) {

    const ggml_tensor * Q      = dst->src[0];
    const ggml_tensor * K      = dst->src[1];
    const ggml_tensor * V      = dst->src[2];
    const ggml_tensor * mask   = dst->src[3];
    const ggml_tensor * sinks  = dst->src[4];
    const ggml_tensor * K_hp   = dst->src[5];
    const ggml_tensor * V_hp   = dst->src[6];
    const ggml_tensor * mask_hp = dst->src[7];
    const ggml_tensor * KQV    = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!K_hp || (K_hp->type == GGML_TYPE_F16 && V_hp && V_hp->type == GGML_TYPE_F16));
    GGML_ASSERT(!mask_hp || mask_hp->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    const size_t nb11 = K->nb[1];
    const size_t nb12 = K->nb[2];
    const size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    const size_t nb21 = V->nb[1];
    const size_t nb22 = V->nb[2];
    const size_t nb23 = V->nb[3];

    constexpr int ncols1 = 1;
    constexpr int ncols2 = 1;

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization: scan the mask to skip masked-out KV regions (same as launch_fattn)
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;
        KV_max.alloc(ne_KV_max);
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
            (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    // non-stream-k path only (the oscar2 kernel has no stream-k support)
    parallel_blocks = std::min(parallel_blocks, ntiles_KV);

    const int blocks_per_wave = nsm * max_blocks_per_sm;
    int nwaves_best = 0;
    int efficiency_percent_best = 0;
    for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
        const int nblocks_total = ntiles_dst * parallel_blocks_test;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

        if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
            break;
        }

        if (efficiency_percent > efficiency_percent_best) {
            nwaves_best = nwaves;
            efficiency_percent_best = efficiency_percent;
            parallel_blocks = parallel_blocks_test;
        }
    }

    const dim3 blocks_num(ntiles_x, parallel_blocks, ntiles_z_gqa*K->ne[2]*Q->ne[3]);

    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % WARP_SIZE == 0);
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        // HP tier
        K_hp ? ((const char *) K_hp->data) : nullptr,
        V_hp ? ((const char *) V_hp->data) : nullptr,
        mask_hp ? ((const char *) mask_hp->data) : nullptr,
        K_hp ? K_hp->ne[1] : 0, K_hp ? K_hp->ne[2] : 0, K_hp ? K_hp->ne[3] : 0,
        K_hp ? K_hp->nb[1] : 0, K_hp ? K_hp->nb[2] : 0, K_hp ? K_hp->nb[3] : 0,
        V_hp ? V_hp->nb[1] : 0, V_hp ? V_hp->nb[2] : 0, V_hp ? V_hp->nb[3] : 0,
        mask_hp ? mask_hp->ne[1] : 0, mask_hp ? mask_hp->ne[2] : 0, mask_hp ? mask_hp->ne[3] : 0,
        mask_hp ? mask_hp->nb[1] : 0, mask_hp ? mask_hp->nb[2] : 0, mask_hp ? mask_hp->nb[3] : 0
    );
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks > 1) {
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_combine_results<D>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}

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
                case 1: { auto k = flash_attn_ext_oscar2<D, 1, true, type_K, type_V>;  launch_fattn_oscar2<D>(ctx, dst, k, nwarps, nbytes, nbatch_fa); } break;
                case 2: { auto k = flash_attn_ext_oscar2<D, 2, true, type_K, type_V>;  launch_fattn_oscar2<D>(ctx, dst, k, nwarps, nbytes, nbatch_fa); } break;
                case 4: { auto k = flash_attn_ext_oscar2<D, 4, true, type_K, type_V>;  launch_fattn_oscar2<D>(ctx, dst, k, nwarps, nbytes, nbatch_fa); } break;
                case 8: { auto k = flash_attn_ext_oscar2<D, 8, true, type_K, type_V>;  launch_fattn_oscar2<D>(ctx, dst, k, nwarps, nbytes, nbatch_fa); } break;
                default: GGML_ABORT("unsupported ncols for oscar2 FA"); break;
            }
        } else {
            switch (ncols_val) {
                case 1: { auto k = flash_attn_ext_oscar2<D, 1, false, type_K, type_V>; launch_fattn_oscar2<D>(ctx, dst, k, nwarps, nbytes, nbatch_fa); } break;
                case 2: { auto k = flash_attn_ext_oscar2<D, 2, false, type_K, type_V>; launch_fattn_oscar2<D>(ctx, dst, k, nwarps, nbytes, nbatch_fa); } break;
                case 4: { auto k = flash_attn_ext_oscar2<D, 4, false, type_K, type_V>; launch_fattn_oscar2<D>(ctx, dst, k, nwarps, nbytes, nbatch_fa); } break;
                case 8: { auto k = flash_attn_ext_oscar2<D, 8, false, type_K, type_V>; launch_fattn_oscar2<D>(ctx, dst, k, nwarps, nbytes, nbatch_fa); } break;
                default: GGML_ABORT("unsupported ncols for oscar2 FA"); break;
            }
        }
    };

    launch(ncols, logit_softcap != 0.0f);
}
