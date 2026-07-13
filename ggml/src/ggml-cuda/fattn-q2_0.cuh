#include "common.cuh"
#include "fattn-common.cuh"

// Flash attention kernel for q2_0 K+V with proper Hadamard dequant.
//
// Block layout: blockDim.x = WARP_SIZE (32), blockDim.y = nwarps.
// For D >= 128: nwarps = 4 (128 threads), cooperative dequant, block-level reductions.
// For D < 128:  nwarps = 1 (32 threads), serial dequant (legacy path).
//
// Flat thread index: tid = threadIdx.y * WARP_SIZE + threadIdx.x

// ---------------------------------------------------------------------------
// Single-threaded helpers (fallback for D < 128)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void q2_0_hadamard_inplace(float * x, int n) {
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += h << 1) {
            for (int j = i; j < i + h; ++j) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float s = rsqrtf((float)n);
    for (int i = 0; i < n; ++i) x[i] *= s;
}

static __device__ __forceinline__ void dequant_row_q2_0(
    const block_q2_0 * blk, int D, float * buf) {
    constexpr int HAD = 128;
    constexpr int HAD_BLK = HAD / QK2_0; // 4
    const int ng = D / HAD;

    for (int ig = 0; ig < ng; ++ig) {
        const int base = ig * HAD_BLK;
        const float mean = __half2float(blk[base].m);

        for (int ib = 0; ib < HAD_BLK && (base + ib) * QK2_0 < D; ++ib) {
            const int bidx = base + ib;
            const float sigma = __half2float(blk[bidx].d);
            constexpr float c[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
            for (int j = 0; j < QK2_0; ++j) {
                uint8_t code = (blk[bidx].qs[j/4] >> (2*(j%4))) & 0x03;
                buf[bidx * QK2_0 + j] = sigma * c[code];
            }
        }

        q2_0_hadamard_inplace(buf + base * QK2_0, HAD);

        for (int i = 0; i < HAD; ++i) buf[base * QK2_0 + i] += mean;
    }
}

// Pre-Hadamard single-threaded dequant: decode + add mean, no inverse Hadamard.
static __device__ __forceinline__ void dequant_row_q2_0_preh(
    const block_q2_preh * blk, int D, float * buf) {
    constexpr int HAD = 128;
    constexpr int HAD_BLK = HAD / QK2_0; // 4
    const int ng = D / HAD;

    for (int ig = 0; ig < ng; ++ig) {
        const int base = ig * HAD_BLK;
        const float mean = __half2float(blk[base].m);

        for (int ib = 0; ib < HAD_BLK && (base + ib) * QK2_0 < D; ++ib) {
            const int bidx = base + ib;
            const float sigma = __half2float(blk[bidx].d);
            constexpr float c[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
            for (int j = 0; j < QK2_0; ++j) {
                uint8_t code = (blk[bidx].qs[j/4] >> (2*(j%4))) & 0x03;
                buf[bidx * QK2_0 + j] = sigma * c[code] + mean;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 128-thread cooperative helpers
// ---------------------------------------------------------------------------

// Cooperative 128-element inverse Hadamard.
// All 128 threads participate. Warp_id = threadIdx.y, lane = threadIdx.x.
// Thread i writes both x[i] and x[i+h]; thread i+h skips to avoid races.
// __syncthreads() between stages.
static __device__ __forceinline__ void hadamard_inverse_128(float * x) {
    constexpr int N = 128;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    for (int h = N/2; h > 0; h >>= 1) {
        if (tid < N && !(tid & h)) {
            const int pair = tid + h;
            const float a = x[tid];
            const float b = x[pair];
            x[tid]  = a + b;
            x[pair] = a - b;
        }
        __syncthreads();
    }
    constexpr float s = 0.08838834764f; // 1/sqrt(128)
    if (tid < N) {
        x[tid] *= s;
    }
    __syncthreads();
}

// Cooperative q2_0 dequant (128 threads, each handles D/128 elements).
static __device__ __forceinline__ void dequant_row_q2_0_parallel(
    const block_q2_0 * blk, int D, float * buf) {
    constexpr int HAD = 128;
    constexpr int nthreads = 128;
    const int ng = D / HAD;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    for (int ig = 0; ig < ng; ++ig) {
        const int group_base = ig * HAD;
        const int base_blk = group_base / QK2_0;

        const float mean = __half2float(blk[base_blk].m);

        for (int off = 0; off < HAD; off += nthreads) {
            const int elem = group_base + off + tid;
            if (elem >= D) break;
            const int ib    = elem / QK2_0;
            const int j     = elem % QK2_0;
            const float sigma = __half2float(blk[ib].d);
            constexpr float c[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
            uint8_t code = (blk[ib].qs[j/4] >> (2*(j%4))) & 0x03;
            buf[elem] = sigma * c[code];
        }
        __syncthreads();

        hadamard_inverse_128(buf + group_base);

        for (int off = 0; off < HAD; off += nthreads) {
            const int elem = group_base + off + tid;
            if (elem >= D) break;
            buf[elem] += mean;
        }
    }
    __syncthreads();
}

// Cooperative pre-Hadamard dequant: decode + add mean in one pass, no inverse Hadamard.
static __device__ __forceinline__ void dequant_row_q2_0_parallel_preh(
    const block_q2_preh * blk, int D, float * buf) {
    constexpr int HAD = 128;
    constexpr int nthreads = 128;
    const int ng = D / HAD;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    for (int ig = 0; ig < ng; ++ig) {
        const int group_base = ig * HAD;
        const int base_blk = group_base / QK2_0;
        const float mean = __half2float(blk[base_blk].m);

        for (int off = 0; off < HAD; off += nthreads) {
            const int elem = group_base + off + tid;
            if (elem >= D) break;
            const int ib    = elem / QK2_0;
            const int j     = elem % QK2_0;
            const float sigma = __half2float(blk[ib].d);
            constexpr float c[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
            uint8_t code = (blk[ib].qs[j/4] >> (2*(j%4))) & 0x03;
            buf[elem] = sigma * c[code] + mean;
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Plain dequant helpers (shared between paths)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void dequant_row_f16(
    const half * row, int D, float * buf) {
    for (int i = 0; i < D; ++i) {
        buf[i] = __half2float(row[i]);
    }
}

static __device__ __forceinline__ void dequant_row_q8_0(
    const block_q8_0 * blk, int D, float * buf) {
    constexpr int qk = 32;
    const int nb = D / qk;
    for (int ib = 0; ib < nb; ++ib) {
        const float d = __half2float(blk[ib].d);
        for (int j = 0; j < qk; ++j) {
            buf[ib * qk + j] = d * blk[ib].qs[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Block-level cross-warp reduction helpers (for 128-thread path)
//
// Block layout: blockDim.x = WARP_SIZE (32), blockDim.y = nwarps.
// Warp ID = threadIdx.y, Lane ID = threadIdx.x.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float block_reduce_sum_broadcast(float val, float * shared) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    val = warp_reduce_sum(val);

    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    const int total_warps = blockDim.y;
    if (warp_id == 0) {
        val = lane_id < total_warps ? shared[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    return shared[0];
}

static __device__ __forceinline__ float block_reduce_max_broadcast(float val, float * shared) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    val = warp_reduce_max(val);

    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    const int total_warps = blockDim.y;
    if (warp_id == 0) {
        val = lane_id < total_warps ? shared[lane_id] : -FLT_MAX/2;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    return shared[0];
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

template <int D, int ncols, bool use_logit_softcap, ggml_type type_K, ggml_type type_V>
static __global__ void flash_attn_ext_q2_0(
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
    // For D >= 128 use 4 warps (128 threads); for D=64 use 1 warp (32 threads)
    constexpr int nwarps_k = D >= 128 ? 4 : 1;
    constexpr int nthreads = nwarps_k * WARP_SIZE;
    static_assert(D % nthreads == 0, "D not divisible by nthreads");
    constexpr int nelems = D / nthreads;

    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    __shared__ float s_buf[2 * D]; // [0..D-1]: K, [D..2D-1]: V
    __shared__ float s_red[32];    // cross-warp reduction buffer

    const int ic0 = blockIdx.x * ncols;
    const int sequence = blockIdx.z / ne02;
    const int head     = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;

    const char * Q = Q_ptr + nb03*sequence + nb02*head + nb01*ic0;
    const char * K = K_ptr + nb13*sequence + nb12*(head / gqa_ratio) + blockIdx.y * nthreads * nb11;
    const char * V = V_ptr + nb23*sequence + nb22*(head / gqa_ratio) + blockIdx.y * nthreads * nb21;
    const half * maskh = mask_ptr ? (const half *)(mask_ptr + nb33*(sequence % ne33) + nb31*ic0 + blockIdx.y * nthreads) : nullptr;
    const float * sinks = sinks_ptr ? (const float *)(sinks_ptr + (sequence*ne02 + head) * 2) : nullptr;
    GGML_UNUSED(sinks); // TODO: implement sink merging

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Load Q into registers (flat indexing: tid)
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

            // ---- K dequant ----
            if constexpr (type_K == GGML_TYPE_Q2_0) {
                if constexpr (nwarps_k > 1) {
                    dequant_row_q2_0_parallel((const block_q2_0 *)(K + i_kv * nb11), D, s_buf);
                } else {
                    if (tid == 0) {
                        dequant_row_q2_0((const block_q2_0 *)(K + i_kv * nb11), D, s_buf);
                    }
                    __syncwarp();
                }
            } else if constexpr (type_K == GGML_TYPE_Q2_PREH) {
                if constexpr (nwarps_k > 1) {
                    dequant_row_q2_0_parallel_preh((const block_q2_preh *)(K + i_kv * nb11), D, s_buf);
                } else {
                    if (tid == 0) {
                        dequant_row_q2_0_preh((const block_q2_preh *)(K + i_kv * nb11), D, s_buf);
                    }
                    __syncwarp();
                }
            } else {
                if (tid == 0) {
                    if constexpr (type_K == GGML_TYPE_F16) {
                        dequant_row_f16((const half *)(K + i_kv * nb11), D, s_buf);
                    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
                        dequant_row_q8_0((const block_q8_0 *)(K + i_kv * nb11), D, s_buf);
                    }
                }
                if constexpr (nwarps_k > 1) {
                    __syncthreads();
                } else {
                    __syncwarp();
                }
            }

            // ---- KQ dot product (flat indexing: tid) ----
            float KQ_val[ncols] = {0.0f};
            #pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = 0.0f;
                #pragma unroll
                for (int e = 0; e < nelems; ++e) {
                    sum += s_buf[tid + e * nthreads] * Q_reg[j][e];
                }
                KQ_val[j] = sum;
            }

            #pragma unroll
            for (int j = 0; j < ncols; ++j) {
                // Full KQ
                float full_kq;
                if constexpr (nwarps_k > 1) {
                    full_kq = block_reduce_sum_broadcast(KQ_val[j], s_red);
                } else {
                    full_kq = warp_reduce_sum(KQ_val[j]);
                }

                if (use_logit_softcap) full_kq = logit_softcap * tanhf(full_kq);
                if (maskh && (ncols == 1 || ic0 + j < (int)ne01.z))
                    full_kq += slope * __half2float(maskh[j*ne11 + i_kv]);

                const float nm = fmaxf(KQ_max[j], full_kq + FATTN_KQ_MAX_OFFSET);
                float rn;
                if constexpr (nwarps_k > 1) {
                    rn = block_reduce_max_broadcast(nm, s_red);
                } else {
                    rn = warp_reduce_max(nm);
                }
                const float ks = expf(KQ_max[j] - rn);
                KQ_max[j] = rn;
                const float ke = expf(full_kq - KQ_max[j]);
                KQ_sum[j] = KQ_sum[j] * ks + ke;

                #pragma unroll
                for (int e = 0; e < nelems; ++e) VKQ[j][e] *= ks;
            // ---- V dequant ----
                if constexpr (type_V == GGML_TYPE_Q2_0) {
                    if constexpr (nwarps_k > 1) {
                        dequant_row_q2_0_parallel((const block_q2_0 *)(V + i_kv * nb21), D, s_buf + D);
                    } else {
                        if (tid == 0) {
                            dequant_row_q2_0((const block_q2_0 *)(V + i_kv * nb21), D, s_buf + D);
                        }
                        __syncwarp();
                    }
                } else if constexpr (type_V == GGML_TYPE_Q2_PREH) {
                    if constexpr (nwarps_k > 1) {
                        dequant_row_q2_0_parallel_preh((const block_q2_preh *)(V + i_kv * nb21), D, s_buf + D);
                    } else {
                        if (tid == 0) {
                            dequant_row_q2_0_preh((const block_q2_preh *)(V + i_kv * nb21), D, s_buf + D);
                        }
                        __syncwarp();
                    }
                } else {
                    if (tid == 0) {
                        if constexpr (type_V == GGML_TYPE_F16) {
                            dequant_row_f16((const half *)(V + i_kv * nb21), D, s_buf + D);
                        } else if constexpr (type_V == GGML_TYPE_Q8_0) {
                            dequant_row_q8_0((const block_q8_0 *)(V + i_kv * nb21), D, s_buf + D);
                        }
                    }
                    if constexpr (nwarps_k > 1) {
                        __syncthreads();
                    } else {
                        __syncwarp();
                    }
                }

                #pragma unroll
                for (int e = 0; e < nelems; ++e) {
                    VKQ[j][e] += ke * s_buf[D + tid + e * nthreads];
                }
            }
        }
    }

    // Write results (flat indexing: tid)
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
void ggml_cuda_flash_attn_ext_q2_0_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    constexpr size_t nbytes = 2 * D * sizeof(float);

    // nwarps = 4 for D >= 128, 1 for D == 64
    constexpr int nwarps = D >= 128 ? 4 : 1;

    // Force a single parallel block to bypass flash_attn_combine_results.
    const int nbatch_fa = dst->src[1]->ne[1];

    auto launch = [&](int ncols, bool lsc) {
        if (lsc) {
            if (ncols == 1) {
                fattn_kernel_t k = flash_attn_ext_q2_0<D, 1, true,  type_K, type_V>;
                launch_fattn<D, 1, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false);
            } else {
                fattn_kernel_t k = flash_attn_ext_q2_0<D, 2, true,  type_K, type_V>;
                launch_fattn<D, 2, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false);
            }
        } else {
            if (ncols == 1) {
                fattn_kernel_t k = flash_attn_ext_q2_0<D, 1, false, type_K, type_V>;
                launch_fattn<D, 1, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false);
            } else {
                fattn_kernel_t k = flash_attn_ext_q2_0<D, 2, false, type_K, type_V>;
                launch_fattn<D, 2, 1>(ctx, dst, k, nwarps, nbytes, nbatch_fa, false, false, false);
            }
        }
    };

    launch(Q->ne[1] > 1 ? 2 : 1, logit_softcap != 0.0f);
}
