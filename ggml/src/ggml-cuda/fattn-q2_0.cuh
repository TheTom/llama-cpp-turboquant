#include "common.cuh"
#include "fattn-common.cuh"

// Flash attention kernel for q2_0 K+V with proper Hadamard dequant.
// Strategy:
//   - Thread 0 loads K/V rows from global memory, dequants with Hadamard, to shared memory
//   - All 32 threads compute dot products using shared memory
//   - Online softmax across positions

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

// Thread 0 only: decode a q2_0 row with Hadamard and mean into a float buffer.
// Hadamard groups of 128 elements (4 blocks) with mean stored in first block of each group.
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
    constexpr int nthreads = WARP_SIZE;
    static_assert(D % nthreads == 0, "D not divisible by WARP_SIZE");
    constexpr int nelems = D / nthreads;

    __shared__ float s_buf[2 * D]; // [0..D-1]: K, [D..2D-1]: V

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

    // Load Q into registers
    float Q_reg[ncols][nelems];
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float * Q_j = (const float *) (Q + j*nb01);
        #pragma unroll
        for (int e = 0; e < nelems; ++e) {
            Q_reg[j][e] = Q_j[threadIdx.x + e * nthreads] * scale;
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

            // Thread 0: dequant K row to s_buf[0..D-1]
            if (threadIdx.x == 0) {
                if constexpr (type_K == GGML_TYPE_Q2_0) {
                    dequant_row_q2_0((const block_q2_0 *)(K + i_kv * nb11), D, s_buf);
                } else if constexpr (type_K == GGML_TYPE_F16) {
                    dequant_row_f16((const half *)(K + i_kv * nb11), D, s_buf);
                } else if constexpr (type_K == GGML_TYPE_Q8_0) {
                    dequant_row_q8_0((const block_q8_0 *)(K + i_kv * nb11), D, s_buf);
                }
            }
            __syncwarp();

            // Compute KQ dot product (all threads use shared memory)
            float KQ_val[ncols] = {0.0f};
            #pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = 0.0f;
                #pragma unroll
                for (int e = 0; e < nelems; ++e) {
                    sum += s_buf[threadIdx.x + e * nthreads] * Q_reg[j][e];
                }
                KQ_val[j] = sum;
            }

            // Full KQ via warp reduction
            #pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float full_kq = warp_reduce_sum(KQ_val[j]);

                if (use_logit_softcap) full_kq = logit_softcap * tanhf(full_kq);
                if (maskh && (ncols == 1 || ic0 + j < (int)ne01.z))
                    full_kq += slope * __half2float(maskh[j*ne11 + i_kv]);

                const float nm = fmaxf(KQ_max[j], full_kq + FATTN_KQ_MAX_OFFSET);
                const float rn = warp_reduce_max(nm);
                const float ks = expf(KQ_max[j] - rn);
                KQ_max[j] = rn;
                const float ke = expf(full_kq - KQ_max[j]);
                KQ_sum[j] = KQ_sum[j] * ks + ke;

                #pragma unroll
                for (int e = 0; e < nelems; ++e) VKQ[j][e] *= ks;

                // Thread 0: dequant V row to s_buf[D..2D-1]
                if (threadIdx.x == 0) {
                    if constexpr (type_V == GGML_TYPE_Q2_0) {
                        dequant_row_q2_0((const block_q2_0 *)(V + i_kv * nb21), D, s_buf + D);
                    } else if constexpr (type_V == GGML_TYPE_F16) {
                        dequant_row_f16((const half *)(V + i_kv * nb21), D, s_buf + D);
                    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
                        dequant_row_q8_0((const block_q8_0 *)(V + i_kv * nb21), D, s_buf + D);
                    }
                }
                __syncwarp();

                #pragma unroll
                for (int e = 0; e < nelems; ++e) {
                    VKQ[j][e] += ke * s_buf[D + threadIdx.x + e * nthreads];
                }
            }
        }
    }

    // Write results
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (ncols > 1 && ic0 + j >= (int)ne01.z) break;
        const float iks = gridDim.y == 1 ? 1.0f / KQ_sum[j] : 1.0f;
        if (gridDim.y != 1 && threadIdx.x == 0) {
            int mi = ((sequence * (int)ne01.z + ic0 + j) * ne02 + head) * gridDim.y + blockIdx.y;
            dst_meta_ptr[mi] = make_float2(KQ_max[j], KQ_sum[j]);
        }
        #pragma unroll
        for (int e = 0; e < nelems; ++e) {
            int di = threadIdx.x + e * nthreads;
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

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_q2_0_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    constexpr size_t nbytes = 2 * D * sizeof(float);

    // Force a single parallel block to bypass flash_attn_combine_results.
    const int nbatch_fa = dst->src[1]->ne[1];

    auto launch = [&](int ncols, bool lsc) {
        if (lsc) {
            if (ncols == 1) {
                fattn_kernel_t k = flash_attn_ext_q2_0<D, 1, true,  type_K, type_V>;
                launch_fattn<D, 1, 1>(ctx, dst, k, 1, nbytes, nbatch_fa, false, false, false);
            } else {
                fattn_kernel_t k = flash_attn_ext_q2_0<D, 2, true,  type_K, type_V>;
                launch_fattn<D, 2, 1>(ctx, dst, k, 1, nbytes, nbatch_fa, false, false, false);
            }
        } else {
            if (ncols == 1) {
                fattn_kernel_t k = flash_attn_ext_q2_0<D, 1, false, type_K, type_V>;
                launch_fattn<D, 1, 1>(ctx, dst, k, 1, nbytes, nbatch_fa, false, false, false);
            } else {
                fattn_kernel_t k = flash_attn_ext_q2_0<D, 2, false, type_K, type_V>;
                launch_fattn<D, 2, 1>(ctx, dst, k, 1, nbytes, nbatch_fa, false, false, false);
            }
        }
    };

    launch(Q->ne[1] > 1 ? 2 : 1, logit_softcap != 0.0f);
}
