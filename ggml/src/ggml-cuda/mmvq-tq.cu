/*
 * Fused mul_mat for TQ4_1S / TQ3_1S weight types.
 *
 * ne[1]≤8: dp4a multi-token kernel (weight reuse across tokens)
 * ne[1]>8: runtime TQ4_1S→q8_0 scratch + cuBLAS tensor core GEMM
 */

#include "mmvq-tq.cuh"
#include "turbo-quant.cuh"
#include "convert.cuh"
#include "mmq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>
using namespace nvcuda;
#define GGML_CUDA_USE_WMMA
#endif

#define MMVQ_TQ_NWARPS 4
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================================
// Pre-rotate activation to q8_1 format (for TQ4_1S dp4a path)
// ============================================================================

static __global__ void tq_prerotate_q8_1(
        const float * __restrict__ src,
        block_q8_1  * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    float sum = val;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) {
        dst[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
    }
}

// ============================================================================
// TQ4_1S: dp4a path with fixed int8 centroid LUT + q8_1 activation
// ============================================================================

// Fixed int8 centroid table: centroid_i8[i] = round(TQ4_CENTROIDS_WEIGHT[i] * 127 / 2.733)
// Rescale factor to recover float centroids: 2.733 / 127
static constexpr float TQ4_CENTROID_I8_RESCALE = 2.733f / 127.0f;

// Register-based centroid lookup: maps 4 qs bytes (1 uint32) to 2 packed 4× centroid_i8 for dp4a.
// Processes a full uint32 at once, sharing nibble extraction across both byte pairs.
__device__ __forceinline__ void tq4_cents8_reg(uint32_t four_bytes, int &c0, int &c1) {
    // 4 qs bytes -> 8 nibble indices -> 8 int8 centroids, in natural element order
    // (element 2k = low nibble of byte k, element 2k+1 = high nibble of byte k).
    //
    // This uses the hardware byte-permute (v_perm_b32) through get_int_from_table_16, the same
    // path IQ4_NL/MXFP4 use on ROCm. NOTE: do NOT express this with HIP's __byte_perm() wrapper
    // -- with a runtime selector its generic fallback lowers through a dynamically indexed byte
    // union that PromoteAlloca turns into a 32KB LDS staging area plus ds_read_u8 chains, which
    // caps occupancy and costs ~3x. __builtin_amdgcn_perm maps 1:1 to the instruction.
    //
    // The previous shift-based fallback avoided that bug but cost ~100 ALU ops per call (~400
    // per 32-weight block), which made the TQ decode kernels ALU-bound rather than
    // bandwidth-bound on CDNA.
    const int2 v = get_int_from_table_16((int) four_bytes, kvalues_tq4);

    // v.x = centroids for even elements (0,2,4,6), v.y = odd elements (1,3,5,7).
    // Interleave back to natural order: c0 = [e0,e1,e2,e3], c1 = [e4,e5,e6,e7].
#if defined(GGML_USE_HIP)
    c0 = (int) __builtin_amdgcn_perm(v.y, v.x, 0x05010400);
    c1 = (int) __builtin_amdgcn_perm(v.y, v.x, 0x07030602);
#else
    // nvcc / MUSA: prmt selector nibbles pick bytes 0-3 from the first operand and 4-7 from
    // the second. Same byte order as the HIP form above.
    c0 = __byte_perm(v.x, v.y, 0x5140);
    c1 = __byte_perm(v.x, v.y, 0x7362);
#endif
}

// ============================================================================
// Pre-rotate activation to float (for TQ3_1S scalar path)
// ============================================================================

// NOTE: dst is float, not half. The rotated activation for a given input
// element is REUSED against every one of the (up to 1024+) output rows in
// mul_mat_tq3_1s_multi/mul_mat_tq4_1s_scalar_multi below, so a per-element
// fp16 rounding error here is not independent per-row noise that averages
// out — it's a fixed bias replayed into every row's dot product. For real
// (non-uniform, elementwise-scaled) model activations — e.g. DeepSeek-V4's
// attn_norm output, which is RMSNorm * a learned per-channel weight with
// large dynamic range — this compounded into ~2% per-layer error at
// blk.0.attn_q_a (verified via eval-callback node diffing against the CPU
// reference: qr-0 sum -0.771243 true weights) but was invisible on
// synthetic/uniform test-backend-ops data, and accumulated across 61 layers
// into token-soup garbage output. Keeping the rotated activation in float
// (the WHT butterfly itself was always computed in float registers; only
// the final store truncated to half) removes this without touching the
// weight-side quantization or the WHT math itself.
static __global__ void tq_prerotate_activation(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    dst[offset] = val;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static __device__ __forceinline__ float tq3_cent_reg(uint32_t idx) {
    switch (idx & 7u) {
        case 0: return -1.996684f;
        case 1: return -1.291398f;
        case 2: return -0.740341f;
        case 3: return -0.247508f;
        case 4: return  0.230106f;
        case 5: return  0.725222f;
        case 6: return  1.277503f;
        case 7: return  1.988943f;
        default: return 0.0f;
    }
}
#endif

static __device__ __forceinline__ uint32_t tq3_extract_index_fast(const uint8_t * __restrict__ qs, int lane) {
    const int group = lane >> 3;
    const int shift = (lane & 7) * 3;
    const uint8_t * qp = qs + group * 3;
    const uint32_t packed = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
    return (packed >> shift) & 7u;
}

// ============================================================================
// Multi-token TQ4_1S dp4a kernel (ncols_dst ≤ 8)
// Weight data loaded once per block, reused across all ncols_dst tokens.
// ============================================================================

template <int ncols_dst>
static __global__ void mul_mat_tq4_1s_dp4a_multi(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,
        float            * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        const block_tq4_1s * blk = &x_row[ib];
        const float fd0 = __half2float(blk->d0);
        const float fd1 = __half2float(blk->d1);

        // Load weight once, reuse across all tokens
        const uint32_t * qs32 = (const uint32_t *)(blk->qs);
        const uint32_t w0 = qs32[0], w1 = qs32[1], w2 = qs32[2], w3 = qs32[3];

        int c0_0, c1_0, c0_1, c1_1, c0_2, c1_2, c0_3, c1_3;
        tq4_cents8_reg(w0, c0_0, c1_0);
        tq4_cents8_reg(w1, c0_1, c1_1);
        tq4_cents8_reg(w2, c0_2, c1_2);
        tq4_cents8_reg(w3, c0_3, c1_3);

        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const block_q8_1 * a_blk = &vy_q8[j * stride_col_y + ib];
            const float d_act = __half2float((__half)a_blk->ds.x);
            const int * a_qs = (const int *)(a_blk->qs);

            const int s0 = ggml_cuda_dp4a(c0_0, a_qs[0], ggml_cuda_dp4a(c1_0, a_qs[1],
                           ggml_cuda_dp4a(c0_1, a_qs[2], ggml_cuda_dp4a(c1_1, a_qs[3], 0))));
            const int s1 = ggml_cuda_dp4a(c0_2, a_qs[4], ggml_cuda_dp4a(c1_2, a_qs[5],
                           ggml_cuda_dp4a(c0_3, a_qs[6], ggml_cuda_dp4a(c1_3, a_qs[7], 0))));

            sumf[j] += d_act * (fd0 * (float)s0 + fd1 * (float)s1);
        }
    }

    // Apply centroid int8→float rescale + warp reduction
    #pragma unroll
    for (int j = 0; j < ncols_dst; j++)
        sumf[j] *= TQ4_CENTROID_I8_RESCALE;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

// ============================================================================
// Multi-token TQ3_1S scalar kernel (ncols_dst ≤ 8)
// ============================================================================

template <int ncols_dst>
static __global__ void mul_mat_tq3_1s_multi(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    __shared__ float s_lut[8];
    if (threadIdx.y == 0 && threadIdx.x < 8) {
        s_lut[threadIdx.x] = TQ3_CENTROIDS_WEIGHT[threadIdx.x];
    }
    __syncthreads();
#endif

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ3_0;
    const block_tq3_1s * x_row = ((const block_tq3_1s *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint32_t idx = tq3_extract_index_fast(x_row[ib].qs, lane);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        const float w = tq3_cent_reg(idx) * d;
#else
        const float w = s_lut[idx] * d;
#endif

        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const float act = vy_rot[j * stride_col_y + ib * QK_TQ3_0 + lane];
            sumf[j] = __fmaf_rn(act, w, sumf[j]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

// ============================================================================
// TQ4_1S scalar/float kernel (AMD fallback — no dp4a)
// Same pattern as TQ3_1S: pre-rotated float activations, scalar centroid lookup.
// On RDNA4, sudot4 throughput differs from NVIDIA dp4a — this path is faster.
// ============================================================================

template <int ncols_dst>
static __global__ void mul_mat_tq4_1s_scalar_multi(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    __shared__ float s_lut[16];
    if (threadIdx.y == 0 && threadIdx.x < 16) {
        s_lut[threadIdx.x] = TQ4_CENTROIDS_WEIGHT[threadIdx.x];
    }
    __syncthreads();

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
        const float w = s_lut[idx] * d;

        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const float act = vy_rot[j * stride_col_y + ib * QK_TQ4_1S + lane];
            sumf[j] += act * w;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

// ============================================================================
// Dispatch: ne[1]=1 (decode), ne[1]≤8 (multi-token dp4a / scalar)
// ne[1]>8 handled by ggml_cuda_mul_mat_tq4_1s_cublas (runtime dequant + cuBLAS)
// AMD: uses scalar half path for TQ4_1S (dp4a regresses on RDNA4)
// ============================================================================

template <int ncols_dst>
static void launch_tq4_1s_multi(
        const void * src0_d, const block_q8_1 * q8_buf,
        float * dst_d, int ncols_x, int nrows_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    mul_mat_tq4_1s_dp4a_multi<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, q8_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <int ncols_dst>
static void launch_tq4_1s_scalar_multi(
        const void * src0_d, const float * act_buf,
        float * dst_d, int ncols_x, int nrows_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    mul_mat_tq4_1s_scalar_multi<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, act_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <int ncols_dst>
static void launch_tq3_1s_multi(
        const void * src0_d, const float * act_buf,
        float * dst_d, int ncols_x, int nrows_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    mul_mat_tq3_1s_multi<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, act_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

#if defined(GGML_CUDA_USE_WMMA)
// ============================================================================
// Multi-token TQ4_1S WMMA Tensor Core kernel (ncols_dst > 8, prompt processing)
// ============================================================================

static __device__ __forceinline__ float tq4_cent_float(uint32_t idx) {
    switch (idx & 0xFu) {
        case 0:  return -2.732590f;
        case 1:  return -2.069017f;
        case 2:  return -1.618046f;
        case 3:  return -1.256231f;
        case 4:  return -0.942340f;
        case 5:  return -0.656759f;
        case 6:  return -0.388048f;
        case 7:  return -0.128395f;
        case 8:  return  0.128395f;
        case 9:  return  0.388048f;
        case 10: return  0.656759f;
        case 11: return  0.942340f;
        case 12: return  1.256231f;
        case 13: return  1.618046f;
        case 14: return  2.069017f;
        case 15: return  2.732590f;
        default: return  0.0f;
    }
}

template <int NWARPS>
static __global__ void mul_mat_tq4_1s_wmma_kernel(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_dst,
        const int stride_col_y,
        const int stride_col_dst) {

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int warp_id_in_grid_m = blockIdx.y * NWARPS + warp_id;
    const int warp_id_in_grid_n = blockIdx.x;

    const int m_base = warp_id_in_grid_m * WMMA_M;
    const int n_base = warp_id_in_grid_n * WMMA_N;

    if (m_base >= nrows_x || n_base >= ncols_dst) return;

    __shared__ half sh_a[NWARPS][WMMA_M][WMMA_K + 1];
    __shared__ half sh_b[NWARPS][WMMA_K][WMMA_N + 1];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);

    const int blocks_per_row = ncols_x / QK_TQ4_1S;

    for (int k_outer = 0; k_outer < ncols_x; k_outer += WMMA_K) {
        const int elem_idx = lane * 8;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int flat = elem_idx + e;
            const int r_local = flat / WMMA_K;
            const int k_local = flat % WMMA_K;
            const int r_global = m_base + r_local;
            const int k_global = k_outer + k_local;

            if (r_global < nrows_x && k_global < ncols_x) {
                const int ib = k_global / QK_TQ4_1S;
                const int lane_in_blk = k_global % QK_TQ4_1S;
                const block_tq4_1s * blk = ((const block_tq4_1s *)vx) + (int64_t)r_global * blocks_per_row + ib;

                const float d = (lane_in_blk < 16) ? __half2float(blk->d0) : __half2float(blk->d1);
                const uint8_t idx = (blk->qs[lane_in_blk / 2] >> ((lane_in_blk & 1) * 4)) & 0xFu;
                sh_a[warp_id][r_local][k_local] = __float2half(tq4_cent_float(idx) * d);
            } else {
                sh_a[warp_id][r_local][k_local] = __float2half(0.0f);
            }
        }

        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int flat = elem_idx + e;
            const int k_local = flat / WMMA_N;
            const int n_local = flat % WMMA_N;
            const int k_global = k_outer + k_local;
            const int n_global = n_base + n_local;

            if (n_global < ncols_dst && k_global < ncols_x) {
                const float act = vy_rot[n_global * stride_col_y + k_global];
                sh_b[warp_id][k_local][n_local] = __float2half(act);
            } else {
                sh_b[warp_id][k_local][n_local] = __float2half(0.0f);
            }
        }

        __syncwarp();

        wmma::load_matrix_sync(frag_a, &sh_a[warp_id][0][0], WMMA_K + 1);
        wmma::load_matrix_sync(frag_b, &sh_b[warp_id][0][0], WMMA_N + 1);

        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    __shared__ float sh_dst[NWARPS][WMMA_M][WMMA_N + 1];
    wmma::store_matrix_sync(&sh_dst[warp_id][0][0], frag_c, WMMA_N + 1, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const int flat = lane * 8 + e;
        const int r_local = flat / WMMA_N;
        const int n_local = flat % WMMA_N;
        const int r_global = m_base + r_local;
        const int n_global = n_base + n_local;

        if (r_global < nrows_x && n_global < ncols_dst) {
            dst[n_global * stride_col_dst + r_global] = sh_dst[warp_id][r_local][n_local];
        }
    }
}

static void launch_tq4_1s_wmma(
        const void * src0_d, const float * act_buf,
        float * dst_d, int ncols_x, int nrows_x, int ncols_dst,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {

    constexpr int NWARPS = 4;
    const dim3 block(NWARPS * 32, 1);
    const dim3 grid((ncols_dst + WMMA_N - 1) / WMMA_N, (nrows_x + (WMMA_M * NWARPS) - 1) / (WMMA_M * NWARPS));

    mul_mat_tq4_1s_wmma_kernel<NWARPS><<<grid, block, 0, stream>>>(
        src0_d, act_buf, dst_d, ncols_x, nrows_x, ncols_dst, stride_col_y, stride_col_dst);
}

// ============================================================================
// Multi-token TQ3_1S WMMA Tensor Core kernel (ncols_dst > 8, prompt processing)
// ============================================================================

template <int NWARPS>
static __global__ void mul_mat_tq3_1s_wmma_kernel(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_dst,
        const int stride_col_y,
        const int stride_col_dst) {

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int warp_id_in_grid_m = blockIdx.y * NWARPS + warp_id;
    const int warp_id_in_grid_n = blockIdx.x;

    const int m_base = warp_id_in_grid_m * WMMA_M;
    const int n_base = warp_id_in_grid_n * WMMA_N;

    if (m_base >= nrows_x || n_base >= ncols_dst) return;

    __shared__ half sh_a[NWARPS][WMMA_M][WMMA_K + 1];
    __shared__ half sh_b[NWARPS][WMMA_K][WMMA_N + 1];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);

    const int blocks_per_row = ncols_x / QK_TQ3_0;

    for (int k_outer = 0; k_outer < ncols_x; k_outer += WMMA_K) {
        const int elem_idx = lane * 8;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int flat = elem_idx + e;
            const int r_local = flat / WMMA_K;
            const int k_local = flat % WMMA_K;
            const int r_global = m_base + r_local;
            const int k_global = k_outer + k_local;

            if (r_global < nrows_x && k_global < ncols_x) {
                const int ib = k_global / QK_TQ3_0;
                const int lane_in_blk = k_global % QK_TQ3_0;
                const block_tq3_1s * blk = ((const block_tq3_1s *)vx) + (int64_t)r_global * blocks_per_row + ib;

                const float d = (lane_in_blk < 16) ? __half2float(blk->d0) : __half2float(blk->d1);
                const uint32_t idx = tq3_extract_index_fast(blk->qs, lane_in_blk);
                sh_a[warp_id][r_local][k_local] = __float2half(tq3_cent_reg(idx) * d);
            } else {
                sh_a[warp_id][r_local][k_local] = __float2half(0.0f);
            }
        }

        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int flat = elem_idx + e;
            const int k_local = flat / WMMA_N;
            const int n_local = flat % WMMA_N;
            const int k_global = k_outer + k_local;
            const int n_global = n_base + n_local;

            if (n_global < ncols_dst && k_global < ncols_x) {
                const float act = vy_rot[n_global * stride_col_y + k_global];
                sh_b[warp_id][k_local][n_local] = __float2half(act);
            } else {
                sh_b[warp_id][k_local][n_local] = __float2half(0.0f);
            }
        }

        __syncwarp();

        wmma::load_matrix_sync(frag_a, &sh_a[warp_id][0][0], WMMA_K + 1);
        wmma::load_matrix_sync(frag_b, &sh_b[warp_id][0][0], WMMA_N + 1);

        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    __shared__ float sh_dst[NWARPS][WMMA_M][WMMA_N + 1];
    wmma::store_matrix_sync(&sh_dst[warp_id][0][0], frag_c, WMMA_N + 1, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const int flat = lane * 8 + e;
        const int r_local = flat / WMMA_N;
        const int n_local = flat % WMMA_N;
        const int r_global = m_base + r_local;
        const int n_global = n_base + n_local;

        if (r_global < nrows_x && n_global < ncols_dst) {
            dst[n_global * stride_col_dst + r_global] = sh_dst[warp_id][r_local][n_local];
        }
    }
}

static void launch_tq3_1s_wmma(
        const void * src0_d, const float * act_buf,
        float * dst_d, int ncols_x, int nrows_x, int ncols_dst,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {

    constexpr int NWARPS = 4;
    const dim3 block(NWARPS * 32, 1);
    const dim3 grid((ncols_dst + WMMA_N - 1) / WMMA_N, (nrows_x + (WMMA_M * NWARPS) - 1) / (WMMA_M * NWARPS));

    mul_mat_tq3_1s_wmma_kernel<NWARPS><<<grid, block, 0, stream>>>(
        src0_d, act_buf, dst_d, ncols_x, nrows_x, ncols_dst, stride_col_y, stride_col_dst);
}
#endif // defined(GGML_CUDA_USE_WMMA)

// ============================================================================
// MoE (MUL_MAT_ID) matvec: capture-safe device-side expert routing.
//
// Replaces the generic ggml_cuda_mul_mat_id host-sync fallback for TQ weights at
// small batch. That fallback copies `ids` to the host, sorts tokens per expert on
// the CPU, copies back, and launches a data-dependent per-expert cuBLAS loop — two
// cudaStreamSynchronize per layer, and illegal to record into a CUDA graph.
//
// Here each block instead reads ids[sample*ids_stride + expert_slot] ON-DEVICE to
// select its expert's weight channel (mirrors the stock mmvq id path), so the launch
// is a fixed grid with no host round-trip: fully graph-capturable, and cheaper.
//   grid.x = row blocks (output neurons)   grid.y = expert slot (ne1 = n_expert_used)
//   grid.z = token/sample (ne2)            one dot product per (row, expert_slot, sample)
// ============================================================================

static __global__ void mul_mat_tq3_1s_moe(
        const void    * __restrict__ vx,       // all experts, base pointer
        const float   * __restrict__ vy_rot,   // pre-rotated activations
        float         * __restrict__ dst,
        const int32_t * __restrict__ ids,
        const int     ncols_x,
        const int     nrows_x,
        const int64_t nb_expert,               // byte stride between experts in vx
        const int     ids_stride,              // element stride between samples in ids
        const int     nchannels_y,             // ne11
        const int64_t stride_channel_y,        // float elements to next y-channel in vy_rot
        const int64_t stride_sample_y,         // float elements to next sample in vy_rot
        const int64_t stride_channel_dst,      // elements to next expert slot in dst
        const int64_t stride_sample_dst) {     // elements to next sample in dst

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    __shared__ float s_lut[8];
    if (threadIdx.y == 0 && threadIdx.x < 8) {
        s_lut[threadIdx.x] = TQ3_CENTROIDS_WEIGHT[threadIdx.x];
    }
    __syncthreads();
#endif

    const int expert_slot = blockIdx.y;
    const int sample      = blockIdx.z;
    const int expert      = ids[sample * ids_stride + expert_slot];
    const int channel_y   = expert_slot % nchannels_y;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ3_0;
    const block_tq3_1s * x_row =
        (const block_tq3_1s *) ((const char *) vx + (int64_t) expert * nb_expert)
        + (int64_t) row * blocks_per_row;
    const float * act = vy_rot + sample * stride_sample_y + (int64_t) channel_y * stride_channel_y;

    float sumf = 0.0f;
    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint32_t idx = tq3_extract_index_fast(x_row[ib].qs, lane);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        const float w = tq3_cent_reg(idx) * d;
#else
        const float w = s_lut[idx] * d;
#endif
        sumf = __fmaf_rn(act[ib * QK_TQ3_0 + lane], w, sumf);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);
    }

    if (lane == 0) {
        dst[sample * stride_sample_dst + (int64_t) expert_slot * stride_channel_dst + row] = sumf;
    }
}

static __global__ void mul_mat_tq4_1s_scalar_moe(
        const void    * __restrict__ vx,
        const float   * __restrict__ vy_rot,
        float         * __restrict__ dst,
        const int32_t * __restrict__ ids,
        const int     ncols_x,
        const int     nrows_x,
        const int64_t nb_expert,
        const int     ids_stride,
        const int     nchannels_y,
        const int64_t stride_channel_y,
        const int64_t stride_sample_y,
        const int64_t stride_channel_dst,
        const int64_t stride_sample_dst) {

    __shared__ float s_lut[16];
    if (threadIdx.y == 0 && threadIdx.x < 16) {
        s_lut[threadIdx.x] = TQ4_CENTROIDS_WEIGHT[threadIdx.x];
    }
    __syncthreads();

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int expert_slot = blockIdx.y;
    const int sample      = blockIdx.z;
    const int expert      = ids[sample * ids_stride + expert_slot];
    const int channel_y   = expert_slot % nchannels_y;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row =
        (const block_tq4_1s *) ((const char *) vx + (int64_t) expert * nb_expert)
        + (int64_t) row * blocks_per_row;
    const float * act = vy_rot + sample * stride_sample_y + (int64_t) channel_y * stride_channel_y;

    float sumf = 0.0f;
    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
        const float w = s_lut[idx] * d;
        sumf += act[ib * QK_TQ4_1S + lane] * w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);
    }

    if (lane == 0) {
        dst[sample * stride_sample_dst + (int64_t) expert_slot * stride_channel_dst + row] = sumf;
    }
}

// NVIDIA TQ4_1S dp4a MoE variant: same device-side ids routing as the scalar kernels above, but
// the int8 dp4a inner loop of mul_mat_tq4_1s_dp4a_multi (warp-strided over blocks, q8_1 activations,
// packed-centroid LUT). One dot product per (row, expert_slot, sample) output element.
template <int LPR>   // lanes cooperating on one output row (must divide WARP_SIZE)
static __global__ void mul_mat_tq4_1s_dp4a_moe(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,      // pre-rotated activations (q8_1)
        float            * __restrict__ dst,
        const int32_t    * __restrict__ ids,
        const int     ncols_x,
        const int     nrows_x,
        const int64_t nb_expert,
        const int     ids_stride,
        const int     nchannels_y,
        const int64_t stride_channel_y,             // q8_1 blocks to next y-channel
        const int64_t stride_sample_y,              // q8_1 blocks to next sample
        const int64_t stride_channel_dst,
        const int64_t stride_sample_dst) {

    // One row per LPR lanes: each lane covers blocks_per_row/LPR blocks and the cross-lane
    // reduction is log2(LPR) rounds instead of 5. With LPR == WARP_SIZE this is the classic
    // one-row-per-warp mapping.
    constexpr int rows_per_warp = WARP_SIZE / LPR;

    const int lane_in_row = threadIdx.x % LPR;
    const int row_in_warp = threadIdx.x / LPR;

    const int row = (blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y) * rows_per_warp + row_in_warp;
    if (row >= nrows_x) return;

    const int expert_slot = blockIdx.y;
    const int sample      = blockIdx.z;
    const int expert      = ids[sample * ids_stride + expert_slot];
    const int channel_y   = expert_slot % nchannels_y;

    const int lane = lane_in_row;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row =
        (const block_tq4_1s *) ((const char *) vx + (int64_t) expert * nb_expert)
        + (int64_t) row * blocks_per_row;
    const block_q8_1 * a_base = vy_q8 + sample * stride_sample_y + (int64_t) channel_y * stride_channel_y;

    float sumf = 0.0f;
    for (int ib = lane; ib < blocks_per_row; ib += LPR) {
        const block_tq4_1s * blk = &x_row[ib];
        const float fd0 = __half2float(blk->d0);
        const float fd1 = __half2float(blk->d1);

        const uint32_t * qs32 = (const uint32_t *)(blk->qs);
        const uint32_t w0 = qs32[0], w1 = qs32[1], w2 = qs32[2], w3 = qs32[3];

        int c0_0, c1_0, c0_1, c1_1, c0_2, c1_2, c0_3, c1_3;
        tq4_cents8_reg(w0, c0_0, c1_0);
        tq4_cents8_reg(w1, c0_1, c1_1);
        tq4_cents8_reg(w2, c0_2, c1_2);
        tq4_cents8_reg(w3, c0_3, c1_3);

        const block_q8_1 * a_blk = &a_base[ib];
        const float d_act = __half2float((__half)a_blk->ds.x);
        const int * a_qs = (const int *)(a_blk->qs);

        const int s0 = ggml_cuda_dp4a(c0_0, a_qs[0], ggml_cuda_dp4a(c1_0, a_qs[1],
                       ggml_cuda_dp4a(c0_1, a_qs[2], ggml_cuda_dp4a(c1_1, a_qs[3], 0))));
        const int s1 = ggml_cuda_dp4a(c0_2, a_qs[4], ggml_cuda_dp4a(c1_2, a_qs[5],
                       ggml_cuda_dp4a(c0_3, a_qs[6], ggml_cuda_dp4a(c1_3, a_qs[7], 0))));

        sumf += d_act * (fd0 * (float)s0 + fd1 * (float)s1);
    }

    sumf *= TQ4_CENTROID_I8_RESCALE;
    #pragma unroll
    for (int offset = LPR / 2; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, LPR);
    }

    if (lane == 0) {
        dst[sample * stride_sample_dst + (int64_t) expert_slot * stride_channel_dst + row] = sumf;
    }
}

// Plain kernel-node zero fill: cudaMemsetAsync would insert a MEMSET node into CUDA graph
// capture, which the graph-update path cannot patch (kernel nodes only) — that forces a full
// re-capture every step and erases the fusion win. A kernel node updates fine.
static __global__ void tq_zero_f32(float * __restrict__ dst, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = 0.0f;
    }
}

// Stage packed copies of the expert weights + ids into pool scratch. Runs BEFORE the output is
// zeroed: the graph allocator routinely aliases the fused output onto these dying tensors, so
// the fused kernel must never read the originals after the zero.
static __global__ void tq_pack_w_ids(
        const float   * __restrict__ w_src,
        const int32_t * __restrict__ ids_src,
        float         * __restrict__ w_dst,
        int32_t       * __restrict__ ids_dst,
        const int n_used, const int n_tokens,
        const int64_t w_stride_slot, const int64_t w_stride_tok, const int64_t ids_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_used * n_tokens) return;
    const int s   = i % n_used;
    const int tok = i / n_used;
    w_dst[i]   = w_src[tok * w_stride_tok + s * w_stride_slot];
    ids_dst[i] = ids_src[tok * ids_stride + s];
}

// ============================================================================
// Fused MoE down-proj + expert-weighted sum (decode).
// Replaces MUL_MAT_ID -> MUL(weights) -> (n_used-1) x ADD with one kernel.
// Expert slots stay parallel on grid.y (a slot-serial loop loses more to the
// 8x-smaller grid than the fusion saves); each slot atomically accumulates its
// weighted dot into dst, which the host zeroes beforehand. 8 float atomics per
// output element is negligible contention.
// ============================================================================
static __global__ void mul_mat_tq4_1s_dp4a_moe_wsum(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,      // pre-rotated activations (q8_1)
        const float      * __restrict__ weights,    // normalized expert weights
        float            * __restrict__ dst,        // [nrows, 1, ntok] summed output
        const int32_t    * __restrict__ ids,
        const int     ncols_x,
        const int     nrows_x,
        const int64_t nb_expert,
        const int     ids_stride,
        const int     nchannels_y,
        const int64_t stride_channel_y,             // q8_1 blocks to next y-channel
        const int64_t stride_sample_y,              // q8_1 blocks to next sample
        const int64_t w_stride_slot,                // f32 elems between weight slots
        const int64_t w_stride_tok,                 // f32 elems between weight tokens
        const int64_t stride_sample_dst) {

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int s      = blockIdx.y;
    const int sample = blockIdx.z;
    const int lane   = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;

    const int expert    = ids[sample * ids_stride + s];
    const int channel_y = s % nchannels_y;
    const block_tq4_1s * x_row =
        (const block_tq4_1s *) ((const char *) vx + (int64_t) expert * nb_expert)
        + (int64_t) row * blocks_per_row;
    const block_q8_1 * a_base = vy_q8 + sample * stride_sample_y + (int64_t) channel_y * stride_channel_y;

    float sumf = 0.0f;
    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        const block_tq4_1s * blk = &x_row[ib];
        const float fd0 = __half2float(blk->d0);
        const float fd1 = __half2float(blk->d1);

        const uint32_t * qs32 = (const uint32_t *)(blk->qs);
        const uint32_t w0 = qs32[0], w1 = qs32[1], w2 = qs32[2], w3 = qs32[3];

        int c0_0, c1_0, c0_1, c1_1, c0_2, c1_2, c0_3, c1_3;
        tq4_cents8_reg(w0, c0_0, c1_0);
        tq4_cents8_reg(w1, c0_1, c1_1);
        tq4_cents8_reg(w2, c0_2, c1_2);
        tq4_cents8_reg(w3, c0_3, c1_3);

        const block_q8_1 * a_blk = &a_base[ib];
        const float d_act = __half2float((__half)a_blk->ds.x);
        const int * a_qs = (const int *)(a_blk->qs);

        const int s0 = ggml_cuda_dp4a(c0_0, a_qs[0], ggml_cuda_dp4a(c1_0, a_qs[1],
                       ggml_cuda_dp4a(c0_1, a_qs[2], ggml_cuda_dp4a(c1_1, a_qs[3], 0))));
        const int s1 = ggml_cuda_dp4a(c0_2, a_qs[4], ggml_cuda_dp4a(c1_2, a_qs[5],
                       ggml_cuda_dp4a(c0_3, a_qs[6], ggml_cuda_dp4a(c1_3, a_qs[7], 0))));

        sumf += d_act * (fd0 * (float)s0 + fd1 * (float)s1);
    }

    const float w = weights[sample * w_stride_tok + (int64_t) s * w_stride_slot];
    float total = w * sumf * TQ4_CENTROID_I8_RESCALE;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
    }

    if (lane == 0) {
        atomicAdd(&dst[sample * stride_sample_dst + row], total);
    }
}

// Shared activation pre-rotation (forward block WHT -> q8_1). The MoE gate and up projections
// consume the same normed activation, so without caching the rotation ran once per projection.
// Keyed by tensor identity, data pointer, size and graph-eval epoch; main stream only (a sibling
// stream could otherwise consume the buffer with no cross-stream ordering). Returns a device
// pointer valid for the rest of this graph eval.
static const block_q8_1 * tq_prerotate_q8_1_cached(ggml_backend_cuda_context & ctx,
                                                   const ggml_tensor * src1,
                                                   const float * src1_d,
                                                   int n_act_elements,
                                                   ggml_cuda_pool_alloc<block_q8_1> & fallback) {
    const int    id       = ggml_cuda_get_device();
    cudaStream_t stream   = ctx.stream();
    const int    n_blocks = n_act_elements / 32;
    const size_t bytes    = (size_t) n_blocks * sizeof(block_q8_1);

    static const bool disabled = getenv("GGML_TQ_ROTCACHE") != nullptr && atoi(getenv("GGML_TQ_ROTCACHE")) == 0;

    auto & rc = ctx.tq_rot_cache;
    const bool cacheable = !disabled && ctx.curr_stream_no == 0 && bytes <= (1u << 20);

    if (cacheable && rc.epoch == ctx.graph_epoch && rc.src1 == src1 && rc.data == src1->data &&
        rc.size == bytes && rc.dev == ctx.device) {
        return (const block_q8_1 *) rc.ptr;
    }

    block_q8_1 * dst = nullptr;
    if (cacheable) {
        if (rc.dev != ctx.device || rc.cap < bytes) {
            if (rc.ptr != nullptr && rc.dev == ctx.device) {
                ctx.pool(rc.dev).free(rc.ptr, rc.cap);
            }
            size_t actual = 0;
            rc.ptr = (char *) ctx.pool(id).alloc(bytes, &actual);
            rc.cap = actual;
            rc.dev = ctx.device;
        }
        dst      = (block_q8_1 *) rc.ptr;
        rc.src1  = src1;
        rc.data  = src1->data;
        rc.epoch = ctx.graph_epoch;
        rc.size  = bytes;
    } else {
        dst = fallback.alloc(n_blocks);
    }

    const int  wpb = 4;
    const dim3 pblock(32, wpb);
    const dim3 pgrid((n_blocks + wpb - 1) / wpb);
    tq_prerotate_q8_1<<<pgrid, pblock, 0, stream>>>(src1_d, dst, n_act_elements);
    return dst;
}

void ggml_cuda_mul_mat_id_tq_wsum(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor * src0,
                                  const ggml_tensor * src1,
                                  const ggml_tensor * ids,
                                  const ggml_tensor * weights,
                                  ggml_tensor * dst_final) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(weights->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst_final->type == GGML_TYPE_F32);

    const int ncols_x = src0->ne[0];   // hidden size
    const int nrows_x = src0->ne[1];   // output features per expert
    GGML_ASSERT(ncols_x % 32 == 0);

    const int n_used      = weights->ne[1];
    const int n_tokens    = src1->ne[2];
    const int nchannels_y = src1->ne[1];

    cudaStream_t stream = ctx.stream();
    const int id = ggml_cuda_get_device();

    const float   * src1_d = (const float *)   src1->data;
    const int32_t * ids_d  = (const int32_t *) ids->data;

    const int64_t nb_expert  = src0->nb[2];
    const int     ids_stride = ids->nb[1] / ggml_type_size(ids->type);

    // Pre-rotate activations to q8_1, contiguous [ncols_x, nchannels_y, n_tokens].
    const int n_act_elements = ncols_x * nchannels_y * n_tokens;
    const int n_blocks_total = n_act_elements / 32;
    ggml_cuda_pool_alloc<block_q8_1> q8_buf(ctx.pool(id));
    const block_q8_1 * q8_act = tq_prerotate_q8_1_cached(ctx, src1, src1_d, n_act_elements, q8_buf);

    // The graph allocator routinely hands dst_final the memory of weights/ids (both die inside
    // the fused range), and the kernel reads them after dst is zeroed. Stage packed copies in
    // pool scratch BEFORE the zero so the kernel never touches the originals.
    const int n_wi = n_used * n_tokens;
    ggml_cuda_pool_alloc<float>   w_scr(ctx.pool(id), n_wi);
    ggml_cuda_pool_alloc<int32_t> ids_scr(ctx.pool(id), n_wi);
    {
        const int64_t w_ss = weights->nb[1] / sizeof(float);
        const int64_t w_st = weights->nb[2] / sizeof(float);
        tq_pack_w_ids<<<(n_wi + 63) / 64, 64, 0, stream>>>(
            (const float *) weights->data, ids_d, w_scr.get(), ids_scr.get(),
            n_used, n_tokens, w_ss, w_st, ids_stride);
    }

    // dst accumulates via atomics: zero it after the staging copy (tiny: nrows * ntok floats).
    // Kernel node, not cudaMemsetAsync — see tq_zero_f32.
    {
        const int n_out = nrows_x * n_tokens;
        tq_zero_f32<<<(n_out + 255) / 256, 256, 0, stream>>>((float *) dst_final->data, n_out);
    }

    const int64_t stride_channel_y = ncols_x / 32;
    const int64_t stride_sample_y  = (int64_t) nchannels_y * (ncols_x / 32);
    const int64_t stride_sample_dst = dst_final->nb[2] / sizeof(float);

    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS, n_used, n_tokens);
    mul_mat_tq4_1s_dp4a_moe_wsum<<<grid, block, 0, stream>>>(
        src0->data, q8_buf.get(), w_scr.get(), (float *) dst_final->data, ids_scr.get(),
        ncols_x, nrows_x, nb_expert, /*ids_stride=*/n_used, nchannels_y,
        stride_channel_y, stride_sample_y, /*w_stride_slot=*/1, /*w_stride_tok=*/n_used, stride_sample_dst);
    CUDA_CHECK(cudaGetLastError());
}

// Small-batch (decode / light speculative) TQ MoE: single fused, graph-capturable launch.
// Requires contiguous src1 (checked by the caller) so the flat WHT pre-rotation is valid.
void ggml_cuda_mul_mat_id_tq(ggml_backend_cuda_context & ctx,
                             const ggml_tensor * src0,
                             const ggml_tensor * src1,
                             const ggml_tensor * ids,
                             ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int ncols_x = ne00;   // hidden size
    const int nrows_x = ne01;   // output features per expert
    GGML_ASSERT(ncols_x % 32 == 0);

    // MUL_MAT_ID layout: dst = [ne0 = nrows_x, ne1 = n_expert_used, ne2 = n_tokens]
    const int n_expert_used = ne1;
    const int n_tokens      = ne2;
    const int nchannels_y   = ne11;

    cudaStream_t stream = ctx.stream();
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    const float   * src1_d = (const float *)   src1->data;
    const int32_t * ids_d  = (const int32_t *) ids->data;
    float         * dst_d  = (float *)         dst->data;

    const int64_t nb_expert          = src0->nb[2];                              // bytes between experts
    const int     ids_stride         = ids->nb[1] / ggml_type_size(ids->type);
    const int64_t stride_channel_dst = dst->nb[1] / ggml_type_size(dst->type);
    const int64_t stride_sample_dst  = dst->nb[2] / ggml_type_size(dst->type);

    const int n_act_elements = ncols_x * nchannels_y * n_tokens;
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS, n_expert_used, n_tokens);

    // NVIDIA TQ4_1S: int8 dp4a path (matches the non-MoE dispatch in ggml_cuda_mul_mat_tq).
    // TQ3_1S (all vendors) + TQ4_1S on AMD: scalar float path (dp4a regresses on RDNA4; no dp4a TQ3).
    // CDNA (gfx90a) has proper v_dot4_i32_i8 throughput, unlike RDNA4 where dp4a regresses --
    // so enable the int8 dp4a decode path on CDNA too, not just NVIDIA.
    const bool use_dp4a = (!GGML_CUDA_CC_IS_AMD(cc) || GGML_CUDA_CC_IS_CDNA(cc)) && src0->type == GGML_TYPE_TQ4_1S;

    if (use_dp4a) {
        // Pre-rotate activations to q8_1, contiguous [ncols_x, nchannels_y, n_tokens].
        const int n_blocks_total = n_act_elements / 32;
        ggml_cuda_pool_alloc<block_q8_1> q8_buf(ctx.pool(id));
        const block_q8_1 * q8_act = tq_prerotate_q8_1_cached(ctx, src1, src1_d, n_act_elements, q8_buf);
        const int64_t stride_channel_y = ncols_x / 32;                           // q8_1 blocks to next y-channel
        const int64_t stride_sample_y  = (int64_t) nchannels_y * (ncols_x / 32); // q8_1 blocks to next token

        // Pick lanes-per-row so every lane gets a useful number of blocks (>= 4) while keeping
        // the reduction short. GGML_TQ_LPR overrides for experiments.
        const int blocks_per_row_h = ncols_x / 32;
        static const int lpr_env = getenv("GGML_TQ_LPR") ? atoi(getenv("GGML_TQ_LPR")) : 0;
        int lpr = lpr_env;
        if (lpr == 0) {
            // 16 measured best on CDNA: 4 reduction rounds instead of 5, while the warp still
            // reads only two contiguous weight regions (smaller LPR scatters reads across more
            // rows and loses more to coalescing than it saves on the reduction).
            // 16 measured best on CDNA: 4 reduction rounds instead of 5, while the warp still
            // reads only two contiguous weight regions. Never assign more lanes than there are
            // blocks, so no lane sits idle on narrow projections.
            lpr = 16;
            while (lpr > 1 && blocks_per_row_h < lpr) {
                lpr /= 2;
            }
        }

        const int rows_per_warp_h = WARP_SIZE / lpr;
        const int rows_per_wg     = MMVQ_TQ_NWARPS * rows_per_warp_h;
        const dim3 grid_l((unsigned) ((nrows_x + rows_per_wg - 1) / rows_per_wg),
                          (unsigned) n_expert_used, (unsigned) n_tokens);

        #define TQ_LAUNCH_MOE(L) mul_mat_tq4_1s_dp4a_moe<L><<<grid_l, block, 0, stream>>>( \
            src0->data, q8_act, dst_d, ids_d, ncols_x, nrows_x, \
            nb_expert, ids_stride, nchannels_y, stride_channel_y, stride_sample_y, \
            stride_channel_dst, stride_sample_dst)

        switch (lpr) {
            case 32: TQ_LAUNCH_MOE(32); break;
            case 16: TQ_LAUNCH_MOE(16); break;
            case  8: TQ_LAUNCH_MOE( 8); break;
            case  4: TQ_LAUNCH_MOE( 4); break;
            case  2: TQ_LAUNCH_MOE( 2); break;
            default: TQ_LAUNCH_MOE(32); break;
        }
        #undef TQ_LAUNCH_MOE
    } else {
        // Pre-rotate activations to float (WHT), contiguous [ncols_x, nchannels_y, n_tokens].
        ggml_cuda_pool_alloc<float> act_buf(ctx.pool(id), n_act_elements);
        {
            const int n_blocks = n_act_elements / 32;
            const int wpb = 4;
            const dim3 pblock(32, wpb);
            const dim3 pgrid((n_blocks + wpb - 1) / wpb);
            tq_prerotate_activation<<<pgrid, pblock, 0, stream>>>(src1_d, act_buf.get(), n_act_elements);
        }
        const int64_t stride_channel_y = ncols_x;                                // float elems to next y-channel
        const int64_t stride_sample_y  = (int64_t) nchannels_y * ncols_x;        // float elems to next token
        if (src0->type == GGML_TYPE_TQ3_1S) {
            mul_mat_tq3_1s_moe<<<grid, block, 0, stream>>>(
                src0->data, act_buf.get(), dst_d, ids_d, ncols_x, nrows_x,
                nb_expert, ids_stride, nchannels_y, stride_channel_y, stride_sample_y,
                stride_channel_dst, stride_sample_dst);
        } else {
            mul_mat_tq4_1s_scalar_moe<<<grid, block, 0, stream>>>(
                src0->data, act_buf.get(), dst_d, ids_d, ncols_x, nrows_x,
                nb_expert, ids_stride, nchannels_y, stride_channel_y, stride_sample_y,
                stride_channel_dst, stride_sample_dst);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_tq(ggml_backend_cuda_context & ctx,
                           const ggml_tensor * src0,
                           const ggml_tensor * src1,
                           ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int ncols_x   = src0->ne[0];
    const int nrows_x   = src0->ne[1];
    const int ncols_dst = src1->ne[1];
    GGML_ASSERT(ncols_x % 32 == 0);

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int n_total_elements = ncols_x * ncols_dst;
    // CDNA (gfx90a) has proper v_dot4_i32_i8 throughput, unlike RDNA4 where dp4a regresses --
    // so enable the int8 dp4a decode path on CDNA too, not just NVIDIA.
    const bool use_dp4a = (!GGML_CUDA_CC_IS_AMD(cc) || GGML_CUDA_CC_IS_CDNA(cc)) && src0->type == GGML_TYPE_TQ4_1S;

    if (use_dp4a) {
        // NVIDIA TQ4_1S: dp4a int8 path (optimized for Turing+ dp4a throughput)
        const int n_total_blocks = n_total_elements / 32;
        ggml_cuda_pool_alloc<block_q8_1> q8_1_buf(ctx.pool(id), n_total_blocks);

        // Phase 1: Pre-rotate all tokens → q8_1
        {
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_total_blocks + wpb - 1) / wpb);
            tq_prerotate_q8_1<<<grid, block, 0, stream>>>(src1_d, q8_1_buf.get(), n_total_elements);
        }

        // Phase 2: dispatch based on ncols_dst
        const int stride_col_y   = ncols_x / 32;  // q8_1 blocks per column
        const int stride_col_dst = nrows_x;

        if (ncols_dst <= 8) {
            switch (ncols_dst) {
                case 1: launch_tq4_1s_multi<1>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                case 2: launch_tq4_1s_multi<2>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                case 3: launch_tq4_1s_multi<3>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                case 4: launch_tq4_1s_multi<4>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                case 5: launch_tq4_1s_multi<5>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                case 6: launch_tq4_1s_multi<6>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                case 7: launch_tq4_1s_multi<7>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                case 8: launch_tq4_1s_multi<8>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            }
        } else {
#if defined(GGML_CUDA_USE_WMMA)
            // Large prefill: Fused Tensor Core (WMMA) path
            ggml_cuda_pool_alloc<float> act_buf(ctx.pool(id), n_total_elements);
            {
                const int n_total_blocks = n_total_elements / 32;
                const int wpb = 4;
                const dim3 block(32, wpb);
                const dim3 grid((n_total_blocks + wpb - 1) / wpb);
                tq_prerotate_activation<<<grid, block, 0, stream>>>(src1_d, act_buf.get(), n_total_elements);
            }
            launch_tq4_1s_wmma(src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, ncols_dst, ncols_x, nrows_x, stream);
#else
            for (int j = 0; j < ncols_dst; j += 8) {
                const int batch = min(8, ncols_dst - j);
                const block_q8_1 * q8_j = q8_1_buf.get() + j * stride_col_y;
                float * dst_j = dst_d + j * nrows_x;
                switch (batch) {
                    case 1: launch_tq4_1s_multi<1>(src0_d, q8_j, dst_j, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                    case 2: launch_tq4_1s_multi<2>(src0_d, q8_j, dst_j, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                    case 3: launch_tq4_1s_multi<3>(src0_d, q8_j, dst_j, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                    case 4: launch_tq4_1s_multi<4>(src0_d, q8_j, dst_j, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                    case 5: launch_tq4_1s_multi<5>(src0_d, q8_j, dst_j, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                    case 6: launch_tq4_1s_multi<6>(src0_d, q8_j, dst_j, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                    case 7: launch_tq4_1s_multi<7>(src0_d, q8_j, dst_j, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                    case 8: launch_tq4_1s_multi<8>(src0_d, q8_j, dst_j, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
                }
            }
#endif
        }
    } else {
        // Scalar float path: TQ3_1S (all vendors) + TQ4_1S on AMD (dp4a regresses on RDNA4).
        // float, not half: see the comment on tq_prerotate_activation for why.
        ggml_cuda_pool_alloc<float> act_buf(ctx.pool(id), n_total_elements);

        {
            const int n_total_blocks = n_total_elements / 32;
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_total_blocks + wpb - 1) / wpb);
            tq_prerotate_activation<<<grid, block, 0, stream>>>(src1_d, act_buf.get(), n_total_elements);
        }

        const int stride_col_y   = ncols_x;  // float elements per column
        const int stride_col_dst = nrows_x;
        const bool is_tq4 = (src0->type == GGML_TYPE_TQ4_1S);

        // Macro to dispatch to the right kernel based on quant type
        #define LAUNCH_SCALAR(N, src0_ptr, act_ptr, dst_ptr) \
            if (is_tq4) { launch_tq4_1s_scalar_multi<N>(src0_ptr, act_ptr, dst_ptr, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); } \
            else        { launch_tq3_1s_multi<N>(src0_ptr, act_ptr, dst_ptr, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); }

        if (ncols_dst <= 8) {
            switch (ncols_dst) {
                case 1: LAUNCH_SCALAR(1, src0_d, act_buf.get(), dst_d); break;
                case 2: LAUNCH_SCALAR(2, src0_d, act_buf.get(), dst_d); break;
                case 3: LAUNCH_SCALAR(3, src0_d, act_buf.get(), dst_d); break;
                case 4: LAUNCH_SCALAR(4, src0_d, act_buf.get(), dst_d); break;
                case 5: LAUNCH_SCALAR(5, src0_d, act_buf.get(), dst_d); break;
                case 6: LAUNCH_SCALAR(6, src0_d, act_buf.get(), dst_d); break;
                case 7: LAUNCH_SCALAR(7, src0_d, act_buf.get(), dst_d); break;
                case 8: LAUNCH_SCALAR(8, src0_d, act_buf.get(), dst_d); break;
            }
        } else {
#if defined(GGML_CUDA_USE_WMMA)
            // Large prefill: Fused Tensor Core (WMMA) path
            if (is_tq4) {
                launch_tq4_1s_wmma(src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, ncols_dst, ncols_x, nrows_x, stream);
            } else {
                launch_tq3_1s_wmma(src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, ncols_dst, ncols_x, nrows_x, stream);
            }
#else
            for (int j = 0; j < ncols_dst; j += 8) {
                const int batch = min(8, ncols_dst - j);
                const float * act_j = act_buf.get() + j * ncols_x;
                float * dst_j = dst_d + j * nrows_x;
                switch (batch) {
                    case 1: LAUNCH_SCALAR(1, src0_d, act_j, dst_j); break;
                    case 2: LAUNCH_SCALAR(2, src0_d, act_j, dst_j); break;
                    case 3: LAUNCH_SCALAR(3, src0_d, act_j, dst_j); break;
                    case 4: LAUNCH_SCALAR(4, src0_d, act_j, dst_j); break;
                    case 5: LAUNCH_SCALAR(5, src0_d, act_j, dst_j); break;
                    case 6: LAUNCH_SCALAR(6, src0_d, act_j, dst_j); break;
                    case 7: LAUNCH_SCALAR(7, src0_d, act_j, dst_j); break;
                    case 8: LAUNCH_SCALAR(8, src0_d, act_j, dst_j); break;
                }
            }
#endif
        }
        #undef LAUNCH_SCALAR
    }
}


// ============================================================================
// Load-time conversion: TQ4_1S → q8_0 (opt-in via GGML_TQ_CONVERT_Q8=1)
// ============================================================================

static __global__ void k_convert_tq4_1s_to_q8_0(
        const block_tq4_1s * __restrict__ src,
        block_q8_0         * __restrict__ dst,
        const int n_blocks) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (block_idx >= n_blocks) return;
    const int lane = threadIdx.x;
    const block_tq4_1s * blk = &src[block_idx];

    const float d_scale = (lane < 16) ? __half2float(blk->d0) : __half2float(blk->d1);
    const uint8_t idx = (blk->qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
    float val = TQ4_CENTROIDS_WEIGHT[idx] * d_scale;

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    val *= TQ_WEIGHT_SIGNS[lane];

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) dst[block_idx].d = __float2half(d);
}

void ggml_cuda_convert_tq4_1s_to_q8_0(const void * src_tq4, void * dst_q8, int64_t n_elements, cudaStream_t stream) {
    GGML_ASSERT(n_elements % QK_TQ4_1S == 0);
    const int n_blocks = n_elements / QK_TQ4_1S;
    const int wpb = 4;
    const dim3 block(32, wpb);
    const dim3 grid((n_blocks + wpb - 1) / wpb);
    k_convert_tq4_1s_to_q8_0<<<grid, block, 0, stream>>>(
        (const block_tq4_1s *)src_tq4, (block_q8_0 *)dst_q8, n_blocks);
}

// ============================================================================
// Large prefill: runtime TQ4_1S → q8_0 scratch + q8_0→fp16 dequant + cuBLAS
// Gets tensor core throughput without permanent 1.7× VRAM cost.
// ============================================================================

void ggml_cuda_mul_mat_tq4_1s_cublas(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor * src0,
                                      const ggml_tensor * src1,
                                      ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];  // K (hidden dim)
    const int64_t ne01 = src0->ne[1];  // M (rows = output features)
    const int64_t ne10 = src1->ne[0];  // K
    const int64_t ne11 = src1->ne[1];  // N (tokens)
    GGML_ASSERT(ne00 == ne10);

    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();

    const int64_t n_elements = ne00 * ne01;

    // Step 1: TQ4_1S → fp16 via warp-cooperative dequant (WHT in-warp)
    ggml_cuda_pool_alloc<half> src0_f16(ctx.pool(id), n_elements);
    {
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_TQ4_1S);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16((const char *)src0->data, src0_f16.get(), n_elements, stream);
    }

    // Step 2: src1 f32 → fp16
    ggml_cuda_pool_alloc<half> src1_f16(ctx.pool(id), ne10 * ne11);
    {
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16((const char *)src1->data, src1_f16.get(), ne10 * ne11, stream);
    }

    // Step 3: cuBLAS fp16 GEMM with fp32 compute (tensor cores)
    // dst[M×N] = src0[M×K]^T × src1[K×N]
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const int64_t ldc = dst->ne[0];  // M

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
    CUBLAS_CHECK(
        cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne00,
                &alpha, src0_f16.get(), CUDA_R_16F, ne00,
                        src1_f16.get(), CUDA_R_16F, ne10,
                &beta,  (float *)dst->data, CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Phase 2: native MFMA-i8 MMQ prefill for TQ4_1S (gfx90a). The block-local turbo WHT cancels on
// the activation side, so we pre-rotate src1 (forward WHT via tq_prerotate_activation) into a
// pooled f32 scratch and run the stock MMQ with the TQ4_1S int8-centroid load_tiles (the weight
// stays as rotated-domain centroids). Env-gated by GGML_TQ_MMQ. src1/dst assumed contiguous f32
// with ne10 % QK_TQ4_1S == 0 (checked by the caller's tq_fast_path_ok).
void ggml_cuda_mul_mat_tq4_1s_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t n_act    = ggml_nelements(src1);   // contiguous f32
    const int64_t n_blocks = n_act / 32;             // 32-elem WHT blocks along ne0

    ggml_cuda_pool_alloc<float> act_buf(ctx.pool(), n_act);

    const dim3 block(32, MMVQ_TQ_NWARPS);
    const dim3 grid((n_blocks + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    tq_prerotate_activation<<<grid, block, 0, stream>>>((const float *) src1->data, act_buf.get(), (int) n_act);

    // Shallow tensor over the rotated activation (same shape/strides, still contiguous f32).
    ggml_tensor src1_rot = *src1;
    src1_rot.data = act_buf.get();

    ggml_cuda_mul_mat_q(ctx, src0, &src1_rot, nullptr, dst, false);
}

// Phase 2 (MoE): native MFMA-i8 MMQ prefill for TQ4_1S experts (gfx90a). Same trick as the dense
// wrapper above -- the block-local turbo WHT is per-hidden-vector and identical across experts, so
// we pre-rotate the whole activation once, then let the stock MMQ_ID (ggml_cuda_mul_mat_q with ids)
// gather per-expert against the rotated-domain TQ4_1S centroids. This replaces the 2x-slower
// dequant-to-f16 cuBLAS fallback for native (GGML_TQ_NATIVE) MoE, so experts can stay at native
// 5bpw (avoiding the ~1.7x-VRAM Q8_0 load-time conversion) without the prefill penalty. Env-gated
// by GGML_TQ_MMQ. src1 assumed contiguous f32 with ne10 % QK_TQ4_1S == 0.
void ggml_cuda_mul_mat_id_tq4_1s_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                     const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t n_act    = ggml_nelements(src1);   // contiguous f32
    const int64_t n_blocks = n_act / 32;             // 32-elem WHT blocks along ne0

    ggml_cuda_pool_alloc<float> act_buf(ctx.pool(), n_act);

    const dim3 block(32, MMVQ_TQ_NWARPS);
    const dim3 grid((n_blocks + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    tq_prerotate_activation<<<grid, block, 0, stream>>>((const float *) src1->data, act_buf.get(), (int) n_act);

    // Shallow tensor over the rotated activation (same shape/strides, still contiguous f32).
    ggml_tensor src1_rot = *src1;
    src1_rot.data = act_buf.get();

    ggml_cuda_mul_mat_q(ctx, src0, &src1_rot, ids, dst, false);
}
