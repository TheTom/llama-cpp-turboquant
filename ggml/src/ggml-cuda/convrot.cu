#include "common.cuh"
#include "convrot.cuh"
#include "mmq.cuh"

static __device__ __forceinline__ void convrot_transform_group(
        const float * src,
        float * rd,
        float * wr,
        const bool active,
        const int lane) {
    // normalized Kronecker-Hadamard base matrix, no all-ones column
    constexpr float h4_cr[4][4] = {
        { 1.0f,  1.0f,  1.0f, -1.0f},
        { 1.0f,  1.0f, -1.0f,  1.0f},
        { 1.0f, -1.0f,  1.0f,  1.0f},
        {-1.0f,  1.0f,  1.0f,  1.0f},
    };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int j = 0; j < QK8_CR/32; ++j) {
        rd[lane + 32*j] = active ? src[lane + 32*j] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int len = 4, shift = 0; len <= QK8_CR; len *= 4, shift += 2) {
        const int half = len >> 2;
#pragma unroll
        for (int k = 0; k < QK8_CR/32; ++k) {
            const int i   = lane + 32*k;
            const int blk = i & ~(len - 1);
            const int r   = (i >> shift) & 3;
            const int t   = i & (half - 1);

            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc += h4_cr[r][j] * rd[blk + j*half + t];
            }
            wr[i] = acc;
        }
        __syncthreads();
        float * tmp = rd; rd = wr; wr = tmp;
    }
}

// One warp transforms one QK8_CR group with the radix-4 butterfly from ggml-quants.c.
template <int warps_per_block>
__global__ void convrot_rotate_cuda(
        const float * src_ptr,
        float * dst_ptr,
        const int64_t n_groups) {
    const float * GGML_CUDA_RESTRICT src = src_ptr;
    float * GGML_CUDA_RESTRICT dst = dst_ptr;

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int64_t g  = (int64_t) blockIdx.x * warps_per_block + warp;

    const bool active = g < n_groups;
    if (active) {
        src += g * QK8_CR;
        dst += g * QK8_CR;
    }

    __shared__ float buf[warps_per_block][2][QK8_CR];

    float * rd = buf[warp][0];
    float * wr = buf[warp][1];

    convrot_transform_group(src, rd, wr, active, lane);

    // the product of the four stages has unit norm, scale once at the end
    constexpr float scale = 1.0f / 16.0f; // 1 / sqrt(QK8_CR)
    if (active) {
#pragma unroll
        for (int j = 0; j < QK8_CR/32; ++j) {
            dst[lane + 32*j] = rd[lane + 32*j] * scale;
        }
    }
}

// One 256-thread CTA transforms and quantizes one ConvRot group.
template <bool mmq_layout>
__global__ void convrot_quantize_q8_1_cuda(
        const float * src_ptr,
        void * dst_ptr,
        const int64_t ne00,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2) {
    const float * GGML_CUDA_RESTRICT src = src_ptr;
    void * GGML_CUDA_RESTRICT dst = dst_ptr;

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int64_t flat_group = blockIdx.x;
    const int64_t groups_per_row = ne0/QK8_CR;
    const int64_t row = flat_group/groups_per_row;
    const int64_t group = flat_group - row*groups_per_row;
    const int64_t i1 = row % ne1;
    const int64_t i23 = row/ne1;
    const int64_t i2 = i23 % ne2;
    const int64_t i3 = i23/ne2;
    const bool active_input = group*QK8_CR < ne00;

    if (active_input) {
        src += i1*s01 + i2*s02 + i3*s03 + group*QK8_CR;
    }

    ggml_cuda_pdl_sync();
    float x = active_input ? src[tid] : 0.0f;

    constexpr float h4_cr[4][4] = {
        { 1.0f,  1.0f,  1.0f, -1.0f},
        { 1.0f,  1.0f, -1.0f,  1.0f},
        { 1.0f, -1.0f,  1.0f,  1.0f},
        {-1.0f,  1.0f,  1.0f,  1.0f},
    };

    // The first two stages use warp shuffles.
#pragma unroll
    for (int len = 4, shift = 0; len <= 16; len *= 4, shift += 2) {
        const int half = len >> 2;
        const int blk = lane & ~(len - 1);
        const int r = (lane >> shift) & 3;
        const int t = lane & (half - 1);

        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += h4_cr[r][j]*__shfl_sync(0xffffffff, x, blk + j*half + t, WARP_SIZE);
        }
        x = acc;
    }

    // The last two stages use shared memory.
    __shared__ float buf[2][QK8_CR];
    buf[0][tid] = x;
    __syncthreads();

    const int r64 = (tid >> 4) & 3;
    const int t64 = tid & 15;
    const int blk64 = tid & ~63;
    float acc64 = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc64 += h4_cr[r64][j]*buf[0][blk64 + 16*j + t64];
    }
    buf[1][tid] = acc64;
    __syncthreads();

    const int r256 = (tid >> 6) & 3;
    const int t256 = tid & 63;
    float acc256 = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc256 += h4_cr[r256][j]*buf[1][64*j + t256];
    }

    constexpr float convrot_scale = 1.0f/16.0f;
    constexpr int q8_blocks_per_group = QK8_CR/QK8_1;
    const float xi = acc256*convrot_scale;
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float d = amax/127.0f;
    const int8_t q = amax == 0.0f ? 0 : int8_t(roundf(xi/d));

    if constexpr (mmq_layout) {
        const int64_t k_block = group*2 + warp/4;
        const int64_t ib = i23*(ne1*(ne0/QK8_1_MMQ)) + k_block*ne1 + i1;
        block_q8_1_mmq * y = (block_q8_1_mmq *) dst + ib;
        y->qs[(warp % 4)*QK8_1 + lane] = q;
        if (lane == 0) {
            y->d4[warp % 4] = d;
        }
    } else {
        const int64_t ib = row*(ne0/QK8_1) + group*q8_blocks_per_group + warp;
        block_q8_1 * y = (block_q8_1 *) dst + ib;
        y->qs[lane] = q;
        const float sum = warp_reduce_sum<QK8_1>((float) q);
        if (lane == 0) {
            y->ds = make_half2(d, d*sum);
        }
    }
}

// Register-only prompt producer; each warp owns one ConvRot group.
template <int warps_per_block, bool mmq_layout>
__global__ void convrot_quantize_q8_1_warp_cuda(
        const float * src_ptr,
        void * dst_ptr,
        const int64_t ne00,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t n_groups) {
    const float * GGML_CUDA_RESTRICT src = src_ptr;
    void * GGML_CUDA_RESTRICT dst = dst_ptr;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int64_t flat_group = int64_t(blockIdx.x)*warps_per_block + warp;
    const int64_t groups_per_row = ne0/QK8_CR;
    const bool active_group = flat_group < n_groups;
    const int64_t row = active_group ? flat_group/groups_per_row : 0;
    const int64_t group = active_group ? flat_group - row*groups_per_row : 0;
    const int64_t i1 = row % ne1;
    const int64_t i23 = row/ne1;
    const int64_t i2 = i23 % ne2;
    const int64_t i3 = i23/ne2;
    const bool active_input = active_group && group*QK8_CR < ne00;

    if (active_input) {
        src += i1*s01 + i2*s02 + i3*s03 + group*QK8_CR;
    }

    constexpr float h4_cr[4][4] = {
        { 1.0f,  1.0f,  1.0f, -1.0f},
        { 1.0f,  1.0f, -1.0f,  1.0f},
        { 1.0f, -1.0f,  1.0f,  1.0f},
        {-1.0f,  1.0f,  1.0f,  1.0f},
    };

    float reg[QK8_CR/WARP_SIZE];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int j = 0; j < QK8_CR/WARP_SIZE; ++j) {
        reg[j] = active_input ? src[j*WARP_SIZE + lane] : 0.0f;
    }

    // Radix-4 stages with lengths 4 and 16 stay within each register's warp.
#pragma unroll
    for (int len = 4, shift = 0; len <= 16; len *= 4, shift += 2) {
        const int half = len >> 2;
        const int blk = lane & ~(len - 1);
        const int r = (lane >> shift) & 3;
        const int t = lane & (half - 1);
#pragma unroll
        for (int k = 0; k < QK8_CR/WARP_SIZE; ++k) {
            const float value = reg[k];
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc += h4_cr[r][j]*__shfl_sync(0xffffffff, value, blk + j*half + t, WARP_SIZE);
            }
            reg[k] = acc;
        }
    }

    // Length 64 pairs adjacent registers and exchanges their 16-value halves.
#pragma unroll
    for (int k = 0; k < QK8_CR/WARP_SIZE; k += 2) {
        const float even = reg[k + 0];
        const float odd  = reg[k + 1];
        const int t = lane & 15;
        const float v0 = __shfl_sync(0xffffffff, even, t,      WARP_SIZE);
        const float v1 = __shfl_sync(0xffffffff, even, t + 16, WARP_SIZE);
        const float v2 = __shfl_sync(0xffffffff, odd,  t,      WARP_SIZE);
        const float v3 = __shfl_sync(0xffffffff, odd,  t + 16, WARP_SIZE);
        const float v[4] = {v0, v1, v2, v3};
        const int r_even = lane >> 4;
        const int r_odd = r_even + 2;
        float acc_even = 0.0f;
        float acc_odd = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc_even += h4_cr[r_even][j]*v[j];
            acc_odd  += h4_cr[r_odd ][j]*v[j];
        }
        reg[k + 0] = acc_even;
        reg[k + 1] = acc_odd;
    }

    // Length 256 combines the four 64-value register pairs directly.
    float prev[QK8_CR/WARP_SIZE];
#pragma unroll
    for (int k = 0; k < QK8_CR/WARP_SIZE; ++k) {
        prev[k] = reg[k];
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int parity = 0; parity < 2; ++parity) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc += h4_cr[r][j]*prev[2*j + parity];
            }
            reg[2*r + parity] = acc;
        }
    }

    if (!active_group) {
        return;
    }

    constexpr float convrot_scale = 1.0f/16.0f;
    constexpr int q8_blocks_per_group = QK8_CR/QK8_1;
#pragma unroll
    for (int j = 0; j < q8_blocks_per_group; ++j) {
        const float xi = reg[j]*convrot_scale;
        const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
        const float d = amax/127.0f;
        const int8_t q = amax == 0.0f ? 0 : int8_t(roundf(xi/d));

        if constexpr (mmq_layout) {
            const int64_t k_block = group*2 + j/4;
            const int64_t ib = i23*(ne1*(ne0/QK8_1_MMQ)) + k_block*ne1 + i1;
            block_q8_1_mmq * y = (block_q8_1_mmq *) dst + ib;
            y->qs[(j % 4)*QK8_1 + lane] = q;
            if (lane == 0) {
                y->d4[j % 4] = d;
            }
        } else {
            const int64_t ib = row*(ne0/QK8_1) + group*q8_blocks_per_group + j;
            block_q8_1 * y = (block_q8_1 *) dst + ib;
            y->qs[lane] = q;
            const float sum = warp_reduce_sum<QK8_1>((float) q);
            if (lane == 0) {
                y->ds = make_half2(d, d*sum);
            }
        }
    }
}

template <bool mmq_layout>
static void launch_convrot_quantize_q8_1(
        const float * src, void * dst,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK8_CR == 0);
    GGML_ASSERT(ne0 % QK8_CR == 0);

    const int64_t n_groups = ne1*ne2*ne3*(ne0/QK8_CR);
    if (n_groups == 0) {
        return;
    }
    constexpr int warp_kernel_min_groups = 1024;
    if (n_groups >= warp_kernel_min_groups) {
        constexpr int warps_per_block = 4;
        const dim3 grid_dims((n_groups + warps_per_block - 1)/warps_per_block, 1, 1);
        const dim3 block_dims(WARP_SIZE, warps_per_block, 1);
        const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);
        ggml_cuda_kernel_launch(convrot_quantize_q8_1_warp_cuda<warps_per_block, mmq_layout>, launch_params,
            src, dst, ne00, s01, s02, s03, ne0, ne1, ne2, n_groups);
    } else {
        const dim3 grid_dims(n_groups, 1, 1);
        const dim3 block_dims(QK8_CR, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);
        ggml_cuda_kernel_launch(convrot_quantize_q8_1_cuda<mmq_layout>, launch_params,
            src, dst, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void ggml_cuda_convrot_rotate(ggml_backend_cuda_context & ctx, const ggml_tensor * src, float * dst) {
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src));
    GGML_ASSERT(src->ne[0] % QK8_CR == 0);

    const int64_t n_groups = ggml_nrows(src) * (src->ne[0] / QK8_CR);
    if (n_groups == 0) {
        return;
    }

    constexpr int warps_per_block = 4;
    const int64_t num_blocks = (n_groups + warps_per_block - 1) / warps_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(32, warps_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    ggml_cuda_kernel_launch(convrot_rotate_cuda<warps_per_block>, launch_params,
        (const float *) src->data, dst, n_groups);
}

void ggml_cuda_convrot_quantize_q8_1(
        const float * src, void * dst,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        cudaStream_t stream) {
    launch_convrot_quantize_q8_1<false>(
        src, dst, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
}

void ggml_cuda_convrot_quantize_mmq_q8_1(
        const float * src, void * dst,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        cudaStream_t stream) {
    launch_convrot_quantize_q8_1<true>(
        src, dst, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
}
