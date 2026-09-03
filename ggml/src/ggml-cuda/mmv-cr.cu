#include "mmv-cr.cuh"

#include "dequantize.cuh"

#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static_assert(QK8_CR == 256, "cooperative CR kernel requires 256-value groups");
static_assert(QK_K == QK8_CR, "Q6_CR must map one block_q6_K to one CR group");
static_assert(sizeof(block_q6_cr) == sizeof(block_q6_K), "Q6_CR physical layout changed");

template <ggml_type type>
static __device__ __forceinline__ float dequantize_cr_value(
        const void * weights, int64_t row, int64_t groups_per_row,
        int64_t group_index, int index);

template <>
__device__ __forceinline__ float dequantize_cr_value<GGML_TYPE_Q8_CR>(
        const void * weights, int64_t row, int64_t groups_per_row,
        int64_t group_index, int index) {
    const block_q8_cr & group =
        static_cast<const block_q8_cr *>(weights)[row*groups_per_row + group_index];
    const block_q8_0 & block = group.blocks[index/QK8_0];
    return float(block.qs[index % QK8_0]) * __half2float(block.d);
}

template <>
__device__ __forceinline__ float dequantize_cr_value<GGML_TYPE_Q5_CR>(
        const void * weights, int64_t row, int64_t groups_per_row,
        int64_t group_index, int index) {
    const block_q5_cr & group =
        static_cast<const block_q5_cr *>(weights)[row*groups_per_row + group_index];
    const block_q5_0 & block = group.blocks[index/QK5_0];
    const int local = index % QK5_0;
    uint32_t qh;
    memcpy(&qh, block.qh, sizeof(qh));
    const int low = local < QK5_0/2
        ? block.qs[local] & 0x0f
        : block.qs[local - QK5_0/2] >> 4;
    const int quant = low | (((qh >> local) & 1) << 4);
    return (float(quant) - 16.0f) * __half2float(block.d);
}

template <>
__device__ __forceinline__ float dequantize_cr_value<GGML_TYPE_Q6_CR>(
        const void * weights, int64_t row, int64_t groups_per_row,
        int64_t group_index, int index) {
    const block_q6_cr & block =
        static_cast<const block_q6_cr *>(weights)[row*groups_per_row + group_index];
    const int ip = index/128;
    const int within = index % 128;
    const int segment = within/32;
    const int il = within % 32;
    const int is = 8*ip + il/16;
    const uint8_t * ql = block.ql + 64*ip + il;
    const uint8_t qh = block.qh[32*ip + il];
    const int ql_index = (segment & 1) == 0 ? 0 : 32;
    const int ql_shift = segment < 2 ? 0 : 4;
    const int quant = ((ql[ql_index] >> ql_shift) & 0x0f) |
        (((qh >> (2*segment)) & 0x03) << 4);
    return __half2float(block.d) * block.scales[is + 2*segment] * (quant - 32);
}

struct cr_registers {
    float value[4];
};

static __device__ __forceinline__ float cr_radix4(
        float a, float b, float c, float d, int row) {
    return row == 0 ?  a + b + c - d :
           row == 1 ?  a + b - c + d :
           row == 2 ?  a - b + c + d :
                       -a + b + c + d;
}

template <int rows_per_cta>
static __device__ __forceinline__ void convrot_inverse_registers(
        cr_registers & values,
        float (&exchange)[rows_per_cta][QK8_CR],
        int row_in_cta,
        int p) {
    const int lane = p & 31;

#pragma unroll
    for (int len = 4; len <= 16; len *= 4) {
        const int half = len/4;
        const int base = lane & ~(len - 1);
        const int radix_row = (lane/half) & 3;
        const int col = lane & (half - 1);
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            const float v = values.value[component];
            const float a = __shfl_sync(0xffffffff, v, base + col,          WARP_SIZE);
            const float b = __shfl_sync(0xffffffff, v, base + half + col,   WARP_SIZE);
            const float c = __shfl_sync(0xffffffff, v, base + 2*half + col, WARP_SIZE);
            const float d = __shfl_sync(0xffffffff, v, base + 3*half + col, WARP_SIZE);
            values.value[component] = cr_radix4(a, b, c, d, radix_row);
        }
    }

#pragma unroll
    for (int component = 0; component < 4; ++component) {
        exchange[row_in_cta][component*64 + p] = values.value[component];
    }
    __syncthreads();

    const int radix_row64 = p/16;
    const int col64 = p & 15;
#pragma unroll
    for (int component = 0; component < 4; ++component) {
        const int base = component*64;
        values.value[component] = cr_radix4(
            exchange[row_in_cta][base + col64],
            exchange[row_in_cta][base + 16 + col64],
            exchange[row_in_cta][base + 32 + col64],
            exchange[row_in_cta][base + 48 + col64],
            radix_row64);
    }
    __syncthreads();

    const float a = values.value[0];
    const float b = values.value[1];
    const float c = values.value[2];
    const float d = values.value[3];
    values.value[0] = cr_radix4(a, b, c, d, 0) * (1.0f/16.0f);
    values.value[1] = cr_radix4(a, b, c, d, 1) * (1.0f/16.0f);
    values.value[2] = cr_radix4(a, b, c, d, 2) * (1.0f/16.0f);
    values.value[3] = cr_radix4(a, b, c, d, 3) * (1.0f/16.0f);
}

static __device__ __forceinline__ float convrot_inverse_group(
        float buf[2][QK8_CR], const int tid) {
    __syncthreads();
    float v = buf[0][tid];
    const int lane = tid & 31;

#pragma unroll
    for (int len = 4; len <= 16; len *= 4) {
        const int half = len/4;
        const int base = lane & ~(len - 1);
        const int r = (lane/half) & 3;
        const int col = lane & (half - 1);
        const float a = __shfl_sync(0xffffffff, v, base + col,          WARP_SIZE);
        const float b = __shfl_sync(0xffffffff, v, base + half + col,   WARP_SIZE);
        const float c = __shfl_sync(0xffffffff, v, base + 2*half + col, WARP_SIZE);
        const float d = __shfl_sync(0xffffffff, v, base + 3*half + col, WARP_SIZE);
        v = r == 0 ?  a + b + c - d :
            r == 1 ?  a + b - c + d :
            r == 2 ?  a - b + c + d :
                      -a + b + c + d;
    }

    buf[0][tid] = v;
    __syncthreads();

    constexpr int half64 = 16;
    const int base64 = tid & ~63;
    const int r64 = (tid/half64) & 3;
    const int col64 = tid & (half64 - 1);
    const float a64 = buf[0][base64 + col64];
    const float b64 = buf[0][base64 + half64 + col64];
    const float c64 = buf[0][base64 + 2*half64 + col64];
    const float d64 = buf[0][base64 + 3*half64 + col64];
    buf[1][tid] = r64 == 0 ?  a64 + b64 + c64 - d64 :
                  r64 == 1 ?  a64 + b64 - c64 + d64 :
                  r64 == 2 ?  a64 - b64 + c64 + d64 :
                              -a64 + b64 + c64 + d64;
    __syncthreads();

    constexpr int half256 = 64;
    const int r256 = tid/half256;
    const int col256 = tid & (half256 - 1);
    const float a256 = buf[1][col256];
    const float b256 = buf[1][half256 + col256];
    const float c256 = buf[1][2*half256 + col256];
    const float d256 = buf[1][3*half256 + col256];
    return (r256 == 0 ?  a256 + b256 + c256 - d256 :
            r256 == 1 ?  a256 + b256 - c256 + d256 :
            r256 == 2 ?  a256 - b256 + c256 + d256 :
                        -a256 + b256 + c256 + d256) * (1.0f/16.0f);
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void mul_mat_vec_cr_reference(
        const void * __restrict__ vx,
        const float * __restrict__ x,
        float * __restrict__ y,
        const int64_t k) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t blocks_per_row = k/QK8_CR;
    float acc = 0.0f;

    __shared__ float buf[2][QK8_CR];
    for (int64_t block = 0; block < blocks_per_row; ++block) {
        if (tid < QK8_CR/2) {
            const int i00 = 2*tid;
            const int64_t ib = (row*blocks_per_row + block)*(QK8_CR/qk) + i00/qk;
            const int iqs = (i00%qk)/qr;
            const int iybs = i00 - i00%qk;
            const int y_offset = qr == 1 ? 1 : qk/2;
            float2 values;
            dequantize_kernel(vx, ib, iqs, values);
            buf[0][iybs + iqs] = values.x;
            buf[0][iybs + iqs + y_offset] = values.y;
        }
        const float weight = convrot_inverse_group(buf, tid);
        acc += weight * x[block*QK8_CR + tid];
        __syncthreads();
    }

    buf[0][tid] = acc;
    __syncthreads();
    for (int stride = QK8_CR/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            buf[0][tid] += buf[0][tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = buf[0][0];
    }
}

static __global__ void mul_mat_vec_q6_cr_reference(
        const void * __restrict__ vx,
        const float * __restrict__ x,
        float * __restrict__ y,
        const int64_t k) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t blocks_per_row = k/QK8_CR;
    float acc = 0.0f;

    __shared__ float buf[2][QK8_CR];
    for (int64_t block = 0; block < blocks_per_row; ++block) {
        if (tid < 64) {
            dequantize_q6_K(vx, row*blocks_per_row + block, buf[0], tid);
        }
        const float weight = convrot_inverse_group(buf, tid);
        acc += weight * x[block*QK8_CR + tid];
        __syncthreads();
    }

    buf[0][tid] = acc;
    __syncthreads();
    for (int stride = QK8_CR/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            buf[0][tid] += buf[0][tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = buf[0][0];
    }
}

static void launch_reference(
        ggml_type type,
        const void * weights,
        const float * x,
        float * y,
        int64_t m,
        int64_t k,
        cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q5_CR:
            mul_mat_vec_cr_reference<QK5_0, QR5_0, dequantize_q5_0>
                <<<m, QK8_CR, 0, stream>>>(weights, x, y, k);
            break;
        case GGML_TYPE_Q6_CR:
            mul_mat_vec_q6_cr_reference<<<m, QK8_CR, 0, stream>>>(weights, x, y, k);
            break;
        case GGML_TYPE_Q8_CR:
            mul_mat_vec_cr_reference<QK8_0, QR8_0, dequantize_q8_0>
                <<<m, QK8_CR, 0, stream>>>(weights, x, y, k);
            break;
        default:
            GGML_ABORT("unsupported CR type");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <ggml_type type, int rows_per_cta>
static __global__ void mul_mat_vec_cr_cooperative(
        const void * weights_ptr,
        const float * x_ptr,
        float * y_ptr,
        int64_t m,
        int64_t n,
        int64_t k) {
    const void * GGML_CUDA_RESTRICT weights = weights_ptr;
    const float * GGML_CUDA_RESTRICT x = x_ptr;
    float * GGML_CUDA_RESTRICT y = y_ptr;
    static_assert(rows_per_cta == 2 || rows_per_cta == 4, "unsupported CR CTA geometry");

    const int p = threadIdx.x;
    const int row_in_cta = threadIdx.y;
    const int linear_tid = row_in_cta*64 + p;
    const int64_t row = int64_t(blockIdx.x)*rows_per_cta + row_in_cta;
    const int64_t token = blockIdx.y;
    const bool active_row = row < m && token < n;
    const int64_t groups_per_row = k/QK8_CR;

    __shared__ float activation[QK8_CR];
    __shared__ float exchange[rows_per_cta][QK8_CR];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    ggml_cuda_pdl_sync();
    for (int64_t group = 0; group < groups_per_row; ++group) {
        for (int i = linear_tid; i < QK8_CR; i += rows_per_cta*64) {
            activation[i] = x[token*k + group*QK8_CR + i];
        }
        __syncthreads();

        cr_registers values = {};
        if (active_row) {
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                values.value[component] = dequantize_cr_value<type>(
                    weights, row, groups_per_row, group, p + 64*component);
            }
        }

        convrot_inverse_registers(values, exchange, row_in_cta, p);

#pragma unroll
        for (int component = 0; component < 4; ++component) {
            acc[component] += values.value[component] * activation[p + 64*component];
        }
        __syncthreads();
    }

    const float sum02 = acc[0] + acc[2];
    const float sum13 = acc[1] + acc[3];
    exchange[row_in_cta][p] = sum02 + sum13;
    __syncthreads();

    float row_sum = exchange[row_in_cta][p];
    if (p < 32) {
        row_sum += exchange[row_in_cta][p + 32];
#pragma unroll
        for (int stride = 16; stride > 0; stride /= 2) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, stride, WARP_SIZE);
        }
        if (p == 0 && active_row) {
            y[token*m + row] = row_sum;
        }
    }
}

template <ggml_type type, int rows_per_cta>
static void launch_cooperative(
        const void * weights, const float * x, float * y,
        int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
    const dim3 grid((m + rows_per_cta - 1)/rows_per_cta, n, 1);
    const dim3 block(64, rows_per_cta, 1);
    const ggml_cuda_kernel_launch_params params(grid, block, 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_cr_cooperative<type, rows_per_cta>, params,
        weights, x, y, m, n, k);
}

template <int rows_per_cta>
static void launch_cooperative_type(
        ggml_type type, const void * weights, const float * x, float * y,
        int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q5_CR:
            launch_cooperative<GGML_TYPE_Q5_CR, rows_per_cta>(weights, x, y, m, n, k, stream);
            break;
        case GGML_TYPE_Q6_CR:
            launch_cooperative<GGML_TYPE_Q6_CR, rows_per_cta>(weights, x, y, m, n, k, stream);
            break;
        case GGML_TYPE_Q8_CR:
            launch_cooperative<GGML_TYPE_Q8_CR, rows_per_cta>(weights, x, y, m, n, k, stream);
            break;
        default:
            GGML_ABORT("unsupported CR type");
    }
}

static void launch_cooperative_tuned(
        ggml_type type, const void * weights, const float * x, float * y,
        int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q5_CR:
            launch_cooperative<GGML_TYPE_Q5_CR, 4>(weights, x, y, m, n, k, stream);
            break;
        case GGML_TYPE_Q6_CR:
            // Four rows increase Q6_CR register pressure; the other CR formats benefit from activation reuse.
            launch_cooperative<GGML_TYPE_Q6_CR, 2>(weights, x, y, m, n, k, stream);
            break;
        case GGML_TYPE_Q8_CR:
            launch_cooperative<GGML_TYPE_Q8_CR, 4>(weights, x, y, m, n, k, stream);
            break;
        default:
            GGML_ABORT("unsupported CR type");
    }
}

bool ggml_cuda_mul_mat_vec_cr(
        ggml_type type,
        const void * weights,
        const float * x,
        float * y,
        int64_t m,
        int64_t n,
        int64_t k,
        int cc,
        cudaStream_t stream) {
    // Each token is an independent grid-Y slice; support the full range for tiny output matrices.
    if (k % QK8_CR != 0 || n < 1 || n > 65535) {
        return false;
    }
    if (m == 0 || k == 0) {
        return true;
    }
    if (cc <= GGML_CUDA_CC_PASCAL) {
        if (n != 1) {
            return false;
        }
        launch_reference(type, weights, x, y, m, k, stream);
        return true;
    }
    if (m <= 2) {
        launch_cooperative_type<2>(type, weights, x, y, m, n, k, stream);
    } else {
        launch_cooperative_tuned(type, weights, x, y, m, n, k, stream);
    }
    return true;
}

#endif
