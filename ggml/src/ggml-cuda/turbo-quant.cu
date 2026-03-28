/*
 * TurboQuant CUDA kernels — dequantize with inverse WHT
 * Based on animehacker/llama-turboquant reference implementation.
 * Per-block WHT32 rotation: no graph-level ops needed.
 */

#include "turbo-quant.cuh"
#include "convert.cuh"

#include <cstdint>

void turbo_quant_init_cuda(cudaStream_t stream) {
    GGML_UNUSED(stream);
}

/* ===== turbo3 dequantize: centroid lookup + cooperative inverse WHT32 ===== */

template<typename dst_t>
static __global__ void dequantize_block_turbo3_0(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const float centroids[8] = {
        -2.1573f, -1.3336f, -0.7434f, -0.2428f,
         0.2428f,  0.7434f,  1.3336f,  2.1573f
    };
    const int8_t signs[32] = {
        +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
        +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1
    };

    const int64_t i = blockIdx.x;
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const float d = __half2float(x[i].gamma);

    // Step 1: Each thread dequantizes its value (3-bit index from qs + qr)
    const int low2 = (x[i].qs[tid / 4] >> (2 * (tid % 4))) & 3;
    const int hi1  = (x[i].qr[tid / 8] >> (tid % 8)) & 1;
    const int idx  = low2 | (hi1 << 2);

    __shared__ float shmem[32];
    shmem[tid] = d * centroids[idx];
    __syncthreads();

    // Step 2: Cooperative inverse WHT (5 butterfly stages)
    for (int step = 1; step < 32; step <<= 1) {
        int partner = tid ^ step;
        float a = shmem[tid];
        float b = shmem[partner];
        __syncthreads();
        if (tid < partner) {
            shmem[tid]     = a + b;
            shmem[partner] = a - b;
        }
        __syncthreads();
    }

    // Step 3: Normalize and undo sign flips
    const float inv_sqrt32 = 0.17677669529663688f;
    yy[i * QK_TURBO3 + tid] = ggml_cuda_cast<dst_t>(shmem[tid] * inv_sqrt32 * signs[tid]);
}

/* ===== turbo4 dequantize block kernel ===== */

template<typename dst_t>
static __global__ void dequantize_block_turbo4_0(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i = blockIdx.x;
    const int tid = threadIdx.x;

    if (i * QK_TURBO4 >= k) return;

    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const block_turbo4_0 & xb = x[i];

    const float norm  = __half2float(xb.norm);
    const float rnorm = __half2float(xb.rnorm);
    const float qjl_scale = TURBO_QJL_CONST / 128.0f * rnorm;

    const int base = tid * 4;
    dst_t * out = y + i * QK_TURBO4 + base;

    for (int jj = 0; jj < 4; jj++) {
        const int j = base + jj;
        const int bit_offset = j * 3;
        const int byte_idx = bit_offset / 8;
        const int bit_pos = bit_offset % 8;
        uint16_t raw = (uint16_t)xb.qs[byte_idx];
        if (bit_pos > 5 && byte_idx + 1 < QK_TURBO4 * 3 / 8) {
            raw |= (uint16_t)xb.qs[byte_idx + 1] << 8;
        }
        const int idx = (raw >> bit_pos) & 0x7;
        const float centroid_val = turbo_centroid_3bit(idx);
        const int sign_bit = (xb.signs[j / 8] >> (j % 8)) & 1;
        const float qjl_val = sign_bit ? qjl_scale : -qjl_scale;
        out[jj] = ggml_cuda_cast<dst_t>((centroid_val + qjl_val) * norm);
    }
}

/* ===== Row dequantize launchers ===== */

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_TURBO3;
    dequantize_block_turbo3_0<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    dequantize_block_turbo4_0<<<nb, 32, 0, stream>>>(vx, y, k);
}

/* Explicit template instantiations */
template void dequantize_row_turbo3_0_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo3_0_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo3_0_cuda<nv_bfloat16>(const void * vx, nv_bfloat16 * y, const int64_t k, cudaStream_t stream);

template void dequantize_row_turbo4_0_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo4_0_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo4_0_cuda<nv_bfloat16>(const void * vx, nv_bfloat16 * y, const int64_t k, cudaStream_t stream);
