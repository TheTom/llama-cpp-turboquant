#pragma once

#include "common.cuh"

/*
 * TurboQuant CUDA backend — KV cache compression
 * Based on animehacker/llama-turboquant reference implementation.
 *
 * turbo3: 3-bit PolarQuant, block_size=32, per-block WHT32, 4.6x compression
 *   - WHT rotation fused into quantize (set_rows) and vec_dot (MMVQ)
 *   - No graph-level rotation ops needed
 * turbo4: 3-bit PolarQuant + 1-bit QJL, block_size=128, 3.8x compression
 */

#define QR_TURBO3  2   /* 2 elements per dequantize call */
#define QK_TURBO4  128

#define TURBO_QJL_CONST  1.2533141373155003f  /* sqrt(pi/2) */

/* 3-bit centroids (register LUT) — scaled centroids for gamma-based dequant */
static __device__ __forceinline__ float turbo_centroid_3bit(int idx) {
    constexpr float c[8] = {
        -2.1573f, -1.3336f, -0.7434f, -0.2428f,
         0.2428f,  0.7434f,  1.3336f,  2.1573f
    };
    return c[idx];
}

/* ===== Declarations ===== */

void turbo_quant_init_cuda(cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);
