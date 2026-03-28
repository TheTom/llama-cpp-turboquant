#pragma once

#include "common.cuh"

/*
 * TurboQuant CUDA backend — KV cache compression
 *
 * turbo3: 3-bit PolarQuant, block_size=32, per-block WHT32, 4.6x compression
 * turbo4: 3-bit PolarQuant + 1-bit QJL, block_size=128, WHT128, 3.8x compression
 */

#define QR_TURBO3  2
#define QR_TURBO4  2
#define QK_TURBO4  256

#define TURBO_QJL_CONST  1.2533141373155003f  /* sqrt(pi/2) */
#define TURBO_INV_SQRT_128 0.08838834764831845f

/* turbo4 8-entry uniform polar grid (all positive, sign handled separately) */
static __device__ __forceinline__ float turbo4_polar_grid(int idx) {
    constexpr float g[8] = {
        1.0f/16.0f, 3.0f/16.0f, 5.0f/16.0f, 7.0f/16.0f,
        9.0f/16.0f, 11.0f/16.0f, 13.0f/16.0f, 15.0f/16.0f
    };
    return g[idx];
}

/* 3-bit scaled centroids */
static __device__ __forceinline__ float turbo_centroid_3bit(int idx) {
    constexpr float c[8] = {
        -2.1573f, -1.3336f, -0.7434f, -0.2428f,
         0.2428f,  0.7434f,  1.3336f,  2.1573f
    };
    return c[idx];
}

/* turbo4 nearest centroid (scaled) */
static __device__ __forceinline__ int turbo4_nearest(float val) {
    if      (val < -1.7455f) return 0;
    else if (val < -1.0385f) return 1;
    else if (val < -0.4931f) return 2;
    else if (val <  0.0f)    return 3;
    else if (val <  0.4931f) return 4;
    else if (val <  1.0385f) return 5;
    else if (val <  1.7455f) return 6;
    else                      return 7;
}

/* ===== WHT128 sign arrays for turbo4 (rotation seed=42, QJL seed=1042) ===== */

__constant__ static int8_t turbo_wht_signs1_d[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

__constant__ static int8_t turbo_wht_signs2_d[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

__constant__ static int8_t turbo_qjl_signs1_d[128] = {
     1,-1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,
     1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1,
     1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1,
    -1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1,
    -1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1,
    -1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1
};

__constant__ static int8_t turbo_qjl_signs2_d[128] = {
     1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1,
    -1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1,
    -1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
     1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1, 1,
    -1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,
     1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1,
     1,-1,-1, 1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,
    -1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1
};

/* ===== WHT128 transform for turbo4 ===== */

static __device__ __forceinline__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h * 2)
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
    for (int i = 0; i < 128; i++) x[i] *= TURBO_INV_SQRT_128;
}

static __device__ __forceinline__ void turbo_rotate_forward_128(
        float * x, const int8_t * __restrict__ s1, const int8_t * __restrict__ s2) {
    for (int i = 0; i < 128; i++) x[i] *= (float)s1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= (float)s2[i];
}

/* ===== Declarations ===== */

void turbo_quant_init_cuda(cudaStream_t stream);
void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo3_0_no_wht_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

void turbo3_inverse_wht32_cuda(float * data, int64_t n_elements, cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);
