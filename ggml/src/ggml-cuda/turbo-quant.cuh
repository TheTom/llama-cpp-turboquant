#pragma once

#include "common.cuh"

/*
 * TurboQuant CUDA backend — 3-bit uniform KV cache compression
 * Based on Lucien2468/Ollama-TurboQuant-Integration
 *
 * turbo3: 3-bit uniform quantization, 32-element blocks, 14 bytes, 4.6x compression
 *   - round(x/d) clamped to [-4,3], packed 8 per 3 bytes
 *   - dp4a compatible for hardware-accelerated dot product
 *   - No rotation, no codebooks — simple and fast
 */

#define QR_TURBO3  2    /* 2 elements per dequantize call */
#define QI_TURBO3  3    /* 12 bytes / sizeof(int) = 3 ints per block */
#define QK_TURBO4  256  /* turbo4 block size (stub) */
#define TURBO3_GROUP 128 /* WHT rotation group = head_dim */

/* WHT128 sign array (seed=42) */
__constant__ static int8_t turbo3_wht128_signs[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

/* WHT128 forward rotation (device, in-place on 128 floats) */
static __device__ __forceinline__ void turbo3_wht128_forward(float * x) {
    #pragma unroll
    for (int j = 0; j < 128; j++) x[j] *= (float)turbo3_wht128_signs[j];
    #pragma unroll
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    const float s = 0.08838834764831845f;  // 1/sqrt(128)
    #pragma unroll
    for (int j = 0; j < 128; j++) x[j] *= s;
}

/* Lloyd-Max 3-bit centroids for N(0, 1/128), precomputed */
__device__ static const float turbo3_centroids[8] = {
    -0.18165533f, -0.11483849f, -0.06487426f, -0.02106513f,
     0.02106513f,  0.06487426f,  0.11483849f,  0.18165533f
};

/* Lloyd-Max boundaries (midpoints between centroids) */
__device__ static const float turbo3_boundaries[7] = {
    -0.14824691f, -0.08985637f, -0.04296969f, 0.0f,
     0.04296969f,  0.08985637f,  0.14824691f
};

/* Find nearest centroid index using boundaries */
static __device__ __forceinline__ int turbo3_nearest_centroid(float val) {
    int idx = 0;
    idx += (val >= turbo3_boundaries[0]);
    idx += (val >= turbo3_boundaries[1]);
    idx += (val >= turbo3_boundaries[2]);
    idx += (val >= turbo3_boundaries[3]);
    idx += (val >= turbo3_boundaries[4]);
    idx += (val >= turbo3_boundaries[5]);
    idx += (val >= turbo3_boundaries[6]);
    return idx;
}

/* Full paper algorithm: norm → normalize → WHT128 → Lloyd-Max quantize */
static __device__ void turbo3_quantize_group_128(
        const float * __restrict__ src, block_turbo3_0 * __restrict__ dst) {
    /* Step 1: Extract norm and normalize */
    float norm_sq = 0.0f;
    float x[128];
    for (int j = 0; j < 128; j++) { x[j] = src[j]; norm_sq += x[j]*x[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;

    /* Step 2: WHT128 rotation (makes coordinates ~ N(0, 1/128)) */
    turbo3_wht128_forward(x);

    /* Step 3: Lloyd-Max quantize each coordinate */
    for (int blk = 0; blk < 4; blk++) {
        /* Store group norm in each block's d field */
        dst[blk].d = __float2half(grp_norm);

        for (int g = 0; g < 4; g++) {
            uint8_t v[8];
            for (int j = 0; j < 8; j++) {
                v[j] = (uint8_t)turbo3_nearest_centroid(x[blk*32 + g*8 + j]);
            }
            dst[blk].qs[g*3+0] = (v[0]&7)|((v[1]&7)<<3)|((v[2]&3)<<6);
            dst[blk].qs[g*3+1] = ((v[2]&4)>>2)|((v[3]&7)<<1)|((v[4]&7)<<4)|((v[5]&1)<<7);
            dst[blk].qs[g*3+2] = ((v[5]&6)>>1)|((v[6]&7)<<2)|((v[7]&7)<<5);
        }
    }
}

/* ===== Declarations ===== */

void turbo_quant_init_cuda(cudaStream_t stream);
void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo3_0_no_wht_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

void turbo3_inverse_wht32_cuda(float * data, int64_t n_elements, cudaStream_t stream);

/* Custom set_rows for turbo3 with WHT128 rotation groups */
template<typename idx_t>
void turbo3_set_rows_cuda(
    const float * src0_d, const idx_t * src1_d, block_turbo3_0 * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);
