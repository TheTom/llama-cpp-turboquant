#pragma once

#include "common.cuh"

/*
 * TurboQuant CUDA backend — KV cache compression via PolarQuant + WHT rotation
 * Port of the Metal implementation from ggml-metal.metal
 *
 * turbo3: 3-bit MSE-only PolarQuant, block_size=32, rotation_group=128, 4.6x compression
 * turbo4: 3-bit PolarQuant + 1-bit QJL, block_size=128, 3.8x compression
 */

/* ===== Constants ===== */

#define QR_TURBO3  2   /* 2 elements per dequantize call (like q4_0) */
#define QK_TURBO3  32
#define QK_TURBO3_GROUP 128
#define QK_TURBO4  128

#define TURBO_QJL_CONST  1.2533141373155003f  /* sqrt(pi/2) */
#define TURBO_INV_SQRT_128 0.08838834764831845f  /* 1/sqrt(128) */

/* WHT sign arrays in __constant__ memory.
 * Stored as int8 (+1/-1) to minimize constant cache usage (512B total for all 4 arrays).
 * Values from turbo-wht.h, generated with seed=42 (rotation) and seed=1042 (QJL). */

/* Rotation signs (seed=42) */
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

/* QJL signs (seed=1042) */
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

/* ===== 3-bit centroids (register LUT) ===== */
/* Per Metal code comment: "On CUDA, register LUT works better" */

static __device__ __forceinline__ float turbo_centroid_3bit(int idx) {
    constexpr float c[8] = {
        -0.190685f, -0.117832f, -0.065717f, -0.021460f,
         0.021460f,  0.065717f,  0.117832f,  0.190685f
    };
    return c[idx];
}

/* 3-bit midpoints — branchless nearest centroid search */
static __device__ __forceinline__ int turbo_nearest_3bit(float val) {
    constexpr float mid[7] = {
        -0.154259f, -0.091775f, -0.043589f, 0.0f,
         0.043589f,  0.091775f,  0.154259f
    };
    int idx = 0;
    idx += (val >= mid[0]);
    idx += (val >= mid[1]);
    idx += (val >= mid[2]);
    idx += (val >= mid[3]);
    idx += (val >= mid[4]);
    idx += (val >= mid[5]);
    idx += (val >= mid[6]);
    return idx;
}

/* ===== Fast Walsh-Hadamard Transform ===== */
/* In-place on 128 floats in registers. O(n log n) = 896 butterfly ops. */

static __device__ __forceinline__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; i++) {
        x[i] *= TURBO_INV_SQRT_128;
    }
}

/* Forward rotation: signs1 * FWHT * signs2 */
static __device__ __forceinline__ void turbo_rotate_forward(
        float * x, const int8_t * __restrict__ s1, const int8_t * __restrict__ s2) {
    for (int i = 0; i < 128; i++) x[i] *= (float)s1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= (float)s2[i];
}

/* turbo3 dequantize (float2 interface) is defined in dequantize.cuh */

/* ===== turbo3 quantize — 128-element group ===== */
/* Processes 128 floats: WHT rotation, quantize into 4 blocks of 32, norm correction.
 * Called from the custom set_rows kernel. */

static __device__ __forceinline__ void turbo3_quantize_group(
        const float * __restrict__ src, block_turbo3_0 * __restrict__ dst) {
    /* Step 1: Compute group norm and normalize */
    float norm_sq = 0.0f;
    float x[128];
    for (int j = 0; j < 128; j++) {
        x[j] = src[j];
        norm_sq += x[j] * x[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;

    /* Step 2: WHT rotation */
    turbo_rotate_forward(x, turbo_wht_signs1_d, turbo_wht_signs2_d);

    /* Step 3: Quantize into 4 blocks + accumulate reconstruction norm */
    float recon_norm_sq = 0.0f;

    for (int b = 0; b < 4; b++) {
        block_turbo3_0 blk;
        blk.norm = __float2half(0.0f);
        memset(blk.qs, 0, QK_TURBO3 / 4);
        memset(blk.signs, 0, QK_TURBO3 / 8);

        for (int j = 0; j < QK_TURBO3; j++) {
            int idx = turbo_nearest_3bit(x[b * QK_TURBO3 + j]);

            /* Pack lower 2 bits into qs[] */
            blk.qs[j / 4] |= (uint8_t)((idx & 0x3) << ((j % 4) * 2));

            /* Pack upper 1 bit into signs[] */
            if (idx & 0x4) {
                blk.signs[j / 8] |= (uint8_t)(1 << (j % 8));
            }

            float c = turbo_centroid_3bit(idx);
            recon_norm_sq += c * c;
        }

        dst[b] = blk;
    }

    /* Step 4: Norm correction — store grp_norm / ||centroid_vector|| */
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    half corrected_h = __float2half(corrected);
    for (int b = 0; b < 4; b++) {
        dst[b].norm = corrected_h;
    }
}

/* ===== turbo4 quantize — 128-element block ===== */
/* Full TurboQuant: WHT rotation, 3-bit PolarQuant, residual, QJL 1-bit signs. */

static __device__ __forceinline__ void turbo4_quantize_block(
        const float * __restrict__ src, block_turbo4_0 * __restrict__ dst) {
    /* Step 1: Norm + normalize */
    float norm_sq = 0.0f;
    float x[128];
    float normalized[128];
    for (int j = 0; j < 128; j++) {
        x[j] = src[j];
        norm_sq += x[j] * x[j];
    }
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    for (int j = 0; j < 128; j++) {
        x[j] *= inv_norm;
        normalized[j] = x[j];
    }

    /* Step 2: WHT rotation */
    turbo_rotate_forward(x, turbo_wht_signs1_d, turbo_wht_signs2_d);

    /* Step 3: 3-bit quantization + pack indices */
    memset(dst->qs, 0, QK_TURBO4 * 3 / 8);
    memset(dst->signs, 0, QK_TURBO4 / 8);

    float recon[128];
    for (int j = 0; j < 128; j++) {
        int idx = turbo_nearest_3bit(x[j]);
        recon[j] = turbo_centroid_3bit(idx);

        /* 3-bit packing across byte boundaries */
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos = bit_offset % 8;
        dst->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
        if (bit_pos > 5 && byte_idx + 1 < QK_TURBO4 * 3 / 8) {
            dst->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
        }
    }

    /* Step 4: Residual (in normalized space, not rotated — matches Metal) */
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        x[j] = normalized[j] - recon[j];
        rnorm_sq += x[j] * x[j];
    }

    dst->norm  = __float2half(norm);
    dst->rnorm = __float2half(sqrtf(rnorm_sq));

    /* Step 5: QJL — WHT with separate signs, store sign bits */
    turbo_rotate_forward(x, turbo_qjl_signs1_d, turbo_qjl_signs2_d);
    for (int i = 0; i < 128; i++) {
        if (x[i] >= 0.0f) {
            dst->signs[i / 8] |= (uint8_t)(1 << (i % 8));
        }
    }
}

/* ===== Declarations for turbo-quant.cu launchers ===== */

void turbo_quant_init_cuda(cudaStream_t stream);

/* TURBO_WHT graph-level op */
void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

/* set_rows launchers */
template<typename idx_t>
void set_rows_turbo3_cuda(
    const float * src0_d, const idx_t * src1_d, block_turbo3_0 * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

template<typename idx_t>
void set_rows_turbo4_cuda(
    const float * src0_d, const idx_t * src1_d, block_turbo4_0 * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);
