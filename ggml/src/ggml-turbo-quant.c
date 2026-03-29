/*
 * TurboQuant: KV cache compression
 * Based on Lucien2468/Ollama-TurboQuant-Integration
 *
 * turbo3 (GGML_TYPE_TURBO3_0): 3-bit uniform quantization
 *   - 32-element blocks, 14 bytes each (3.5 bits/value, 4.6x compression)
 *   - Simple: round(x/d) clamped to [-4,3], stored with +4 offset as [0,7]
 *   - dp4a compatible packing: 8 elements per 3 bytes
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

/* ===== turbo3 quantize ===== */

/* WHT128 sign array (seed=42, matches CUDA constant memory) */
static const int8_t turbo3_signs128[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

/* WHT128 forward (CPU) */
static void turbo3_wht128_forward(float * x) {
    for (int j = 0; j < 128; j++) x[j] *= turbo3_signs128[j];
    for (int step = 1; step < 128; step <<= 1) {
        for (int i = 0; i < 128; i += step * 2) {
            for (int j = i; j < i + step; j++) {
                float a = x[j], b = x[j + step];
                x[j] = a + b; x[j + step] = a - b;
            }
        }
    }
    const float s = 0.08838834764831845f;  /* 1/sqrt(128) */
    for (int j = 0; j < 128; j++) x[j] *= s;
}

/* WHT128 inverse (CPU) */
static void turbo3_wht128_inverse(float * x) {
    for (int step = 1; step < 128; step <<= 1) {
        for (int i = 0; i < 128; i += step * 2) {
            for (int j = i; j < i + step; j++) {
                float a = x[j], b = x[j + step];
                x[j] = a + b; x[j + step] = a - b;
            }
        }
    }
    const float s = 0.08838834764831845f;
    for (int j = 0; j < 128; j++) x[j] *= s * turbo3_signs128[j];
}

/* Lloyd-Max centroids and boundaries for N(0, 1/128) */
static const float turbo3_cpu_centroids[8] = {
    -0.18165533f, -0.11483849f, -0.06487426f, -0.02106513f,
     0.02106513f,  0.06487426f,  0.11483849f,  0.18165533f
};
static const float turbo3_cpu_boundaries[7] = {
    -0.14824691f, -0.08985637f, -0.04296969f, 0.0f,
     0.04296969f,  0.08985637f,  0.14824691f
};

static int turbo3_nearest_centroid(float val) {
    int idx = 0;
    idx += (val >= turbo3_cpu_boundaries[0]);
    idx += (val >= turbo3_cpu_boundaries[1]);
    idx += (val >= turbo3_cpu_boundaries[2]);
    idx += (val >= turbo3_cpu_boundaries[3]);
    idx += (val >= turbo3_cpu_boundaries[4]);
    idx += (val >= turbo3_cpu_boundaries[5]);
    idx += (val >= turbo3_cpu_boundaries[6]);
    return idx;
}

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    const int64_t n_groups = k / 128;

    for (int64_t g = 0; g < n_groups; g++) {
        /* Step 1: Extract norm and normalize */
        float norm_sq = 0.0f;
        float xn[128];
        for (int j = 0; j < 128; j++) { xn[j] = x[g*128 + j]; norm_sq += xn[j]*xn[j]; }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
        for (int j = 0; j < 128; j++) xn[j] *= inv_norm;

        /* Step 2: WHT128 rotation */
        turbo3_wht128_forward(xn);

        /* Step 3: Lloyd-Max quantize into 4 blocks */
        for (int blk = 0; blk < 4; blk++) {
            block_turbo3_0 * yb = &y[g*4 + blk];
            yb->d = GGML_FP32_TO_FP16(grp_norm);  /* store norm in each block */

            for (int group = 0; group < 4; group++) {
                uint8_t v[8];
                for (int j = 0; j < 8; j++) {
                    v[j] = (uint8_t)turbo3_nearest_centroid(xn[blk*32 + group*8 + j]);
                }
                yb->qs[group*3+0] = (v[0]&7)|((v[1]&7)<<3)|((v[2]&3)<<6);
                yb->qs[group*3+1] = ((v[2]&4)>>2)|((v[3]&7)<<1)|((v[4]&7)<<4)|((v[5]&1)<<7);
                yb->qs[group*3+2] = ((v[5]&6)>>1)|((v[6]&7)<<2)|((v[7]&7)<<5);
            }
        }
    }
}

void quantize_row_turbo3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_turbo3_0_ref(x, (block_turbo3_0 *)vy, k);
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    const int64_t n_groups = k / 128;

    for (int64_t g = 0; g < n_groups; g++) {
        /* Get norm from first block */
        const float grp_norm = GGML_FP16_TO_FP32(x[g*4].d);

        /* Unpack 4 blocks: centroid[idx] (rotated space) */
        float rotated[128];
        for (int blk = 0; blk < 4; blk++) {
            const block_turbo3_0 * xb = &x[g*4 + blk];
            for (int group = 0; group < 4; group++) {
                const uint8_t * qs = &xb->qs[group*3];
                uint8_t v[8];
                v[0] = (qs[0] & 0x07);
                v[1] = (qs[0] >> 3) & 0x07;
                v[2] = ((qs[0] >> 6) & 0x03) | ((qs[1] & 0x01) << 2);
                v[3] = (qs[1] >> 1) & 0x07;
                v[4] = (qs[1] >> 4) & 0x07;
                v[5] = ((qs[1] >> 7) & 0x01) | ((qs[2] & 0x03) << 1);
                v[6] = (qs[2] >> 2) & 0x07;
                v[7] = (qs[2] >> 5) & 0x07;

                for (int j = 0; j < 8; j++) {
                    rotated[blk*32 + group*8 + j] = turbo3_cpu_centroids[v[j]];
                }
            }
        }

        /* Inverse WHT128 + rescale by norm */
        turbo3_wht128_inverse(rotated);

        for (int j = 0; j < 128; j++) {
            y[g*128 + j] = rotated[j] * grp_norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    const size_t row_size = ggml_row_size(GGML_TYPE_TURBO3_0, n_per_row);
    block_turbo3_0 * dy = (block_turbo3_0 *)dst;
    for (int64_t i = 0; i < nrows; i++) {
        quantize_row_turbo3_0_ref(src + i*n_per_row, dy, n_per_row);
        dy += n_per_row / QK_TURBO3;
    }
    return nrows * row_size;
}

/* ===== turbo4 stubs (not yet implemented in this approach) ===== */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    GGML_UNUSED(x); assert(k % QK_TURBO4 == 0);
    memset(y, 0, (k / QK_TURBO4) * sizeof(block_turbo4_0));
}

void quantize_row_turbo4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_turbo4_0_ref(x, (block_turbo4_0 *)vy, k);
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    GGML_UNUSED(x); assert(k % QK_TURBO4 == 0);
    memset(y, 0, k * sizeof(float));
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix); GGML_UNUSED(src);
    return nrows * ggml_row_size(GGML_TYPE_TURBO4_0, n_per_row);
}
