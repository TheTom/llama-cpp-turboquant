/*
 * TurboQuant: KV cache compression via PolarQuant + WHT rotation
 * Based on: animehacker/llama-turboquant reference implementation
 *
 * turbo3 (GGML_TYPE_TURBO3_0): 3-bit PolarQuant with per-block WHT32
 *   - 32-element blocks, 14 bytes each (3.5 bits/value, 4.6x compression)
 *   - WHT rotation fused into quantize/dequant (no external rotation matrices)
 * turbo4 (GGML_TYPE_TURBO4_0): 3-bit PolarQuant + 1-bit QJL residual
 *   - 128-element blocks, 68 bytes each (4.25 bits/value, 3.8x compression)
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

/* ===== turbo3 constants ===== */

static const float turbo3_centroids[8] = {
    -2.1573f, -1.3336f, -0.7434f, -0.2428f,
     0.2428f,  0.7434f,  1.3336f,  2.1573f
};

static const int8_t turbo3_signs[32] = {
    +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
    +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1
};

/* ===== turbo3 WHT32 ===== */

static void turbo3_wht32_forward(float * x) {
    for (int j = 0; j < 32; j++) x[j] *= turbo3_signs[j];
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step * 2) {
            for (int j = i; j < i + step; j++) {
                float a = x[j], b = x[j + step];
                x[j] = a + b; x[j + step] = a - b;
            }
        }
    }
    const float s = 0.17677669529663688f;  /* 1/sqrt(32) */
    for (int j = 0; j < 32; j++) x[j] *= s;
}

static void turbo3_wht32_inverse(float * x) {
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step * 2) {
            for (int j = i; j < i + step; j++) {
                float a = x[j], b = x[j + step];
                x[j] = a + b; x[j + step] = a - b;
            }
        }
    }
    const float s = 0.17677669529663688f;
    for (int j = 0; j < 32; j++) x[j] *= s * turbo3_signs[j];
}

/* ===== turbo3 quantize/dequantize ===== */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int64_t nb = k / QK_TURBO3;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TURBO3;

        float rotated[QK_TURBO3];
        for (int j = 0; j < QK_TURBO3; j++) rotated[j] = xb[j];
        turbo3_wht32_forward(rotated);

        float amax = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) {
            float av = fabsf(rotated[j]);
            if (av > amax) amax = av;
        }

        const float d = amax / 2.1573f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        y[i].gamma = GGML_FP32_TO_FP16(d);

        memset(y[i].qs, 0, sizeof(y[i].qs));
        memset(y[i].qr, 0, sizeof(y[i].qr));

        for (int j = 0; j < QK_TURBO3; j++) {
            float xn = rotated[j] * id;
            int idx;
            if      (xn < -1.7455f) { idx = 0; }
            else if (xn < -1.0385f) { idx = 1; }
            else if (xn < -0.4931f) { idx = 2; }
            else if (xn <  0.0f)    { idx = 3; }
            else if (xn <  0.4931f) { idx = 4; }
            else if (xn <  1.0385f) { idx = 5; }
            else if (xn <  1.7455f) { idx = 6; }
            else                     { idx = 7; }

            y[i].qs[j / 4] |= ((idx & 3) << (2 * (j % 4)));
            y[i].qr[j / 8] |= (((idx >> 2) & 1) << (j % 8));
        }
    }
}

void quantize_row_turbo3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_turbo3_0_ref(x, (block_turbo3_0 *)vy, k);
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int64_t nb = k / QK_TURBO3;

    for (int64_t i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].gamma);

        float rotated[QK_TURBO3];
        for (int j = 0; j < QK_TURBO3; j++) {
            const int low2 = (x[i].qs[j / 4] >> (2 * (j % 4))) & 3;
            const int hi1  = (x[i].qr[j / 8] >> (j % 8)) & 1;
            const int idx  = low2 | (hi1 << 2);
            rotated[j] = d * turbo3_centroids[idx];
        }

        turbo3_wht32_inverse(rotated);

        for (int j = 0; j < QK_TURBO3; j++) {
            y[i * QK_TURBO3 + j] = rotated[j];
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    const size_t row_size = ggml_row_size(GGML_TYPE_TURBO3_0, n_per_row);
    quantize_row_turbo3_0_ref(src, (block_turbo3_0 *)dst, nrows * n_per_row);
    return nrows * row_size;
}

/* ===== turbo4: PolarQuant (3-bit angle grid + 1-bit QJL sign) ===== */

static const float turbo4_grid[8] = {
    1.0f/16.0f, 3.0f/16.0f, 5.0f/16.0f, 7.0f/16.0f,
    9.0f/16.0f, 11.0f/16.0f, 13.0f/16.0f, 15.0f/16.0f
};

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int64_t nb = k / QK_TURBO4;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TURBO4;

        float amax = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) {
            float av = fabsf(xb[j]);
            if (av > amax) amax = av;
        }
        y[i].d = GGML_FP32_TO_FP16(amax);
        float inv_d = amax > 0.0f ? 1.0f / amax : 0.0f;

        memset(y[i].al, 0, QK_TURBO4 / 4);
        memset(y[i].ah, 0, QK_TURBO4 / 8);
        memset(y[i].signs, 0, QK_TURBO4 / 8);

        for (int j = 0; j < QK_TURBO4; j++) {
            float normalized = xb[j] * inv_d;
            int best_idx = 0, best_sign = 0;
            float best_err = 1e30f;
            for (int g = 0; g < 8; g++) {
                float gv = turbo4_grid[g];
                float e_pos = (normalized - gv) * (normalized - gv);
                float e_neg = (normalized + gv) * (normalized + gv);
                if (e_pos < best_err) { best_err = e_pos; best_idx = g; best_sign = 0; }
                if (e_neg < best_err) { best_err = e_neg; best_idx = g; best_sign = 1; }
            }
            y[i].al[j / 4] |= (uint8_t)((best_idx & 0x3) << ((j % 4) * 2));
            if (best_idx & 0x4) y[i].ah[j / 8] |= (uint8_t)(1 << (j % 8));
            if (best_sign)       y[i].signs[j / 8] |= (uint8_t)(1 << (j % 8));
        }
    }
}

void quantize_row_turbo4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_turbo4_0_ref(x, (block_turbo4_0 *)vy, k);
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int64_t nb = k / QK_TURBO4;

    for (int64_t i = 0; i < nb; ++i) {
        float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK_TURBO4; j++) {
            uint8_t lo  = (x[i].al[j / 4] >> ((j % 4) * 2)) & 0x3;
            uint8_t hi  = (x[i].ah[j / 8] >> (j % 8)) & 0x1;
            uint8_t idx = lo | (hi << 2);
            uint8_t sign = (x[i].signs[j / 8] >> (j % 8)) & 0x1;
            y[i * QK_TURBO4 + j] = d * turbo4_grid[idx] * (1.0f - 2.0f * sign);
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    const size_t row_size = ggml_row_size(GGML_TYPE_TURBO4_0, n_per_row);
    quantize_row_turbo4_0_ref(src, (block_turbo4_0 *)dst, nrows * n_per_row);
    return nrows * row_size;
}
