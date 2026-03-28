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

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int64_t nb = k / QK_TURBO3;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            amax = fmaxf(amax, fabsf(x[i*32 + j]));
        }

        const float d = amax / 4.0f;  // 3-bit range: [-4, 3]
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);

        for (int group = 0; group < 4; group++) {
            const int64_t base = i*32 + group*8;
            uint8_t v[8];
            for (int j = 0; j < 8; j++) {
                const int q = (int)roundf(x[base + j] * id);
                const int c = q < -4 ? -4 : (q > 3 ? 3 : q);
                v[j] = (uint8_t)(c + 4);  // offset to [0, 7]
            }
            // Pack 8 elements into 3 bytes
            y[i].qs[group*3 + 0] = (v[0] & 0x07) | ((v[1] & 0x07) << 3) | ((v[2] & 0x03) << 6);
            y[i].qs[group*3 + 1] = ((v[2] & 0x04) >> 2) | ((v[3] & 0x07) << 1) | ((v[4] & 0x07) << 4) | ((v[5] & 0x01) << 7);
            y[i].qs[group*3 + 2] = ((v[5] & 0x06) >> 1) | ((v[6] & 0x07) << 2) | ((v[7] & 0x07) << 5);
        }
    }
}

void quantize_row_turbo3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_turbo3_0_ref(x, (block_turbo3_0 *)vy, k);
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int64_t nb = k / QK_TURBO3;

    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int group = 0; group < 4; group++) {
            const uint8_t * qs = &x[i].qs[group*3];
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
                y[i*32 + group*8 + j] = ((int)v[j] - 4) * d;
            }
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
