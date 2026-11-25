#pragma once

#include "common.h"

// ggml-common.h must already be included: it provides block_tq3_1s and
// block_tq4_1s. dequantize.h includes it, so a source file that uses this
// header must include dequantize.h first.
#ifndef QK_TQ3_0
#error "tq.h must be included after ggml-common.h"
#endif

// ============================================================================
// TQ3_1S / TQ4_1S: WHT-rotated weight quantization constants and dequant
// ============================================================================

constant float tq3_centroids[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f,
};

constant float tq4_centroids[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};

constant float tq3_signs[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

constant float tq3_inv_sqrt32 = 0.17677669529663688f; // 1/sqrt(32)

// Unpack 32 x 3-bit indices from 12 bytes (4 groups of 3 bytes = 8 indices each)
inline void tq3_unpack_indices(device const uint8_t * qs, thread uint8_t * indices) {
    for (int g = 0; g < 4; g++) {
        device const uint8_t * qp = qs + g * 3;
        thread uint8_t * idx = indices + g * 8;
        idx[0] =  qp[0]       & 7;
        idx[1] = (qp[0] >> 3) & 7;
        idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
        idx[3] = (qp[1] >> 1) & 7;
        idx[4] = (qp[1] >> 4) & 7;
        idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
        idx[6] = (qp[2] >> 2) & 7;
        idx[7] = (qp[2] >> 5) & 7;
    }
}

// Unpack 8 indices from a single 3-byte group
inline void tq3_unpack_group(device const uint8_t * qp, thread uint8_t * idx) {
    idx[0] =  qp[0]       & 7;
    idx[1] = (qp[0] >> 3) & 7;
    idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
    idx[3] = (qp[1] >> 1) & 7;
    idx[4] = (qp[1] >> 4) & 7;
    idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
    idx[6] = (qp[2] >> 2) & 7;
    idx[7] = (qp[2] >> 5) & 7;
}

// In-place inverse RHT: WHT butterfly + normalize + undo sign flips
inline void tq3_rht_inverse(thread float * buf) {
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < 32; i++) {
        buf[i] *= tq3_inv_sqrt32 * tq3_signs[i];
    }
}

// TQ3_1S dequant: 4x4 with inverse RHT
template <typename type4x4>
void dequantize_tq3_1s(device const block_tq3_1s * xb, short il, thread type4x4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    uint8_t indices[32];
    tq3_unpack_indices(xb->qs, indices);

    float buf[32];
    for (int j = 0; j < 16; j++) buf[j]  = tq3_centroids[indices[j]]  * d0;
    for (int j = 16; j < 32; j++) buf[j] = tq3_centroids[indices[j]] * d1;

    tq3_rht_inverse(buf);

    float4x4 reg_f;
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            reg_f[i][k] = buf[il * 16 + i * 4 + k];
        }
    }
    reg = (type4x4) reg_f;
}

// TQ3_1S dequant: t4 with inverse RHT
template <typename type4>
void dequantize_tq3_1s_t4(device const block_tq3_1s * xb, short il, thread type4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    uint8_t indices[32];
    tq3_unpack_indices(xb->qs, indices);

    float buf[32];
    for (int j = 0; j < 16; j++) buf[j]  = tq3_centroids[indices[j]]  * d0;
    for (int j = 16; j < 32; j++) buf[j] = tq3_centroids[indices[j]] * d1;

    tq3_rht_inverse(buf);

    int off = il * 4;
    reg = (type4) float4(buf[off], buf[off + 1], buf[off + 2], buf[off + 3]);
}

// TQ3_1S: No-RHT dequant for 4x4 (pre-rotated activations)
template <typename type4x4>
void dequantize_tq3_1s_rotated(device const block_tq3_1s * xb, short il, thread type4x4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    uint8_t indices[32];
    tq3_unpack_indices(xb->qs, indices);

    float4x4 reg_f;
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            int j = il * 16 + i * 4 + k;
            float d = (j < 16) ? d0 : d1;
            reg_f[i][k] = tq3_centroids[indices[j]] * d;
        }
    }
    reg = (type4x4) reg_f;
}

// TQ3_1S: No-RHT dequant for t4 (pre-rotated activations)
template <typename type4>
void dequantize_tq3_1s_rotated_t4(device const block_tq3_1s * xb, short il, thread type4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    const short g = il / 2;
    const short g_off = (il % 2) * 4;

    uint8_t idx[8];
    tq3_unpack_group(xb->qs + g * 3, idx);

    const float d = (il < 4) ? d0 : d1;
    reg = (type4) float4(
        tq3_centroids[idx[g_off + 0]] * d,
        tq3_centroids[idx[g_off + 1]] * d,
        tq3_centroids[idx[g_off + 2]] * d,
        tq3_centroids[idx[g_off + 3]] * d
    );
}

// TQ4_1S dequant: 4x4 with inverse RHT
template <typename type4x4>
void dequantize_tq4_1s(device const block_tq4_1s * xb, short il, thread type4x4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    float buf[32];
    for (int j = 0; j < 32; j++) {
        uint8_t idx = (xb->qs[j / 2] >> ((j & 1) * 4)) & 0xF;
        float d = (j < 16) ? d0 : d1;
        buf[j] = tq4_centroids[idx] * d;
    }

    tq3_rht_inverse(buf);

    float4x4 reg_f;
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            reg_f[i][k] = buf[il * 16 + i * 4 + k];
        }
    }
    reg = (type4x4) reg_f;
}

// TQ4_1S dequant: t4 with inverse RHT
template <typename type4>
void dequantize_tq4_1s_t4(device const block_tq4_1s * xb, short il, thread type4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    float buf[32];
    for (int j = 0; j < 32; j++) {
        uint8_t idx = (xb->qs[j / 2] >> ((j & 1) * 4)) & 0xF;
        float d = (j < 16) ? d0 : d1;
        buf[j] = tq4_centroids[idx] * d;
    }

    tq3_rht_inverse(buf);

    int off = il * 4;
    reg = (type4) float4(buf[off], buf[off + 1], buf[off + 2], buf[off + 3]);
}

// TQ4_1S: No-RHT dequant for 4x4 (pre-rotated activations)
template <typename type4x4>
void dequantize_tq4_1s_rotated(device const block_tq4_1s * xb, short il, thread type4x4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    float4x4 reg_f;
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            int j = il * 16 + i * 4 + k;
            uint8_t idx = (xb->qs[j / 2] >> ((j & 1) * 4)) & 0xF;
            float d = (j < 16) ? d0 : d1;
            reg_f[i][k] = tq4_centroids[idx] * d;
        }
    }
    reg = (type4x4) reg_f;
}

// TQ4_1S: No-RHT dequant for t4 (pre-rotated activations)
template <typename type4>
void dequantize_tq4_1s_rotated_t4(device const block_tq4_1s * xb, short il, thread type4 & reg) {
    const float d0 = float(xb->d0);
    const float d1 = float(xb->d1);

    int off = il * 4;
    float4 vals;
    for (int i = 0; i < 4; i++) {
        int j = off + i;
        uint8_t idx = (xb->qs[j / 2] >> ((j & 1) * 4)) & 0xF;
        float d = (j < 16) ? d0 : d1;
        vals[i] = tq4_centroids[idx] * d;
    }
    reg = (type4) vals;
}
